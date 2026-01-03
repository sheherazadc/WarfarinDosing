from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Ensures that padded 'empty' data doesn't get any attention. 
    We fill the scores with negative infinity so the softmax function ignores them.
    """
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = torch.nan_to_num(scores, neginf=-1e9)
    return torch.softmax(scores, dim=dim)


class FeatureAttention(nn.Module):
    """
    Feature-level attention: This layer looks at all input variables (F) 
    and decides which ones are most useful for the prediction at each step.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_layer = nn.Linear(input_dim, input_dim, bias=True)
        # We start with very small weights so the model begins by 
        # treating all features roughly equally.
        nn.init.xavier_uniform_(self.attention_layer.weight, gain=0.01)
        nn.init.zeros_(self.attention_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [Batch, Time, Features]
        feature_scores = self.attention_layer(x)  
        
        # Softmax here identifies the 'importance' of each feature
        alpha_f = torch.softmax(feature_scores, dim=-1)  
        
        # We scale the original features by their importance scores
        x_weighted = x * alpha_f  
        
        # We take the average across the timeline to help humans 
        # see which features mattered most overall.
        avg_attention = alpha_f.mean(dim=1)  
        
        return x_weighted, avg_attention


class TimeAttention(nn.Module):
    """
    Time-level attention: This layer decides which past visits are most relevant.
    It includes an exponential decay so that older information naturally 
    carries less weight unless it is extremely significant.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_t = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_t = nn.Linear(hidden_size, 1, bias=False)
        
        # This linear layer learns the 'decay rate'.
        self.decay_transform = nn.Linear(1, 1, bias=True)
        nn.init.constant_(self.decay_transform.weight, -0.1) 
        nn.init.constant_(self.decay_transform.bias, 0.0)

    def forward(
        self, 
        h: torch.Tensor,  
        time_intervals: torch.Tensor,  
        mask: Optional[torch.Tensor] = None  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Standard attention score calculation
        time_scores = self.v_t(torch.tanh(self.W_t(h))).squeeze(-1)  
        
        # Calculate decay: closer visits (smaller delta_t) have factors closer to 1.0
        delta_t_transformed = self.decay_transform(time_intervals.unsqueeze(-1)).squeeze(-1)
        decay_factors = torch.exp(-torch.abs(delta_t_transformed))
        
        # We dampen the importance of older steps
        decayed_scores = time_scores * decay_factors  
        
        alpha_t = masked_softmax(decayed_scores, mask, dim=-1)  
        
        # Create a single 'context vector' representing the patient's history
        context = torch.sum(h * alpha_t.unsqueeze(-1), dim=1)  
        
        return context, alpha_t


class WarfarinNet(nn.Module):
    """
    The main architecture. 
    Flow: Features -> Feature Attention -> LSTM -> Time Attention -> Output.
    """
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        dt_index: int = -1, 
        use_pack: bool = False,
        attention: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_size = hidden_size
        self.dt_index = dt_index
        self.use_pack = use_pack
        self.attention = attention
        
        if self.attention:
            self.feature_attention = FeatureAttention(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim + time_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        
        if self.attention:
            self.time_attention = TimeAttention(hidden_size)
        
        # Normalization and final prediction layers
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_size, 1)

    def _encode_with_packing(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Groups data by sequence length to skip unnecessary calculations on 
        the zero-padding used for shorter patient histories.
        """
        if self.use_pack and mask is not None:
            lengths = mask.sum(dim=1).to(torch.int64).cpu()
            lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
            x_sorted = x.index_select(0, sort_idx)
            
            packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lengths_sorted, batch_first=True)
            h_packed, _ = self.lstm(packed)
            h_sorted, _ = nn.utils.rnn.pad_packed_sequence(h_packed, batch_first=True, total_length=x.size(1))
            
            inv_idx = torch.empty_like(sort_idx)
            inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
            h = h_sorted.index_select(0, inv_idx)
        else:
            h, _ = self.lstm(x)
        
        return h

    def _get_last_valid(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Grabs the hidden state from the patient's most recent visit."""
        if mask is None:
            return h[:, -1, :]
        
        lengths = mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(h.size(0), 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        # Identify the 'absolute time' column to calculate how far back events were
        t_abs = t[..., self.dt_index]  
        
        feat_attention = None
        time_attention = None
        
        # Feature attention: decides which inputs matter most
        if self.attention:
            x_weighted, feat_attention = self.feature_attention(x)
        else:
            x_weighted = x
        
        # Combine the weighted features with the raw time metadata
        lstm_input = torch.cat([x_weighted, t], dim=-1)  
        
        # LSTM: tracks the sequential patterns in the history
        h = self._encode_with_packing(lstm_input, mask)  
        
        # Time attention: summarizes history while decaying old info
        if self.attention:
            # We calculate intervals relative to the very last recorded timestamp
            time_intervals = (t_abs[:, [-1]] - t_abs).clamp_min(0.0)  
            context, time_attention = self.time_attention(h, time_intervals, mask)
        else:
            # Fallback if attention is off: just use the latest data point
            context = self._get_last_valid(h, mask)
        
        # Prediction head: generates the final recommended dose
        rep = self.norm(context)
        rep = self.head(rep)
        y = self.output(rep).squeeze(-1)  
        
        # Return results along with maps for interpretability if requested
        if return_attention:
            B, T = x.size(0), x.size(1)
            if time_attention is None:
                time_attention = torch.zeros(B, T, device=x.device, dtype=x.dtype)
            if feat_attention is None:
                feat_attention = torch.ones(B, self.input_dim, device=x.device, dtype=x.dtype) / self.input_dim
            return y, time_attention, feat_attention
        
        return y

# Labels used to refer to this model elsewhere in the project
WarfarinNetLSTM = WarfarinNet
NETLSTM = WarfarinNet