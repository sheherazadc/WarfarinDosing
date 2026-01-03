from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    Standard softmax but ignores padded values. 
    We set masked values to -inf so they effectively contribute zero to the final weights.
    """
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    # nan_to_num prevents crashes if a whole row is masked out (all -inf)
    scores = torch.nan_to_num(scores, neginf=-1e9)
    return torch.softmax(scores, dim=dim)


class TimeLSTM(nn.Module):
    """
    A specialized LSTM that considers not just 'what' happened, but 'when'.
    It uses a temporal attention mechanism that penalizes older events 
    based on how much time has passed since they occurred.
    """
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        attention: bool = True,
        dt_index: int = -1,
        use_pack: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = bool(attention)
        self.dt_index = dt_index
        self.use_pack = bool(use_pack)

        # We concatenate features and time info before feeding them into the LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim + time_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        if self.attention:
            # Traditional Bahdanau-style attention components
            self.attn_W = nn.Linear(hidden_size, hidden_size, bias=True)
            self.attn_v = nn.Linear(hidden_size, 1, bias=False)
            
            # This is the 'decay' part: it learns how much weight to drop as time passes
            self.attn_decay = nn.Linear(1, 1, bias=True)
            nn.init.zeros_(self.attn_decay.weight)
            nn.init.zeros_(self.attn_decay.bias)

            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.norm = nn.Identity()
            self.head = nn.Identity()

        self.out = nn.Linear(hidden_size, 1)

    @staticmethod
    def _last_valid(H: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        In a batch of varying sequence lengths, we can't just take the final index.
        This helper finds the actual 'last' non-padded step for each sequence.
        """
        if mask is None:
            return H[:, -1, :]
        lengths = mask.sum(dim=1).clamp(min=1)
        # We generate indices to grab the last hidden state for each patient/sequence
        idx = (lengths - 1).view(-1, 1, 1).expand(H.size(0), 1, H.size(-1))
        return H.gather(1, idx).squeeze(1)

    def _encode(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Passes the combined feature/time vector through the LSTM layers.
        """
        X = torch.cat([x, t], dim=-1)
        
        if self.use_pack and (mask is not None):
            # Packing avoids doing math on 'zeros' in padded areas, making it faster
            lengths = mask.sum(dim=1).to(torch.int64).cpu()
            
            # NOTE: PyTorch's pack_padded_sequence needs sequences sorted by length
            lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
            X_sorted = X.index_select(0, sort_idx)
            
            packed = nn.utils.rnn.pack_padded_sequence(X_sorted, lengths_sorted, batch_first=True)
            H_packed, _ = self.lstm(packed)
            H_sorted, _ = nn.utils.rnn.pad_packed_sequence(H_packed, batch_first=True, total_length=X.size(1))
            
            # Put the sequences back in the original batch order
            inv_idx = torch.empty_like(sort_idx)
            inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
            H = H_sorted.index_select(0, inv_idx)
        else:
            H, _ = self.lstm(X) 
        return H

    def _temporal_attention_with_decay(
        self,
        H: torch.Tensor,
        t_abs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates attention weights by combining feature importance 
        with a time-based penalty.
        """
        # Get baseline importance scores for each hidden state
        e = self.attn_v(torch.tanh(self.attn_W(H))).squeeze(-1) 

        # Calculate time elapsed since each step relative to the final step
        # t_last - t_current
        tau_rel = (t_abs[:, [-1]] - t_abs).clamp_min(0.0).unsqueeze(-1)
        
        # Softplus ensures the penalty is always positive
        penalty = F.softplus(self.attn_decay(tau_rel)).squeeze(-1)
        
        # Subtracting the penalty in log-space (before softmax) 
        # effectively diminishes the weight of older steps.
        e = e - penalty

        alpha = masked_softmax(e, mask=mask, dim=-1)
        # Weighted sum of hidden states based on the scores we just calculated
        ctx = torch.sum(H * alpha.unsqueeze(-1), dim=1)
        return ctx, alpha

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_attention: bool = False,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Processes a batch of sequences.
        x: features [Batch, Time, Features]
        t: time info [Batch, Time, TimeChannels]
        mask: 1 for real data, 0 for padding [Batch, Time]
        """
        if self.time_dim == 0:
            raise ValueError("TimeLSTMFast requires at least one absolute time channel in `t`.")
        
        # Pull the absolute time used for the decay calculation
        t_abs = t[..., self.dt_index]

        # Generate hidden states for all time steps
        H = self._encode(x, t, mask)
        h_last = self._last_valid(H, mask)

        # Apply the attention mechanism
        if self.attention:
            ctx, a_time = self._temporal_attention_with_decay(H, t_abs, mask)
            # Residual connection: we combine the 'context' (summary of history) 
            # with the 'last hidden state' (most recent info)
            rep = self.norm(ctx + h_last)
            rep = self.head(rep)
        else:
            a_time = None
            rep = self.norm(h_last)

        # Final linear projection to get a single prediction value
        y = self.out(rep).squeeze(-1)

        if return_attention:
            B, T = x.size(0), x.size(1)
            if a_time is None:
                a_time = torch.zeros(B, T, device=x.device, dtype=x.dtype)
            return y, a_time, None
        return y