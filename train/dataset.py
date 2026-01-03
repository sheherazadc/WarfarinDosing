import torch
from torch.utils.data import Dataset
import numpy as np

class WarfarinDataset(Dataset):
    """
    Converts a standard table into a sequence-based dataset.
    It groups data by patient (subject_id) and creates sliding windows 
    so the model can look at 'W' previous visits to predict the next INR.
    """
    def __init__(self, X, y, time, group_ids=None, sample_size=None, window_size=1, original_indices=None, allow_padding=False): 
        assert len(X) == len(y) == len(time), "X, y, time must have same length"
        self.window_size = int(window_size)
        self.group_ids = group_ids
        
        # We track original indices so we can map predictions back to the 
        # specific rows in the original CSV later for analysis.
        if original_indices is not None:
            self.original_indices = np.asarray(original_indices)
        else:
            self.original_indices = np.arange(len(X))
        
        if sample_size is not None:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X, y, time = X[idx], y[idx], time[idx]
            if group_ids is not None:
                self.group_ids = np.asarray(group_ids)[idx]
            self.original_indices = self.original_indices[idx]
        
        # Simple case: if window is 1, it acts like a normal feed-forward dataset
        if self.window_size <= 1:
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
            time = torch.tensor(time, dtype=torch.float32).unsqueeze(1)
            y = torch.tensor(y, dtype=torch.float32)
            self.X, self.time, self.y = X, time, y
            self.original_idx_windows = self.original_indices
            self._length = X.shape[0]
            return
        
        # Sliding window logic for LSTMs
        X = np.asarray(X)
        y = np.asarray(y)
        time = np.asarray(time)
        
        X_windows, t_windows, y_windows, group_windows = [], [], [], []
        original_idx_windows = []
        
        if group_ids is None:
            # If no groups are provided, we just slide across the whole dataset
            for start in range(0, len(X) - self.window_size + 1):
                end = start + self.window_size
                X_windows.append(X[start:end])
                t_windows.append(time[start:end])
                y_windows.append(y[end - 1])
                group_windows.append(-1)
                original_idx_windows.append(self.original_indices[end - 1])
        else:
            # Patient-aware windowing: ensures sequences stay within one patient
            group_ids = np.asarray(group_ids)
            # We sort by time to ensure the 'history' the model sees is in the right order
            sort_col = 2 if time.shape[1] >= 3 else 0
            
            for sid in np.unique(group_ids):
                mask = (group_ids == sid)
                Xg, yg, tg = X[mask], y[mask], time[mask]
                orig_indices_g = self.original_indices[mask]
                
                order = np.argsort(tg[:, sort_col])
                Xg, yg, tg = Xg[order], yg[order], tg[order]
                orig_indices_g = orig_indices_g[order]  
                
                if len(Xg) < self.window_size:
                    # If a patient has too few visits, we either skip them or pad the start
                    if not allow_padding:
                        continue 
                    else:
                        # Padding: repeats the first visit to fill the window requirement
                        padding_needed = self.window_size - len(Xg)
                        first_X = np.repeat(Xg[0:1], padding_needed, axis=0)
                        first_t = np.repeat(tg[0:1], padding_needed, axis=0)
                        first_orig = np.repeat(orig_indices_g[0], padding_needed)
                        
                        Xg = np.vstack([first_X, Xg])
                        tg = np.vstack([first_t, tg])
                        orig_indices_g = np.concatenate([first_orig, orig_indices_g])
                    
                        X_windows.append(Xg)
                        t_windows.append(tg)
                        y_windows.append(yg[-1])
                        group_windows.append(sid)
                        original_idx_windows.append(orig_indices_g[-1])
                else:
                    # Create sliding slices of length 'window_size'
                    for start in range(0, len(Xg) - self.window_size + 1):
                        end = start + self.window_size
                        X_windows.append(Xg[start:end])
                        t_windows.append(tg[start:end])
                        y_windows.append(yg[end - 1])
                        group_windows.append(sid)
                        original_idx_windows.append(orig_indices_g[end - 1])
            
        if not X_windows:
            raise ValueError("No windows were created. Try smaller window_size or check data.")
        
        # Convert all window lists into final PyTorch tensors
        self.X = torch.tensor(np.stack(X_windows, axis=0), dtype=torch.float32)      
        self.time = torch.tensor(np.stack(t_windows, axis=0), dtype=torch.float32) 
        self.y = torch.tensor(np.array(y_windows), dtype=torch.float32)        
        self.group_ids = np.array(group_windows)
        self.original_idx_windows = np.array(original_idx_windows)
        
        self._length = self.X.shape[0]
    
    def __getitem__(self, idx):
        """Grabs one specific sequence for the training loop."""
        return {
            "features": self.X[idx],   # Shape: (Window, Features)
            "time": self.time[idx],    # Shape: (Window, TimeFeatures)
            "label": self.y[idx],      # The target INR value
            "group": self.group_ids[idx] if self.group_ids is not None else -1,
            "original_indices": torch.LongTensor([self.original_idx_windows[idx]]) 
        }
    
    def __len__(self):
        """Returns the total number of windows available in the dataset."""
        return self._length