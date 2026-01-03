**Files**: `time_aware_lstm.py`, `warfarinnet_lstm.py`

# Neural Network Architecture

The project features a custom "Time-Aware" architecture designed to mimic clinical reasoning.

### Key Innovations
1.  **Time-Aware LSTM**: Incorporates the time delta ($\Delta t$) between visits directly into the hidden state transitions.
2.  **Feature Attention**: Identifies which clinical variables (e.g., age, weight, previous INR) are most relevant at each specific step.
3.  **Temporal Attention with Decay**: Applies an exponential decay penalty to past visits. This forces the model to prioritize recent data while still "remembering" significant events from the distant past.



### Usage
The model is automatically initialised by the training scripts, but can be used standalone:
```python
from warfarinnet_lstm import WarfarinNetLSTM
model = WarfarinNetLSTM(input_dim=18, time_dim=1, hidden_size=64)
