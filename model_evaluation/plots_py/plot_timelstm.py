import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Using a clean visual style for clinical reporting
plt.style.use('seaborn-v0_8-whitegrid')

def _normalize_to_sequences(attn):
    """
    Standardizes different attention tensor shapes into a list of 1D arrays.
    Whether the data is 2D, 3D (multi-head), or 4D, this pulls out the 
    core time-attention weights.
    """
    seqs = []
    if attn.ndim == 1:
        v = np.asarray(attn, dtype=float).ravel()
        if np.isfinite(v).all() and v.size > 0:
            seqs.append(v)
        return seqs

    if attn.ndim == 2:  # [Number of Patients, Time Steps]
        for row in attn:
            v = np.asarray(row, dtype=float).ravel()
            if np.isfinite(v).all() and v.size > 0:
                seqs.append(v)
        return seqs

    if attn.ndim == 3:
        N, A, B = attn.shape
        if A == B:  # If it's a square matrix [N, T, T], we extract the diagonal
            for M in attn:
                try:
                    v = np.diag(M)
                except Exception:
                    # Fallback: grab the last row if diagonal extraction fails
                    v = M[-1, :]
                v = np.asarray(v, dtype=float).ravel()
                if np.isfinite(v).all() and v.size > 0:
                    seqs.append(v)
        else:
            # For non-square 3D tensors, we pick the most relevant 'layer' of attention
            rows = attn[:, -1, :] if B >= A else attn[:, :, -1] 
            for row in rows:
                v = np.asarray(row, dtype=float).ravel()
                if np.isfinite(v).all() and v.size > 0:
                    seqs.append(v)
        return seqs

    if attn.ndim == 4:  # [Batch, Heads, Time, Time]
        # We average across multiple attention heads before processing
        return _normalize_to_sequences(attn.mean(axis=1))

    return seqs

def _find_time_attn_key(npz_obj):
    """
    Flexible search for the attention data inside the NPZ file, 
    since key names can vary between trial versions.
    """
    if "time_attn" in npz_obj.files:
        return "time_attn"
    for k in npz_obj.files:
        kl = k.lower()
        if "attn" in kl and "time" in kl:
            return k
    return None

def plot_average_time_attention_overall(data_dir: str, save_path: str, flip: str = "auto"):
    """
    Main plotting logic: aggregates all sequences, aligns them to 
    the most recent visit, and generates the final line graph.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        print(f"No .npz files found in: {data_dir}")
        return

    seqs = []
    used = 0
    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            key = _find_time_attn_key(z)
            if key is None:
                continue
            seqs.extend(_normalize_to_sequences(z[key]))
            used += 1
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not seqs:
        print("No valid time-attention sequences found.")
        return

    # --- Auto-detect reversed order ---
    # We assume the model usually cares more about the present than the distant past.
    # If the 'head' (start) of the vector is much heavier than the 'tail' (end),
    # the vector is likely ordered newest-to-oldest and needs to be flipped.
    def tail_mean(v, k=3): return float(np.mean(v[-k:])) if len(v) >= k else float(v.mean())
    def head_mean(v, k=3): return float(np.mean(v[:k])) if len(v) >= k else float(v.mean())
    head = np.mean([head_mean(v) for v in seqs])
    tail = np.mean([tail_mean(v) for v in seqs])

    do_flip = False
    if flip == "yes":
        do_flip = True
    elif flip == "no":
        do_flip = False
    else:  # auto-mode logic
        if head > tail * 1.25:
            do_flip = True

    if do_flip:
        # Reverse the sequences so the plot consistently shows past -> present
        seqs = [v[::-1] for v in seqs]
        print("Auto-flip applied (vectors looked newest→oldest).")
    else:
        print("No flip applied (vectors looked oldest→newest).")

    # --- Right-align at t-0 and aggregate ---
    # Since patients have different history lengths, we align them so 
    # 't-0' (the current moment) is always the last column.
    max_len = max(len(s) for s in seqs)
    padded = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        L = len(s)
        padded[i, -L:] = s  

    # Calculate mean and standard deviation ignoring the missing values (NaNs)
    mean_w = np.nanmean(padded, axis=0)
    std_w  = np.nanstd(padded, axis=0)
    counts = np.sum(~np.isnan(padded), axis=0)  

    # Create x-axis labels like t-5, t-4, ..., t-0
    lags = list(range(max_len - 1, -1, -1))
    xlbl = [f"t-{k}" for k in lags]

    # --- Plotting the attention decay ---
    plt.figure(figsize=(8, 6))
    plt.plot(xlbl, mean_w, marker="o", linewidth=2, label="Mean attention", color = 'mediumblue')
    
    # Shade the area representing standard deviation to show variability between patients
    plt.fill_between(xlbl, mean_w - std_w, mean_w + std_w, alpha=0.25, label="±1 SD", color = 'blueviolet')
    
    plt.title("Average Time-Level Attention")
    plt.xlabel("Lag (t-0 = most recent)")
    plt.ylabel("Attention weight")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {save_path}")

    # Output diagnostics to the console to verify data integrity
    print(f"[Diag] Sequences used: {len(seqs)} from {used} NPZ files")
    print(f"[Diag] Mean(head 3)={np.mean(mean_w[:3]):.3f} | Mean(last 3)={np.mean(mean_w[-3:]):.3f}")

def main():
    """Parses command line arguments for flexible directory paths."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="outputs_nested_cv/WarfarinNet",
                    help="Directory containing NPZ attention files.")
    ap.add_argument("--out_dir", type=str, default="figures_time_lstm",
                    help="Directory to save the figure.")
    ap.add_argument("--out_name", type=str, default="overall_avg_time_attention.png",
                    help="Output filename for the plot.")
    ap.add_argument("--flip", type=str, default="auto", choices=["auto", "yes", "no"],
                    help="Force flip newest↔oldest (auto tries to detect).")
    args = ap.parse_args()
    save_path = os.path.join(args.out_dir, args.out_name)
    plot_average_time_attention_overall(args.data_dir, save_path, flip=args.flip)

if __name__ == "__main__":
    main()