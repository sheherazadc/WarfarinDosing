import os
import numpy as np
import glob

def gather_attention_outputs(model_name="WarfarinNetLSTM", base_dir="NET_outputs_nested_cv"):
    """
    Scans the results directory to find the 'best' version of each trial.
    It combines the predictions and attention maps from every fold into one 
    compressed master file.
    """
    model_dir = os.path.join("NET_outputs_nested_cv", "WarfarinNetLSTM")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    # We look for the 'best' snapshot from each trial to avoid using 
    # data from under-performing or early-stopped epochs.
    files = sorted(glob.glob(os.path.join(model_dir, "attn_outputs_trial_*_best.npz")))
    if not files:
        raise RuntimeError(f"No best attention .npz files found in {model_dir}")

    all_preds, all_targets, all_time, all_feat = [], [], [], []

    for f in files:
        data = np.load(f, allow_pickle=True)
        # We only grab the keys if they actually exist in the file to prevent errors
        if "preds" in data:
            all_preds.append(data["preds"])
        if "targets" in data:
            all_targets.append(data["targets"])
        
        # Check that the attention arrays aren't empty before adding them
        if "attn_time" in data and data["attn_time"].size > 0:
            all_time.append(data["attn_time"])
        if "attn_feat" in data and data["attn_feat"].size > 0:
            all_feat.append(data["attn_feat"])

    # We stack all the individual arrays into single, large matrices
    preds = np.concatenate(all_preds, axis=0) if all_preds else None
    targets = np.concatenate(all_targets, axis=0) if all_targets else None
    attn_time = np.concatenate(all_time, axis=0) if all_time else None
    attn_feat = np.concatenate(all_feat, axis=0) if all_feat else None

    # Save the consolidated data back to the model folder
    out_path = os.path.join(model_dir, f"gathered_best_attention_{model_name}.npz")
    np.savez_compressed(out_path,
                        preds=preds,
                        targets=targets,
                        attn_time=attn_time,
                        attn_feat=attn_feat)

    print(f"Gathered best attention outputs saved to: {out_path}")


if __name__ == "__main__":
    # The default model name can be overridden here
    gather_attention_outputs("WarfarinNet")