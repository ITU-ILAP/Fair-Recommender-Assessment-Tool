"""
run_nfcf_grid.py
================
Runs NFCF on MovieLens-1M subsets with a full hyperparameter grid and writes
all fairness + accuracy metrics to a single XLSX file.

Grid (16 combinations per subset):
  learning_rate  : [0.0005, 0.001]
  dropout        : [0.1, 0.2]
  fair_weight    : [0.05, 0.1]
  embedding_size : [32, 64]

Subsets: sample_1 … sample_100  (dataset_v2/ml-1M/URM_subsets_filtered/)
Sensitive attribute: age
Evaluation: Top-5 (NDCG@5, Recall@5, Hit@5, MRR@5 + all fairness metrics)
No model saving: saved=False

Output: nfcf_grid_results.xlsx  (same directory as this script)
  - Appends new rows, replacing any existing rows for the same subset.
"""

import os
import sys
import time
import logging
import warnings
import itertools

import pandas as pd

# ── ensure we always run from the project root ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ── silence everything before importing RecBole ───────────────────────────────
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_NAME  = "ml-1M"
DATA_PATH     = os.path.join("dataset_v2", DATASET_NAME)
SUBSET_FOLDER = "URM_subsets_filtered"
CONFIG_FILE   = os.path.join(SCRIPT_DIR, "test.yaml")
OUTPUT_XLSX   = os.path.join(SCRIPT_DIR, "nfcf_grid_results.xlsx")

SENSITIVE_ATTR = "age"
N_SUBSETS      = 100
N_COMBOS       = 16
TOTAL_RUNS     = N_SUBSETS * N_COMBOS

# ── hyperparameter grid ───────────────────────────────────────────────────────
PARAM_GRID = {
    "learning_rate":  [0.0005, 0.001],
    "dropout":        [0.1, 0.2],
    "fair_weight":    [0.05, 0.1],
    "embedding_size": [32, 64],
}

def grid_combinations():
    keys, values = list(PARAM_GRID.keys()), list(PARAM_GRID.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ── XLSX helpers ──────────────────────────────────────────────────────────────
def load_xlsx(path):
    if os.path.exists(path):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception:
            pass
    return pd.DataFrame()

def save_xlsx(df, path):
    df.to_excel(path, index=False, engine="openpyxl")


# ── RecBole helpers ───────────────────────────────────────────────────────────

# NOTE: Config._set_default_parameters() appends the dataset name to data_path
# and data_path_inter at __init__ time. Override those keys AFTER construction.
def _fix_paths(config, subset_name):
    config["data_path"]       = DATA_PATH
    config["data_path_inter"] = os.path.join(DATA_PATH, SUBSET_FOLDER, f"{subset_name}.inter")

def _base_overrides():
    return {
        "topk":           [5],
        "valid_metric":   "NDCG@5",
        "sst_attr_list":  [SENSITIVE_ATTR],
        "load_col": {
            "inter": ["user_id", "item_id", "rating"],
            "user":  ["user_id", SENSITIVE_ATTR, "location"],
        },
        "show_progress":  False,
        "saved":          False,
        # Disable SST embedding saving: _save_sst_embed() tries to torch.load
        # the checkpoint file which doesn't exist when saved=False.
        "save_sst_embed": False,
    }

def prepare_subset_data(subset_name):
    """Create train/valid/test split once per subset (reused across all combos)."""
    d = _base_overrides()
    d["embedding_size"] = 64  # placeholder; data loading is model-param-independent
    config = Config(model="NFCF", dataset=DATASET_NAME,
                    config_file_list=[CONFIG_FILE], config_dict=d)
    _fix_paths(config, subset_name)
    dataset = create_dataset(config)
    return data_preparation(config, dataset)

def run_single(subset_name, combo, train_data, valid_data, test_data):
    """Train + evaluate one hyperparameter combination. Returns metrics dict."""
    d = _base_overrides()
    d.update(combo)
    config = Config(model="NFCF", dataset=DATASET_NAME,
                    config_file_list=[CONFIG_FILE], config_dict=d)
    _fix_paths(config, subset_name)

    init_seed(config["seed"], config["reproducibility"])

    model_obj = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer   = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_obj)

    trainer.fit(train_data, valid_data, saved=False, show_progress=False)
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)

    return {str(k): float(v) for k, v in test_result.items()}


# ── progress display ──────────────────────────────────────────────────────────
def _progress(run_idx, subset_name, elapsed, error=None):
    """Overwrite the current terminal line with a compact progress update."""
    pct = run_idx / TOTAL_RUNS * 100
    eta = (elapsed / run_idx) * (TOTAL_RUNS - run_idx) if run_idx else 0
    tag = f"  [ERROR: {error}]" if error else ""
    line = (f"\r{run_idx}/{TOTAL_RUNS}  {pct:.1f}%  |  {subset_name}"
            f"  |  elapsed {elapsed/60:.1f}m  ETA ~{eta/60:.0f}m{tag}   ")
    sys.stdout.write(line)
    sys.stdout.flush()


# ── already-done lookup ───────────────────────────────────────────────────────
def _done_keys(df):
    """Return a set of (subset, lr, dropout, fair_weight, emb) tuples already in the XLSX."""
    if df.empty or "subset" not in df.columns:
        return set()
    needed = ["subset", "learning_rate", "dropout", "fair_weight", "embedding_size"]
    if not all(c in df.columns for c in needed):
        return set()
    return set(df[needed].itertuples(index=False, name=None))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"NFCF grid search  —  {N_SUBSETS} subsets × {N_COMBOS} combos = {TOTAL_RUNS} runs")
    print(f"Output → {OUTPUT_XLSX}\n")

    combos       = list(grid_combinations())
    global_start = time.time()

    # Count already-completed runs for accurate progress display
    df_existing  = load_xlsx(OUTPUT_XLSX)
    done         = _done_keys(df_existing)
    run_idx      = len(done)   # start counter from whatever is already done

    if run_idx:
        print(f"Resuming — {run_idx} runs already done, {TOTAL_RUNS - run_idx} remaining.\n")

    for sidx in range(1, N_SUBSETS + 1):
        subset_name  = f"sample_{sidx}"
        subset_combos = [c for c in combos
                         if (subset_name, c["learning_rate"], c["dropout"],
                             c["fair_weight"], c["embedding_size"]) not in done]

        if not subset_combos:
            continue  # already counted in len(done) at startup

        # Load data once for this subset (only if there's work to do)
        try:
            train_data, valid_data, test_data = prepare_subset_data(subset_name)
        except Exception as exc:
            run_idx += len(subset_combos)
            _progress(run_idx, subset_name, time.time() - global_start,
                      error=f"data load failed: {exc}")
            continue

        for combo in subset_combos:
            run_idx += 1
            _progress(run_idx, subset_name, time.time() - global_start)

            try:
                metrics = run_single(subset_name, combo, train_data, valid_data, test_data)
            except Exception as exc:
                metrics = {"error": str(exc)}
                _progress(run_idx, subset_name, time.time() - global_start, error=str(exc))

            row = {
                "subset":         subset_name,
                "learning_rate":  combo["learning_rate"],
                "dropout":        combo["dropout"],
                "fair_weight":    combo["fair_weight"],
                "embedding_size": combo["embedding_size"],
            }
            row.update(metrics)

            # Append to in-memory df and flush to disk — no re-read needed
            df_existing = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
            save_xlsx(df_existing, OUTPUT_XLSX)
            done.add((subset_name, combo["learning_rate"], combo["dropout"],
                      combo["fair_weight"], combo["embedding_size"]))

    elapsed_total = time.time() - global_start
    print(f"\n\nDone — {TOTAL_RUNS} runs in {elapsed_total/60:.1f} min. "
          f"Results saved to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
