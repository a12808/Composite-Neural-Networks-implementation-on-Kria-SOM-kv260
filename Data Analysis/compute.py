"""
kria_gpu_inference_analysis.py

Complete, modular script for analyzing inference results on KV260 and GPU.
Updated to calculate Time Breakdown, Throughput, and Power per test (not per model).
Includes debug outputs, metric calculations, and bar plot generation.
"""

import os
import re
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# ----------------- Configuration -----------------
RESULT_PATTERN = re.compile(r'^test_([^_]+)_(kv260|gpu)_results.*\.csv$', re.IGNORECASE)
POWER_PATTERN = re.compile(r'^test_([^_]+)_(kv260|gpu)_power_results.*\.csv$', re.IGNORECASE)


# ----------------- File Handling -----------------
def find_test_files(input_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:

    files = os.listdir(input_dir)
    tests = {}

    for f in files:

        fm = RESULT_PATTERN.match(f)
        pm = POWER_PATTERN.match(f)

        if fm:
            testname, platform = fm.group(1).upper(), fm.group(2).lower()
            tests.setdefault(testname, {}).setdefault(platform, {})['result'] = os.path.join(input_dir, f)
        if pm:
            testname, platform = pm.group(1).upper(), pm.group(2).lower()
            tests.setdefault(testname, {}).setdefault(platform, {})['power'] = os.path.join(input_dir, f)

    print('[DEBUG] Found test files:')
    for t, d in tests.items():
        print(f'  Test {t}: {d}')

    return tests


def load_result_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    print(f"[DEBUG] Loaded results {os.path.basename(path)} shape={df.shape}")

    return df


def load_power_csv(path: str) -> pd.DataFrame:

    df = pd.read_csv(path)

    if 'power_watts' not in df.columns:
        for c in df.columns:
            if 'power' in c.lower():
                df = df.rename(columns={c: 'power_watts'})
                break

    print(f"[DEBUG] Loaded power {os.path.basename(path)} shape={df.shape}")

    return df


# ----------------- Metrics Calculation -----------------
def detect_models(df: pd.DataFrame) -> List[int]:

    model_ids = set()

    for c in df.columns:
        m = re.match(r'pred_lbl_(\d+)', c.lower())
        if m:
            model_ids.add(int(m.group(1)))

    return sorted(model_ids)


def compute_accuracy_per_model(df: pd.DataFrame) -> Dict[int, Tuple[float, int, int]]:

    gt_col = next((c for c in df.columns if c.lower().startswith('gt') or 'gt_lbl' in c.lower()), None)

    if gt_col is None:
        model_ids = detect_models(df)
        return {k: (np.nan, 0, len(df)) for k in model_ids}

    model_acc = {}

    for mid in detect_models(df):

        pred_col = next((c for c in df.columns if re.match(rf'pred_lbl_{mid}(?:$|_)', c.lower())), None)

        if pred_col is None:
            pred_col = next((c for c in df.columns if c.lower().startswith('pred_lbl')), None)
        if pred_col is None:
            model_acc[mid] = (np.nan, 0, len(df))
            continue

        y_true = df[gt_col].astype(str).str.strip()
        y_pred = df[pred_col].astype(str).str.strip()

        try:
            acc = float(accuracy_score(y_true, y_pred))
            correct = int((y_true == y_pred).sum())
        except:
            acc, correct = np.nan, 0

        model_acc[mid] = acc, correct, len(df)

        print(f"[DEBUG] Model {mid} accuracy: {acc:.4f} ({correct}/{len(df)})")

    return model_acc


def compute_time_breakdown(df: pd.DataFrame) -> Dict[str, float]:

    time_cols = [c for c in df.columns if 'time' in c.lower()]
    stages = {}

    for c in time_cols:
        mean_time = pd.to_numeric(df[c], errors='coerce').fillna(0).mean()
        stages[c] = float(mean_time)

    print(f"[DEBUG] Time breakdown per test: {stages}")

    return stages


def compute_throughput(df: pd.DataFrame, time_breakdown: Dict[str, float], n_models: int = 1) -> float:

    total_time_per_cycle = sum(time_breakdown.values())
    throughput = float(n_models / total_time_per_cycle) if total_time_per_cycle > 0 else np.nan
    print(f"[DEBUG] Throughput: {throughput:.6f} imgs/s (para {n_models} modelos)")

    return throughput


def compute_power_energy(power_df: pd.DataFrame, total_duration_s: float) -> Tuple[float, float]:

    if power_df is None or power_df.empty or total_duration_s <= 0:
        return np.nan, np.nan

    mean_power = float(power_df['power_watts'].mean())
    energy = mean_power * total_duration_s
    print(f"[DEBUG] Mean power: {mean_power:.3f} W, Total energy: {energy:.3f} J")

    return mean_power, energy


def compute_efficiencies(acc: Dict[int, Tuple[float, int]], total_energy: float,
                         throughput: float, mean_power: float) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute multiple efficiency metrics per model
    :param acc: dict {model_id: (accuracy, n_correct)}
    :param total_energy: total energy used for the test (J)
    :param throughput: images per second for the test
    :param mean_power: mean power in W
    :return: dict with efficiencies
    """
    eff = {}

    # Accuracy per Joule
    acc_per_joule = {mid: (a[0]/total_energy if total_energy > 0 else np.nan) for mid, a in acc.items()}
    eff['acc_per_joule'] = acc_per_joule

    # Correct images per Joule
    correct_imgs_per_joule = {mid: (a[1]/total_energy if total_energy > 0 else np.nan) for mid, a in acc.items()}
    eff['correct_imgs_per_joule'] = correct_imgs_per_joule

    # Throughput per Watt
    throughput_per_watt = throughput / mean_power if mean_power > 0 else np.nan
    eff['throughput_per_watt'] = throughput_per_watt

    # Relative efficiency (normalized to max acc_per_joule for this test)
    max_eff = max(acc_per_joule.values()) if acc_per_joule else 1
    relative_efficiency = {mid: v / max_eff * 100 for mid, v in acc_per_joule.items()} if max_eff > 0 else {}
    eff['relative_efficiency'] = relative_efficiency

    return eff


def compute_metrics_for_file(result_path: str, power_path: str, n_models: int = 1) -> Dict:

    try:

        df = load_result_csv(result_path)
        power_df = load_power_csv(power_path) if power_path and os.path.exists(power_path) else None

        # Accuracy per model (only metric per model)
        acc = compute_accuracy_per_model(df)

        # Time breakdown per test (test-level, not per model)
        tb = compute_time_breakdown(df)

        # Throughput per test
        th = compute_throughput(df, tb, n_models=n_models)

        # Total duration based on time breakdown
        total_duration = sum(tb.values()) * len(df)

        # Power & energy
        mean_power, total_energy = compute_power_energy(power_df, total_duration)

        # Compute all efficiencies
        eff = compute_efficiencies(acc, total_energy, th, mean_power)

        return {
            'model_ids': list(acc.keys()),
            'accuracy': acc,
            'time_breakdown': tb,
            'throughput': th,
            'mean_power_w': mean_power,
            'energy_j': total_energy,
            'efficiency': eff
        }

    except Exception as e:
        print(f'[DEBUG] Error processing {result_path}: {e}')
        raise


# ----------------- Main -----------------
def compute_data(input_dir: str):

    tests = find_test_files(input_dir)

    all_results = {}

    for tname, platforms in tests.items():

        all_results[tname] = {}

        if tname.startswith(('B', 'C')):
            n_models = 2
        else:
            n_models = 1

        for platform, files in platforms.items():

            if 'result' not in files or 'power' not in files:
                raise RuntimeError(f'Missing files for {tname} {platform}')

            print(f'\n[DEBUG] Computing metrics for {tname} {platform}')
            metrics = compute_metrics_for_file(files['result'], files['power'], n_models=n_models)
            all_results[tname][platform] = metrics

    print('\n[DEBUG] All metrics computed successfully.')
    return all_results
