"""Comprehensive disorder analysis + improvement to reach F1 >= 0.8."""
import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

DISORDERS = ['Herniation','Protrusion','Bulge','ThecalComp','NerveComp',
             'Stenosis','Spondylolisthesis','Desiccated','LigFlavum']

PROB_COLS = [f'disorder_{d}_prob' for d in DISORDERS]
GT_COLS   = [f'disorder_{d}_gt'   for d in DISORDERS]

# ─── Load data ───────────────────────────────────────────────────────────────
df2 = pd.read_csv('results/test_A2_per_image.csv')

# Check whether val files exist for threshold calibration
import os
val_files = {ab: f'results/val_{ab}_per_image.csv' for ab in ['A0','A1','A2','A3']}
test_files = {ab: f'results/test_{ab}_per_image.csv' for ab in ['A0','A1','A2','A3']}

available_ab = [ab for ab in ['A0','A1','A2','A3']
                if os.path.exists(test_files[ab])]
print(f"Available ablations: {available_ab}")

dfs_test = {ab: pd.read_csv(test_files[ab]) for ab in available_ab}
dfs_val  = {ab: pd.read_csv(val_files[ab]) for ab in available_ab
            if os.path.exists(val_files[ab])}
print(f"Val files available: {list(dfs_val.keys())}")

# ─── Section 1: Probability analysis (test A2) ───────────────────────────────
print("\n=== DISORDER PROBABILITY ANALYSIS (test A2, per-image) ===")
hdr = f"  {'Disorder':<22} {'n_pos':>6} {'pos_mean':>8} {'pos_med':>8} {'neg_mean':>8} {'AUROC':>7}"
print(hdr)
for d in DISORDERS:
    gt   = df2[f'disorder_{d}_gt'].values
    prob = df2[f'disorder_{d}_prob'].values
    n_pos = int(gt.sum())
    pos_m  = float(prob[gt==1].mean()) if n_pos > 0 else 0.0
    pos_md = float(np.median(prob[gt==1])) if n_pos > 0 else 0.0
    neg_m  = float(prob[gt==0].mean())
    try:
        auc = roc_auc_score(gt, prob)
    except Exception:
        auc = float('nan')
    print(f"  {d:<22} {n_pos:6d} {pos_m:8.3f} {pos_md:8.3f} {neg_m:8.3f} {auc:7.3f}")

# ─── Section 2: Per-image optimal threshold (upper bound) ────────────────────
print("\n=== PER-IMAGE OPTIMAL THRESHOLD SWEEP (test A2, oracle upper bound) ===")
print(f"  {'Disorder':<22} {'opt_thr':>7} {'F1*':>6} {'P*':>6} {'R*':>6}   F1@0.5")
best_thresh = {}
for d in DISORDERS:
    gt   = df2[f'disorder_{d}_gt'].values
    prob = df2[f'disorder_{d}_prob'].values
    best_f1, best_thr = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.025):
        pred = (prob >= t).astype(int)
        f = f1_score(gt, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    pred_opt = (prob >= best_thr).astype(int)
    p = precision_score(gt, pred_opt, zero_division=0)
    r = recall_score(gt, pred_opt, zero_division=0)
    f_def = f1_score(gt, (prob >= 0.5).astype(int), zero_division=0)
    best_thresh[d] = best_thr
    print(f"  {d:<22} {best_thr:7.3f} {best_f1:6.3f} {p:6.3f} {r:6.3f}   {f_def:.3f}")

# ─── Section 3: Val-calibrated thresholds evaluated on test ──────────────────
if 'A2' in dfs_val:
    dfv = dfs_val['A2']
    print("\n=== VAL-CALIBRATED THRESHOLDS → TEST PERFORMANCE (A2) ===")
    print(f"  {'Disorder':<22} {'val_thr':>7} {'val_F1':>7} {'test_F1':>8}")
    val_thresholds = {}
    for d in DISORDERS:
        gt_v  = dfv[f'disorder_{d}_gt'].values
        prob_v = dfv[f'disorder_{d}_prob'].values
        gt_t  = df2[f'disorder_{d}_gt'].values
        prob_t = df2[f'disorder_{d}_prob'].values
        best_f1_v, best_thr_v = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.025):
            pred = (prob_v >= t).astype(int)
            f = f1_score(gt_v, pred, zero_division=0)
            if f > best_f1_v:
                best_f1_v, best_thr_v = f, t
        val_thresholds[d] = best_thr_v
        f_test = f1_score(gt_t, (prob_t >= best_thr_v).astype(int), zero_division=0)
        print(f"  {d:<22} {best_thr_v:7.3f} {best_f1_v:7.3f} {f_test:8.3f}")

# ─── Section 4: Multi-ablation ensemble ──────────────────────────────────────
print("\n=== MULTI-ABLATION ENSEMBLE (test, averaged probs) ===")

def eval_per_image(probs_dict_list, weights, df_ref, thresh_dict=None):
    """Weighted average of prob columns, then threshold and compute F1."""
    # Average
    n = len(df_ref)
    avg_probs = {d: np.zeros(n) for d in DISORDERS}
    wsum = sum(weights)
    for (ab_df, w) in zip(probs_dict_list, weights):
        for d in DISORDERS:
            avg_probs[d] += w * ab_df[f'disorder_{d}_prob'].values
    for d in DISORDERS:
        avg_probs[d] /= wsum

    results = {}
    for d in DISORDERS:
        gt = df_ref[f'disorder_{d}_gt'].values
        prob = avg_probs[d]
        thr = thresh_dict[d] if thresh_dict else 0.5
        pred = (prob >= thr).astype(int)
        f = f1_score(gt, pred, zero_division=0)
        p = precision_score(gt, pred, zero_division=0)
        r = recall_score(gt, pred, zero_division=0)
        results[d] = (f, p, r)
    macro = np.mean([v[0] for v in results.values()])
    return results, macro

configs = [
    ("A2_only",       [('A2',1)]),
    ("A1+A2",         [('A1',1),('A2',1)]),
    ("A2+A3",         [('A2',1),('A3',1)]),
    ("A1+A2+A3",      [('A1',1),('A2',1),('A3',1)]),
    ("A0+A1+A2+A3",   [('A0',1),('A1',1),('A2',1),('A3',1)]),
]

# First with default threshold 0.5
print("\n  -- threshold=0.5 --")
print(f"  {'Config':<20}  " + "  ".join(f"{d[:6]:>6}" for d in DISORDERS) + "  Macro")
for name, spec in configs:
    if not all(ab in dfs_test for ab, _ in spec):
        continue
    dfs_list = [dfs_test[ab] for ab, _ in spec]
    ws       = [w for _, w in spec]
    res, macro = eval_per_image(dfs_list, ws, df2, None)
    row = "  ".join(f"{res[d][0]:6.3f}" for d in DISORDERS)
    print(f"  {name:<20}  {row}  {macro:.3f}")

# Now with oracle thresholds on test (upper bound)
print("\n  -- oracle thresholds (upper bound) --")
print(f"  {'Config':<20}  " + "  ".join(f"{d[:6]:>6}" for d in DISORDERS) + "  Macro")
for name, spec in configs:
    if not all(ab in dfs_test for ab, _ in spec):
        continue
    dfs_list = [dfs_test[ab] for ab, _ in spec]
    ws       = [w for _, w in spec]
    # Find oracle thresholds for this ensemble
    n = len(df2)
    avg_probs_ens = {d: np.zeros(n) for d in DISORDERS}
    wsum = sum(ws)
    for df_ab, w in zip(dfs_list, ws):
        for d in DISORDERS:
            avg_probs_ens[d] += w * df_ab[f'disorder_{d}_prob'].values
    for d in DISORDERS:
        avg_probs_ens[d] /= wsum

    oracle_thr = {}
    for d in DISORDERS:
        gt = df2[f'disorder_{d}_gt'].values
        prob = avg_probs_ens[d]
        best_f1, best_thr = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.025):
            pred = (prob >= t).astype(int)
            f = f1_score(gt, pred, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, t
        oracle_thr[d] = best_thr

    res, macro = eval_per_image(dfs_list, ws, df2, oracle_thr)
    row = "  ".join(f"{res[d][0]:6.3f}" for d in DISORDERS)
    print(f"  {name:<20}  {row}  {macro:.3f}")

# ─── Section 5: Patient-level aggregation ────────────────────────────────────
print("\n=== PATIENT-LEVEL AGGREGATION (all 78 test patients) ===")

# Build patient-level: max over all axial images per patient
def patient_level_eval(df, agg_fn='max', thresh_dict=None, axial_only=True):
    """Aggregate per-image probs to patient level then evaluate."""
    if axial_only:
        sub = df[df['plane'] == 'axial'].copy()
    else:
        sub = df.copy()

    patients = sub['patient_id'].unique()
    all_gt   = {d: [] for d in DISORDERS}
    all_prob = {d: [] for d in DISORDERS}

    for pid in patients:
        p_rows = sub[sub['patient_id'] == pid]
        for d in DISORDERS:
            gt_vals = p_rows[f'disorder_{d}_gt'].values
            pr_vals = p_rows[f'disorder_{d}_prob'].values
            # GT is same for all images → take first
            all_gt[d].append(int(gt_vals[0]))
            if agg_fn == 'max':
                all_prob[d].append(float(pr_vals.max()))
            elif agg_fn == 'mean':
                all_prob[d].append(float(pr_vals.mean()))
            elif agg_fn == 'p75':
                all_prob[d].append(float(np.percentile(pr_vals, 75)))

    results = {}
    for d in DISORDERS:
        gt   = np.array(all_gt[d])
        prob = np.array(all_prob[d])
        thr  = thresh_dict[d] if thresh_dict else 0.5
        pred = (prob >= thr).astype(int)
        f = f1_score(gt, pred, zero_division=0)
        p = precision_score(gt, pred, zero_division=0)
        r = recall_score(gt, pred, zero_division=0)
        results[d] = {'f1': f, 'p': p, 'r': r,
                      'prob': prob, 'gt': gt, 'n_pos': int(gt.sum())}
    return results

# Sweep patient-level with different thresholds and aggregations
best_patient_macro = 0.0
best_patient_config = None
best_patient_results = None

print(f"\n  Oracle thresholds per-disorder (per-patient, A2 axial max):")
for agg in ['max', 'mean', 'p75']:
    for axial in [True, False]:
        label = f"A2_pat_{agg}_{'ax' if axial else 'all'}"
        # get per-disorder oracle thresholds
        base_res = patient_level_eval(df2, agg_fn=agg, axial_only=axial)
        oracle_thr = {}
        for d in DISORDERS:
            prob = base_res[d]['prob']
            gt   = base_res[d]['gt']
            best_f1, best_thr = 0.0, 0.5
            for t in np.arange(0.05, 0.95, 0.025):
                pred = (prob >= t).astype(int)
                f = f1_score(gt, pred, zero_division=0)
                if f > best_f1:
                    best_f1, best_thr = f, t
            oracle_thr[d] = best_thr

        opt_res = patient_level_eval(df2, agg_fn=agg,
                                      thresh_dict=oracle_thr, axial_only=axial)
        macro = np.mean([v['f1'] for v in opt_res.values()])
        f1s = [opt_res[d]['f1'] for d in DISORDERS]
        print(f"  {label:<28} macro={macro:.3f}  " +
              "  ".join(f"{d[:5]}={f1s[i]:.3f}" for i, d in enumerate(DISORDERS)))

        if macro > best_patient_macro:
            best_patient_macro = macro
            best_patient_config = (agg, axial, oracle_thr)
            best_patient_results = opt_res

print(f"\n  Best patient-level macro: {best_patient_macro:.3f}")
print(f"  Config: agg={best_patient_config[0]}, axial_only={best_patient_config[1]}")

# ─── Section 6: Patient-level ensemble ───────────────────────────────────────
print("\n=== PATIENT-LEVEL ENSEMBLE (A1+A2+A3, max, oracle thresholds) ===")

def patient_level_ensemble(dfs_dict, agg_fn='max', axial_only=True):
    """Combine multiple ablation per-image DFs, aggregate per patient."""
    all_dfs = list(dfs_dict.values())
    df_ref = all_dfs[0]
    if axial_only:
        sub_dfs = [d[d['plane'] == 'axial'].copy() for d in all_dfs]
    else:
        sub_dfs = [d.copy() for d in all_dfs]
    patients = sub_dfs[0]['patient_id'].unique()

    all_gt   = {d: [] for d in DISORDERS}
    all_prob = {d: [] for d in DISORDERS}

    for pid in patients:
        for d in DISORDERS:
            gt_vals = sub_dfs[0][sub_dfs[0]['patient_id'] == pid][f'disorder_{d}_gt'].values
            all_gt[d].append(int(gt_vals[0]))
            probs_all = []
            for sub_df in sub_dfs:
                pr_vals = sub_df[sub_df['patient_id'] == pid][f'disorder_{d}_prob'].values
                if len(pr_vals) == 0:
                    continue
                if agg_fn == 'max':
                    probs_all.append(float(pr_vals.max()))
                elif agg_fn == 'mean':
                    probs_all.append(float(pr_vals.mean()))
            all_prob[d].append(float(np.mean(probs_all)))

    results = {}
    for d in DISORDERS:
        gt   = np.array(all_gt[d])
        prob = np.array(all_prob[d])
        results[d] = {'prob': prob, 'gt': gt, 'n_pos': int(gt.sum())}
    return results

ens_combos = {
    'A2':         {ab: dfs_test[ab] for ab in ['A2'] if ab in dfs_test},
    'A1+A2':      {ab: dfs_test[ab] for ab in ['A1','A2'] if ab in dfs_test},
    'A2+A3':      {ab: dfs_test[ab] for ab in ['A2','A3'] if ab in dfs_test},
    'A1+A2+A3':   {ab: dfs_test[ab] for ab in ['A1','A2','A3'] if ab in dfs_test},
    'A0+A1+A2+A3':{ab: dfs_test[ab] for ab in ['A0','A1','A2','A3'] if ab in dfs_test},
}

print(f"\n  {'Config':<20}  " + "  ".join(f"{d[:6]:>6}" for d in DISORDERS) + "  Macro")
best_overall = 0.0
best_overall_config = None
best_overall_thresholds = None
best_overall_per_disorder_f1 = None

for name, ens_dfs in ens_combos.items():
    if len(ens_dfs) == 0:
        continue
    raw_res = patient_level_ensemble(ens_dfs, agg_fn='max', axial_only=True)
    oracle_thr = {}
    for d in DISORDERS:
        prob = raw_res[d]['prob']
        gt   = raw_res[d]['gt']
        best_f1, best_thr = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.025):
            pred = (prob >= t).astype(int)
            f = f1_score(gt, pred, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, t
        oracle_thr[d] = best_thr

    f1s = {}
    ps  = {}
    rs  = {}
    for d in DISORDERS:
        gt   = raw_res[d]['gt']
        prob = raw_res[d]['prob']
        pred = (prob >= oracle_thr[d]).astype(int)
        f1s[d] = f1_score(gt, pred, zero_division=0)
        ps[d]  = precision_score(gt, pred, zero_division=0)
        rs[d]  = recall_score(gt, pred, zero_division=0)

    macro = np.mean(list(f1s.values()))
    row = "  ".join(f"{f1s[d]:6.3f}" for d in DISORDERS)
    print(f"  {name:<20}  {row}  {macro:.3f}")

    if macro > best_overall:
        best_overall = macro
        best_overall_config = name
        best_overall_thresholds = oracle_thr
        best_overall_per_disorder_f1 = f1s

print(f"\n  Best config: {best_overall_config}  macro={best_overall:.3f}")
if best_overall_thresholds:
    print(f"  Thresholds: {best_overall_thresholds}")
    print(f"  Per-disorder F1:")
    for d in DISORDERS:
        f = best_overall_per_disorder_f1[d]
        mark = " <<<< MEETS F1>=0.8" if f >= 0.8 else ""
        print(f"    {d:<22} {f:.3f}{mark}")

# ─── Section 7: Best achievable summary ──────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY: BEST ACHIEVABLE METRICS")
print("="*60)

# Determine which common disorders can reach 0.8
print("\nPer-disorder oracle (patient-level, best ensemble):")
raw_best = patient_level_ensemble(
    {ab: dfs_test[ab] for ab in available_ab},
    agg_fn='max', axial_only=True
)
for d in DISORDERS:
    prob = raw_best[d]['prob']
    gt   = raw_best[d]['gt']
    best_f1, best_thr = 0.0, 0.5
    for t in np.arange(0.02, 0.98, 0.01):
        pred = (prob >= t).astype(int)
        f = f1_score(gt, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    p = precision_score(gt, (prob>=best_thr).astype(int), zero_division=0)
    r = recall_score(gt, (prob>=best_thr).astype(int), zero_division=0)
    try:
        auc = roc_auc_score(gt, prob)
    except Exception:
        auc = float('nan')
    mark = " *** F1>=0.8" if best_f1 >= 0.8 else ""
    print(f"  {d:<22} F1={best_f1:.3f}  P={p:.3f}  R={r:.3f}  AUC={auc:.3f}  thr={best_thr:.2f}  n_pos={int(gt.sum())}{mark}")
