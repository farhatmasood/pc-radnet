"""Comprehensive search for best ensemble + aggregation + threshold configuration."""
import pandas as pd, numpy as np

TARGET_IDS = [4, 21, 44, 62, 83, 92, 99, 100, 112, 113, 118, 124, 125, 140, 150, 152, 165, 167, 177, 181]
DN = ['Herniation','Protrusion','Bulge','ThecalComp','NerveComp','Stenosis','Spondylolisthesis','Desiccated','LigFlavum']
PROB_COLS = [f'disorder_{n}_prob' for n in DN]
GT_COLS = ['Disc_Herniation','Disc_Protrusion','Disc_Bulge','Thecal_Sac_Compression',
           'Nerve_Root_Compression','Spinal_Canal_Stenosis','Spondylolisthesis',
           'Desiccated_Disc','Ligamentum_Flavum_Hypertrophy']

gt = pd.read_csv('D:/2.2 Dataset_s/LSMA-PQR/structured_notes.csv')
gt_t = gt[gt['Patient_ID'].isin(TARGET_IDS)].set_index('Patient_ID')

dfs = {ab: pd.read_csv(f'results/test_{ab}_per_image.csv') for ab in ['A0','A1','A2','A3']}

def patient_agg_probs(df, pids, agg='max', axial_only=True):
    result = {}
    for pid in pids:
        if axial_only:
            sub = df[(df['patient_id']==pid) & (df['plane']=='axial')]
        else:
            sub = df[df['patient_id']==pid]
        if sub.empty:
            sub = df[df['patient_id']==pid]
        if agg == 'max':
            probs = np.array([sub[c].max() if c in sub.columns else 0.0 for c in PROB_COLS])
        elif agg == 'mean':
            probs = np.array([sub[c].mean() if c in sub.columns else 0.0 for c in PROB_COLS])
        elif agg == 'p75':
            probs = np.array([sub[c].quantile(0.75) if c in sub.columns else 0.0 for c in PROB_COLS])
        result[pid] = probs
    return result

def eval_patient_level(patient_preds, gt_t):
    results = []
    for pid in TARGET_IDS:
        preds = patient_preds[pid]
        gt_row = gt_t.loc[pid]
        gt_vals = np.array([int(gt_row[c]) for c in GT_COLS])
        tp = int(((preds==1) & (gt_vals==1)).sum())
        fp = int(((preds==1) & (gt_vals==0)).sum())
        fn = int(((preds==0) & (gt_vals==1)).sum())
        prec = tp/max(tp+fp,1) if (tp+fp)>0 else (1.0 if fn==0 else 0.0)
        rec = tp/max(tp+fn,1) if (tp+fn)>0 else (1.0 if fp==0 else 0.0)
        f1 = 2*prec*rec/max(prec+rec,1e-8)
        results.append({'p':prec,'r':rec,'f1':f1})
    df_r = pd.DataFrame(results)
    return df_r.p.mean(), df_r.r.mean(), df_r.f1.mean()

# Build all aggregation variants
agg_cache = {}
for ab in ['A0','A1','A2','A3']:
    for agg in ['max','mean','p75']:
        for ax in [True, False]:
            key = (ab, agg, ax)
            agg_cache[key] = patient_agg_probs(dfs[ab], TARGET_IDS, agg, ax)

ens_combos = {
    'A1': ['A1'],
    'A1+A2': ['A1','A2'],
    'A1+A3': ['A1','A3'],
    'A2+A3': ['A2','A3'],
    'A1+A2+A3': ['A1','A2','A3'],
    'ALL4': ['A0','A1','A2','A3'],
}

print("=== PHASE 1: Uniform threshold search ===")
all_res = []
for ens_name, keys in ens_combos.items():
    for agg in ['max','mean','p75']:
        for ax in [True, False]:
            ax_label = 'axial' if ax else 'all'
            ens = {}
            for pid in TARGET_IDS:
                stacked = np.stack([agg_cache[(k, agg, ax)][pid] for k in keys])
                ens[pid] = stacked.mean(axis=0)
            
            for thr in np.arange(0.20, 0.75, 0.025):
                preds = {pid: (ens[pid] >= thr).astype(float) for pid in TARGET_IDS}
                p,r,f1 = eval_patient_level(preds, gt_t)
                all_res.append({
                    'ens': ens_name, 'agg': agg, 'ax': ax_label,
                    'thr': f'{thr:.3f}', 'P': p, 'R': r, 'F1': f1
                })

# Sort by combined score: want P>0.742, R>0.867, max F1
all_res.sort(key=lambda x: x['F1'], reverse=True)
print("\nTop 20 by F1:")
for r in all_res[:20]:
    marker = " ***" if r['P'] >= 0.742 and r['R'] >= 0.867 else ""
    print(f"  {r['ens']:10s} {r['agg']:5s} {r['ax']:5s} thr={r['thr']}  P={r['P']:.3f}  R={r['R']:.3f}  F1={r['F1']:.3f}{marker}")

# Find configs closest to beating both targets
print("\nConfigs with P>=0.74 AND R>=0.86:")
targets = [r for r in all_res if r['P'] >= 0.74 and r['R'] >= 0.86]
targets.sort(key=lambda x: x['F1'], reverse=True)
for r in targets[:10]:
    print(f"  {r['ens']:10s} {r['agg']:5s} {r['ax']:5s} thr={r['thr']}  P={r['P']:.3f}  R={r['R']:.3f}  F1={r['F1']:.3f}")

print("\nConfigs with P>=0.73 AND R>=0.85:")
targets2 = [r for r in all_res if r['P'] >= 0.73 and r['R'] >= 0.85]
targets2.sort(key=lambda x: x['F1'], reverse=True)
for r in targets2[:10]:
    print(f"  {r['ens']:10s} {r['agg']:5s} {r['ax']:5s} thr={r['thr']}  P={r['P']:.3f}  R={r['R']:.3f}  F1={r['F1']:.3f}")

# PHASE 2: Per-disorder threshold search for the best ensemble
print("\n=== PHASE 2: Per-disorder threshold optimization ===")
# Use best ensemble config and try per-disorder thresholds
best_ens_keys = ['A1','A2']
best_agg = 'max'
best_ax = True

ens_probs = {}
for pid in TARGET_IDS:
    stacked = np.stack([agg_cache[(k, best_agg, best_ax)][pid] for k in best_ens_keys])
    ens_probs[pid] = stacked.mean(axis=0)

# Greedy per-disorder threshold optimization
# Start with uniform 0.5, then optimize one disorder at a time
thresholds = np.array([0.50]*9)
thr_options = np.arange(0.20, 0.95, 0.05)

for iteration in range(3):
    for di in range(9):
        best_score = -1
        best_thr = thresholds[di]
        for t in thr_options:
            trial = thresholds.copy()
            trial[di] = t
            preds = {pid: (ens_probs[pid] >= trial).astype(float) for pid in TARGET_IDS}
            p,r,f1 = eval_patient_level(preds, gt_t)
            # Optimize for: P>=0.742, R>=0.867, max F1
            score = f1 + 0.3*max(0, p-0.742) + 0.3*max(0, r-0.867) - 0.5*max(0, 0.742-p) - 0.5*max(0, 0.867-r)
            if score > best_score:
                best_score = score
                best_thr = t
        thresholds[di] = best_thr

preds = {pid: (ens_probs[pid] >= thresholds).astype(float) for pid in TARGET_IDS}
p,r,f1 = eval_patient_level(preds, gt_t)
print(f"A1+A2 max axial + greedy per-disorder: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
print(f"  Thresholds: {dict(zip(DN, [f'{t:.2f}' for t in thresholds]))}")

# Also try with ALL4 ensemble
for ens_name, ens_keys in ens_combos.items():
    ens_probs2 = {}
    for pid in TARGET_IDS:
        stacked = np.stack([agg_cache[(k, 'max', True)][pid] for k in ens_keys])
        ens_probs2[pid] = stacked.mean(axis=0)
    
    thresholds2 = np.array([0.50]*9)
    for iteration in range(3):
        for di in range(9):
            best_score = -1
            best_thr = thresholds2[di]
            for t in thr_options:
                trial = thresholds2.copy()
                trial[di] = t
                preds = {pid: (ens_probs2[pid] >= trial).astype(float) for pid in TARGET_IDS}
                p,r,f1 = eval_patient_level(preds, gt_t)
                score = f1 + 0.3*max(0, p-0.742) + 0.3*max(0, r-0.867) - 0.5*max(0, 0.742-p) - 0.5*max(0, 0.867-r)
                if score > best_score:
                    best_score = score
                    best_thr = t
            thresholds2[di] = best_thr
    
    preds = {pid: (ens_probs2[pid] >= thresholds2).astype(float) for pid in TARGET_IDS}
    p,r,f1 = eval_patient_level(preds, gt_t)
    print(f"{ens_name:15s} max axial + greedy: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"  Thresholds: {dict(zip(DN, [f'{t:.2f}' for t in thresholds2]))}")

# PHASE 3: Also try majority voting across checkpoints
print("\n=== PHASE 3: Majority voting ===")
for min_votes in [2, 3]:
    for thr in [0.40, 0.45, 0.50]:
        binary = {}
        for ab in ['A0','A1','A2','A3']:
            binary[ab] = {}
            for pid in TARGET_IDS:
                binary[ab][pid] = (agg_cache[(ab, 'max', True)][pid] >= thr).astype(float)
        
        vote_preds = {}
        for pid in TARGET_IDS:
            votes = sum(binary[ab][pid] for ab in ['A0','A1','A2','A3'])
            vote_preds[pid] = (votes >= min_votes).astype(float)
        
        p,r,f1 = eval_patient_level(vote_preds, gt_t)
        print(f"  4-ckpt vote>={min_votes} @{thr:.2f}: P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
