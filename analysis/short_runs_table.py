"""
short_runs_table.py - evaluate the reduced-budget runs and print LaTeX rows for Table 2.

Runs evaluate_unified.py for each finished checkpoint (if not yet evaluated), then
ablation_table.row() with the reduced-budget A2 (seed 42) as the paired reference.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
RUNS = [("A2_s42_short", "A2"), ("A2_s7_short", "A2"), ("anatOnly_s42_short", "anatOnly"),
        ("simpleEmb_s42_short", "simpleEmb"), ("segOnly_s42_short", "segOnly")]
LABEL = {"A2_s42_short": "A2 (seed 42)", "A2_s7_short": "A2 (seed 7)", "anatOnly_s42_short": "A2-anat",
         "simpleEmb_s42_short": "A2-simple", "segOnly_s42_short": "A2-seg"}

sys.path.insert(0, str(HERE))
import ablation_table as at  # noqa: E402

done = []
for name, variant in RUNS:
    ck = ROOT / "checkpoints" / "variants" / f"{name}_best.pt"
    if not ck.exists():
        print("missing", name)
        continue
    if not (ROOT / "results" / "analysis" / "unified" / f"{name}_test_clinical.csv").exists():
        subprocess.run([PY, str(HERE / "evaluate_unified.py"), "--name", name, "--ckpt", str(ck), "--kind", "rev",
                        "--variant", variant], check=True)
    done.append(name)

rows = {n: at.row(n, "A2_s42_short" if n != "A2_s42_short" else None) for n in done}
json.dump(rows, open(ROOT / "results" / "analysis" / "ablation_short_rows.json", "w"), indent=1)
import torch  # noqa: E402
for n in done:
    r = rows[n]
    ck = torch.load(ROOT / "checkpoints" / "variants" / f"{n}_best.pt", map_location="cpu", weights_only=False)
    ntr = None
    d, lo, hi = r["dice"]
    line = (f"\\add{{{LABEL[n]}}} & \\add{{PARAMS}} & \\add{{{d:.3f} [{lo:.3f},{hi:.3f}]}} & \\add{{{r['iou']:.3f}}} & "
            f"\\add{{{r['dice_mcsel']:.3f}}} & \\add{{{r['grade_exact']:.3f}}} & \\add{{{r['kappa_quad']:.2f}}} & "
            f"\\add{{{r['height_mae']:.2f}}} & \\add{{{r['dis_macro_auc']:.2f}}} & \\add{{--}} \\\\")
    print(line.replace("{.", "{0.").replace("[.", "[0."))
    if "vs_ref" in r:
        v = r["vs_ref"]
        print(f"   vs A2_s42_short: diff {v['mean_diff']:+.4f} CI {v['ci'][0]:+.4f} {v['ci'][1]:+.4f} "
              f"perm p {v['perm_p']:.2g} better {v['patients_a_better']}/78; best epoch {ck.get('epoch')} phase {ck.get('phase')} val {ck['val'].get('dice'):.4f}")
    else:
        print(f"   best epoch {ck.get('epoch')} phase {ck.get('phase')} val {ck['val'].get('dice'):.4f}")
