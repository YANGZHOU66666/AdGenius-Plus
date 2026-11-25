"""Render the November 24 calibration figure from recorded metrics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / "results/nov24"
with (out/"judge_summary.csv").open() as f:
    rows = list(csv.DictReader(f))
bt = json.loads((out/"generator_bt/metrics.json").read_text())
fig, (agreement, rank) = plt.subplots(1, 2, figsize=(10.4, 3.9), layout="constrained")
for i, row in enumerate(rows):
    for offset, name, color, marker, value in [
        (-.13,"Preference (100 pairs)","#275c89","o",float(row["pairwise_order_mean"])),
        (.13,"Conflict (130 ads)","#af581f","s",int(row["rubric_correct"])/int(row["rubric_n"]))]:
        agreement.scatter(value*100,i+offset,color=color,marker=marker,s=46,label=name if i==0 else None)
agreement.set(yticks=range(4),yticklabels=[r["judge"] for r in rows],xlim=(65,100),xlabel="Agreement with one annotator (%)",title="A   Judge calibration")
agreement.invert_yaxis()
agreement.legend(loc="lower center",bbox_to_anchor=(.5,-.46),frameon=False,fontsize=9)
names={"glm53_flash":"GLM","gemma_31b":"Gemma","qwen_local":"Qwen 2B","longcat":"LongCat"}
for i,r in enumerate(bt["systems"]):
    lo,hi = r["score_ci95"]
    rank.plot([lo,hi],[i,i],color="#4c6875",linewidth=2)
    rank.scatter([r["score"]],[i],color="#244a5b",s=40)
rank.set(yticks=range(4),yticklabels=[names[r["system"]] for r in bt["systems"]],xlabel="Bradley–Terry score (Qwen 2B fixed at 0)",title="B   Generator preference")
rank.invert_yaxis()
for ax in [agreement,rank]:
    ax.spines[["top","right","left"]].set_visible(False)
    ax.grid(axis="x",color="#eeeeee")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y",length=0)
fig.savefig(out/"gold_calibration.pdf")
fig.savefig(out/"gold_calibration.png",dpi=180)
