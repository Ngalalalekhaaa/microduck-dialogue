"""Build static report figures from the frozen log snapshot; never touch training."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    # This Noto collection contains CFF outlines; Type 3 avoids a false
    # TrueType declaration when Matplotlib embeds it in a PDF figure.
    plt.rcParams["pdf.fonttype"] = 3
    rows = [json.loads(line) for line in (ROOT / "snapshot_metrics.jsonl").read_text().splitlines()]
    training = [row for row in rows if "train_loss" in row]
    baseline = json.loads((ROOT / "snapshot_baseline_evaluation.json").read_text())
    validation = [row for row in rows if "validation_loss" in row]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [1.7, 1]})
    color = "#137C8B"
    left.plot([r["step"] for r in training], [r["train_loss"] for r in training], color=color, lw=1.4)
    left.set(title="训练损失：前101次参数更新", xlabel="优化器更新次数", ylabel="四样本平均损失（nats）")
    left.grid(alpha=0.18)
    vals = [baseline["validation_loss"], validation[0]["validation_loss"]]
    bars = right.bar(["微调前", f"第{validation[0]['step']}步"], vals, color=["#8FA2B5", color], width=0.55)
    right.set(title="固定32条验证样本", ylabel="给定正确前文的损失（nats）", ylim=(0, 1.22))
    for bar, value in zip(bars, vals):
        right.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.4f}", ha="center", fontsize=10)
    for axis in (left, right):
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "assets/training_snapshot.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
