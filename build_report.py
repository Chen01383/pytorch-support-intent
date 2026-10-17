"""Build static scientific figures and a report entirely from recorded results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import probabilities, routing_metrics

ROOT = Path(__file__).resolve().parent


def build_report():
    folder = ROOT / "reports"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((folder / "selection_locked.json").read_text(encoding="utf-8"))
    audit = json.loads((folder / "data_audit.json").read_text(encoding="utf-8"))
    seed = str(summary["deployed_seed"])
    deployed = summary["neural_seeds"][seed]
    with np.load(ROOT / "artifacts/test_outputs.npz") as outputs:
        y = outputs["y"].copy()
        logits = outputs[f"logits_seed{seed}"].copy()
    raw = probabilities(logits)
    calibrated = probabilities(logits, lock["calibration"][seed]["temperature"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["Test accuracy", "Test macro F1"]):
        means = [summary["baseline"][metric], summary["neural_mean_std"][metric]["mean"]]
        ax.bar(["TF-IDF + logistic", "MiniLM + PyTorch"], means, color=["#91a5b7", "#2878a5"],
               yerr=[0, summary["neural_mean_std"][metric]["sample_std"]], capsize=5)
        for i, value in enumerate(means):
            ax.text(i, value + 0.012, f"{value:.3f}", ha="center")
        ax.set_ylim(0, 1.02)
        ax.set_title(title + " (3-seed mean; sample SD)")
    fig.savefig(folder / "model_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for probs, label, color in [(raw, "Uncalibrated", "#b35c40"), (calibrated, "Temperature scaled", "#2878a5")]:
        confidences, correct = probs.max(1), probs.argmax(1) == y
        xs, ys = [], []
        for i in range(10):
            mask = (confidences >= i / 10) & (confidences <= (i + 1) / 10 if i == 9 else confidences < (i + 1) / 10)
            if mask.any():
                xs.append(confidences[mask].mean())
                ys.append(correct[mask].mean())
        axes[0].plot(xs, ys, "o-", label=label, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Mean confidence (10 bins)", ylabel="Observed accuracy", title="Test reliability: deployment seed 42", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(fontsize=8)
    route_points = [routing_metrics(calibrated, y, float(t)) for t in np.linspace(0, 0.99, 100)]
    valid = [r for r in route_points if r["accepted"]]
    axes[1].plot([r["coverage"] for r in valid], [r["accepted_accuracy"] for r in valid], color="#2878a5")
    route = deployed["routing"]
    if route["accepted"]:
        axes[1].scatter([route["coverage"]], [route["accepted_accuracy"]], color="#b35c40", zorder=5,
                        label="Policy fixed on calibration split")
    axes[1].set(xlabel="Fraction automatically routed", ylabel="Accuracy among routed queries",
                title="Test coverage / accuracy (descriptive)", xlim=(0, 1.02), ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    fig.savefig(folder / "calibration_and_routing.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for run_seed, run in lock["final_training"].items():
        history = run["history"]
        axes[0].plot([h["epoch"] for h in history], [h["train_cross_entropy"] for h in history], label=f"seed {run_seed}")
        axes[1].plot([h["epoch"] for h in history], [h["selection_macro_f1"] for h in history], label=f"seed {run_seed}")
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy", title="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Selection split (not test)")
    axes[1].legend()
    fig.savefig(folder / "training_curves.png", dpi=160)
    plt.close(fig)
    bootstrap = summary["paired_accuracy_bootstrap"]
    interval = bootstrap["percentile_95"]
    ci = route["accepted_accuracy_wilson_95"]
    selected_accuracy = "undefined" if route["accepted_accuracy"] is None else f"{route['accepted_accuracy']:.2%}"
    lines = ["# 实验结果：英文客服意图识别", "", "全部数值来自本机真实运行，模型和分流规则均在测试评估前确定。", "",
             "## 数据与协议", "", f"官方数据：{audit['official_train_rows']:,} 条训练文本、{audit['official_test_rows']:,} 条测试文本、{audit['classes']} 个意图。",
             f"规范化去重及去污染后划分：`{audit['split_sizes']}`。去除记录：`{audit['removed_training_rows']}`。",
             "保留完整官方测试集。仅用测试输入的规范化文本做精确去污染，不用测试标签决定训练或模型选择。",
             "词表仅在训练集拟合；预训练 MiniLM 全程冻结。模型选择集用于结构选择和早停，校准集用于温度与分流阈值。",
             f"选中隐藏层维度 `{lock['selected_hidden_dim']}`，基线 C=`{lock['selected_baseline_C']}`；部署种子预定为 42。", "",
             "## 测试集指标", "", "| 模型 | Accuracy | Macro F1 | Top-3 accuracy |", "|---|---:|---:|---:|"]
    baseline = summary["baseline"]
    lines.append(f"| TF-IDF + Logistic Regression | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} | {baseline['top3_accuracy']:.4f} |")
    for run_seed, run in summary["neural_seeds"].items():
        m = run["calibrated"]
        lines.append(f"| MiniLM + PyTorch, seed {run_seed} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['top3_accuracy']:.4f} |")
    means = summary["neural_mean_std"]
    lines += [f"| 三种子 mean ± sample SD | {means['accuracy']['mean']:.4f} ± {means['accuracy']['sample_std']:.4f} | {means['macro_f1']['mean']:.4f} ± {means['macro_f1']['sample_std']:.4f} | {means['top3_accuracy']['mean']:.4f} ± {means['top3_accuracy']['sample_std']:.4f} |", "",
              f"固定 seed 42 相对基线准确率差：**{bootstrap['accuracy_difference'] * 100:+.2f} 个百分点**；按查询成对 bootstrap 的 95% 区间为 [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] 个百分点。",
              "这个区间以已经训练好的模型为条件，不能表示训练随机性、未来分布变化或线上收益。三种子标准差单独报告。", "",
              "![模型对比](model_comparison.png)", "", "## 置信度与人工分流", "",
              f"部署模型温度 T={lock['calibration'][seed]['temperature']:.4f}，阈值={route['threshold']:.4f}。",
              f"测试 NLL：{deployed['raw']['nll']:.4f} → {deployed['calibrated']['nll']:.4f}；ECE(15 bins)：{deployed['raw']['ece_15_bins']:.4f} → {deployed['calibrated']['ece_15_bins']:.4f}。",
              f"测试自动分流 {route['accepted']}/{route['total']} 条，覆盖率 **{route['coverage']:.2%}**，自动分流部分准确率 **{selected_accuracy}**；其余交给人工。",
              f"自动分流准确率的 Wilson 95% 区间：`{ci}`。校准集 95% 的经验目标并非测试或线上保证，阈值本身也经过校准集搜索。",
              "温度缩放不会改变 Top-1 类别；它只改变概率分布。图上的完整测试风险覆盖曲线仅供事后描述，没有用它重选阈值。", "",
              "![置信度与分流](calibration_and_routing.png)", "", "![训练曲线](training_curves.png)", "",
              "## 错误分析", "", "`confusion_pairs.json` 汇总最常混淆的意图，`error_examples.json` 列出高置信度误判的 30 个真实例子。",
              "数据样例归 PolyAI 原数据集所有，按 CC-BY-4.0 使用；来源和引用见 README。", "", "## 边界", "",
              "- 英文、银行客服、77 类闭集分类；未验证中文、其他行业或未知意图识别。",
              "- 只训练小型 PyTorch 分类头，不是大语言模型微调，也不生成客服答案。",
              "- 预训练模型使用的语料没有在本项目中独立审计，不能证明外部预训练不存在数据重叠。",
              "- 去重训练协议与直接使用全部官方训练集的论文设置不同，不作榜单或 SOTA 比较。",
              "- 推理性能只适用于报告中给出的设备、线程、批量和计时范围。", "",
              f"训练环境：`{summary['environment']}`；实验用时 {summary['experiment_seconds']:.1f} 秒（若有缓存会复用编码）。", ""]
    (folder / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build_report()
