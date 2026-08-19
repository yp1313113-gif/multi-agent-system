"""
绘制混合检索 vs 纯向量 / 纯 BM25 的召回率对比图。

用法：
    python eval/plot_retrieval_eval.py

输出：
    docs/assets/retrieval_comparison.png
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 显式加载系统 Noto Sans CJK 中文字体（.ttc 集合字体）
_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
_chinese_font = None
for fp in _FONT_CANDIDATES:
    if os.path.exists(fp):
        _chinese_font = fm.FontProperties(fname=fp)
        break

# 默认值来自 README 中报告的本地验证结果。
# 如果你本地运行了 eval/retrieval_eval.py，可以把真实结果写进同目录的
# eval_results.txt（格式：strategy=rate），脚本会自动读取。
DEFAULT_RATES = {
    "纯 BM25": 0.61,
    "纯向量": 0.72,
    "混合检索\n(BM25+向量+RRF+Rerank)": 0.88,
}

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "eval_results.txt")
OUTPUT_PNG = os.path.join(
    os.path.dirname(__file__), "..", "docs", "assets", "retrieval_comparison.png"
)


def load_rates():
    """尝试读取真实评估结果；失败则使用默认值。"""
    if not os.path.exists(RESULTS_FILE):
        return DEFAULT_RATES
    rates = {}
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                rates[k.strip()] = float(v.strip())
        if len(rates) >= 3:
            return rates
    except Exception:
        pass
    return DEFAULT_RATES


def main():
    rates = load_rates()
    labels = list(rates.keys())
    values = [rates[k] * 100 for k in labels]

    colors = ["#9ca3af", "#60a5fa", "#34d399"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)

    # 数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall@3", fontsize=12)
    ax.set_title(
        "RAG 检索召回率对比：混合检索 vs 单路检索",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 使用中文字体渲染中文标签
    if _chinese_font:
        for label in ax.get_xticklabels():
            label.set_fontproperties(_chinese_font)
            label.set_fontsize(11)
        ax.get_yaxis().get_label().set_fontproperties(_chinese_font)
        ax.get_yaxis().get_label().set_fontsize(12)
        ax.title.set_fontproperties(_chinese_font)
        ax.title.set_fontsize(14)

    # 底部注释
    caption = "测试集：12 条员工高频政策咨询；命中定义：top-3 片段包含期望关键词"
    if _chinese_font:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#6b7280", fontproperties=_chinese_font)
    else:
        fig.text(0.5, 0.01, caption, ha="center", fontsize=9, color="#6b7280")
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ 图表已保存: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
