# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 基于 README 中实测数据生成展示图表

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_BASE = "#4C72B0"
C_MID = "#7F9FC9"
C_BEST = "#2E7D32"
C_REF = "#DD8452"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------
# 图 1：三尺寸 TFLOPS（k1->k5 vs cuBLAS）
# ---------------------------------------------------------------
kernels = ["k1 naive", "k2 smem", "k3 pad", "k4 db", "k5 rb"]
tflops = {
    "1024^3": [8.4, 6.9, 9.7, 13.5, 24.4],
    "4096^3": [6.2, 5.4, 6.1, 10.9, 19.5],
    "8192^3": [1.9, 4.7, 5.5, 9.6, 18.7],
}
cublas = {"1024^3": 32.9, "4096^3": 30.3, "8192^3": 32.3}

fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), sharey=True)
for ax, (size, vals) in zip(axes, tflops.items()):
    x = list(range(len(kernels)))
    bars = ax.bar(x, vals, color=[C_MID, C_MID, C_MID, C_BASE, C_BEST], width=0.6)
    bars[4].set_edgecolor("black")
    bars[4].set_linewidth(1.6)
    ax.axhline(cublas[size], color=C_REF, linestyle="--", linewidth=1.6)
    ax.text(len(kernels) - 0.35, cublas[size] + 1.0, f"cuBLAS {cublas[size]}",
            ha="right", va="bottom", fontsize=7.5, color=C_REF, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["k1", "k2", "k3", "k4", "k5"], fontsize=9)
    ax.set_title(f"{size}  ({vals[4]/cublas[size]*100:.0f}% vs cuBLAS)",
                 fontweight="bold", fontsize=10)
    ax.set_ylim(0, 40)

axes[0].set_ylabel("TFLOPS (FP16 in, FP32 acc)")
fig.suptitle("WMMA FP16 GEMM: Kernel Evolution vs cuBLAS (RTX 5060 Laptop)",
             fontweight="bold", fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_tflops.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------
# 图 2：各尺寸下 k5 占比 + 加速来源拆解
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

sizes = ["1024^3", "4096^3", "8192^3"]
pct = [74.0, 64.2, 57.8]
bars = axes[0].bar(sizes, pct, color=[C_BEST, C_BASE, C_REF], width=0.5)
for b, v in zip(bars, pct):
    axes[0].text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom",
                 fontweight="bold", fontsize=10)
axes[0].axhline(100, color="gray", linestyle="--", linewidth=1.2)
axes[0].text(2.42, 101, "cuBLAS", ha="right", fontsize=8, color="gray")
axes[0].set_ylabel("% of cuBLAS TFLOPS")
axes[0].set_title("k5 register blocking: 占比随 size 下降", fontweight="bold", fontsize=10.5)
axes[0].set_ylim(0, 118)
axes[0].annotate("L2 命中 -> 带宽受限\ncuBLAS pipeline 优势扩大", xy=(1.78, 68),
                 xytext=(0.42, 96), va="top", ha="left",
                 arrowprops=dict(arrowstyle="->", color=C_REF, lw=1.6),
                 fontsize=8.5, color=C_REF, fontweight="bold")

steps = ["k1 naive", "k2 smem\n+padding", "k3 pad", "k4 double\nbuffering", "k5 register\nblocking"]
vals8k = [1.9, 4.7, 5.5, 9.6, 18.7]
axes[1].plot(range(5), vals8k, marker="o", markersize=7, linewidth=2.2, color=C_BEST,
             markerfacecolor="white", markeredgewidth=2)
for i, v in enumerate(vals8k):
    axes[1].annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                     xytext=(0, 10 if i != 1 else -16), ha="center", fontsize=8.5,
                     fontweight="bold", color=C_BEST)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(steps, fontsize=7.5)
axes[1].set_ylabel("TFLOPS (8192^3)")
axes[1].set_title("8192^3: 9.8x from k1 to k5", fontweight="bold", fontsize=10.5)
axes[1].set_ylim(0, 24)
axes[1].set_xlim(-0.4, 4.4)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_scaling.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------
# 图 0：架构图
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.0)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.13",
                                facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold")


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color))


box(0.2, 2.05, 1.45, 1.0, "A (fp16)  B (fp16)\n\nrow-major\nM N K", "#ECEFF1", "#90A4AE", fs=8.5)
arrow(1.65, 2.55, 2.1, 2.55)

ax.add_patch(FancyBboxPatch((2.1, 0.5), 8.7, 3.8,
                            boxstyle="round,pad=0.08,rounding_size=0.18",
                            facecolor="#FAFCFF", edgecolor=C_BASE, linewidth=1.7, alpha=0.5))
ax.text(2.3, 4.05, "WMMA GEMM kernel chain (sm_120, 26 SM, 32 MB L2)", fontsize=9.5,
        fontweight="bold", color=C_BASE)

kerns = [
    (2.3, "k1 naive\n\nglobal mem\nfragment load", "#F5F5F5", "#9E9E9E"),
    (4.05, "k2 smem\n\nsmem tile\nbank conflict", "#F5F5F5", "#9E9E9E"),
    (5.8, "k3 pad\n\nPAD=8\n去 bank conflict", "#E3F2FD", C_BASE),
    (7.55, "k4 db\n\nasync copy\n+ double buffer", "#E8F5E9", C_BEST),
    (9.3, "k5 rb\n\n16x64 per warp\n4 accumulators", "#FFF3E0", C_REF),
]
for x, t, fc, ec in kerns:
    box(x, 2.35, 1.5, 1.0, t, fc, ec, fs=7.5)
for i in range(4):
    arrow(kerns[i][0] + 1.5, 2.85, kerns[i+1][0], 2.85)

ax.text(2.3, 2.05, "shared memory / register 层优化路径 —— 每一步都用实测 TFLOPS 验证是否真的变快",
        fontsize=8, color="#546E7A", style="italic")

box(2.3, 1.0, 2.0, 0.8, "wmma::fragment\n16x16x16 per warp\nfp16 in / fp32 acc", "#EDE7F6", "#5E35B1", fs=7.5)
box(4.65, 1.0, 2.0, 0.8, "fragment layout\nrow/col major\nn_frag / k_frag stride", "#FFEBEE", "#C62828", fs=7.5)
box(7.0, 1.0, 1.9, 0.8, "correctness\nvs CPU: max 8.2e-3\nvs k1: max = 0", "#E8F5E9", C_BEST, fs=7.5)

arrow(9.9, 2.35, 9.9, 1.85, color=C_REF)
box(9.0, 1.0, 1.7, 0.8, "cuBLAS\n32.3 TFLOPS\nk5 = 57.8%", "#ECEFF1", "#90A4AE", fs=7.5)

ax.text(5.5, 4.82, "WMMA FP16 GEMM: k1 -> k5", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238")
ax.text(5.5, 4.52, "five kernels, one GEMM, every step verified by measured TFLOPS",
        ha="center", va="center", fontsize=9, color="#546E7A", style="italic")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig0_architecture.png"), bbox_inches="tight", dpi=170)
plt.close(fig)

print("已生成:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}  ({os.path.getsize(os.path.join(FIG_DIR, f))} bytes)")