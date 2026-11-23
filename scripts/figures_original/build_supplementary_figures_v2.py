# -*- coding: utf-8 -*-
"""
生成 Supplementary Materials 全部三张正式补充图（Fig. S1–S3），严格基于项目本地封存真实实验数据，
统一采用 v2 遥感 SCI 顶刊视觉体系（Microsoft YaHei + pdf.fonttype=42 + svg.fonttype="none" + 600 dpi PNG）。

数据来源（全部只读）：
  - Fig. S1: results/P5_02/tables/architecture_efficiency_comparison.csv + 12 份真实 training_log.json
  - Fig. S2: results/P5_03/tables/all_288_27_scenarios.csv (38,880 条真实情景记录)
  - Fig. S3: experiments/P3_01_FIX03/*/training_log.json, results/P5_01/training/*/*/training_log.json,
             experiments/P5_02/training/*/*/training_log.json (全部 12 次真实训练日志)

输出目录：paper/P6_integrated_v2/figures/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from src.figure_paths import read_table
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import os
ROOT = Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT_DIR = ROOT / "figures" / "regenerated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.linewidth"] = 0.8

ARCH_COLORS = {"UNet": "#2b6cb0", "DeepLabV3Plus": "#dd6b20"}
SEED_STYLES = {
    20261115: ("-", "o", "种子 20261115"),
    20261121: ("--", "^", "种子 20261121"),
    20261122: (":", "s", "种子 20261122"),
}


def load_all_12_training_logs() -> pd.DataFrame:
    """读取全部 12 份真实封存 training_log.json（2 架构 × 2 城市 × 3 种子 × 12 轮）。"""
    log_specs = [
        ("UNet", 20261115, "AOI_3_Paris", ROOT / "experiments/P3_01_FIX03/AOI_3_Paris/training_log.json"),
        ("UNet", 20261115, "AOI_5_Khartoum", ROOT / "experiments/P3_01_FIX03/AOI_5_Khartoum/training_log.json"),
        ("UNet", 20261121, "AOI_3_Paris", ROOT / "results/P5_01/training/20261121/AOI_3_Paris/training_log.json"),
        ("UNet", 20261121, "AOI_5_Khartoum", ROOT / "results/P5_01/training/20261121/AOI_5_Khartoum/training_log.json"),
        ("UNet", 20261122, "AOI_3_Paris", ROOT / "results/P5_01/training/20261122/AOI_3_Paris/training_log.json"),
        ("UNet", 20261122, "AOI_5_Khartoum", ROOT / "results/P5_01/training/20261122/AOI_5_Khartoum/training_log.json"),
        ("DeepLabV3Plus", 20261115, "AOI_3_Paris", ROOT / "experiments/P5_02/training/20261115/AOI_3_Paris/training_log.json"),
        ("DeepLabV3Plus", 20261115, "AOI_5_Khartoum", ROOT / "experiments/P5_02/training/20261115/AOI_5_Khartoum/training_log.json"),
        ("DeepLabV3Plus", 20261121, "AOI_3_Paris", ROOT / "experiments/P5_02/training/20261121/AOI_3_Paris/training_log.json"),
        ("DeepLabV3Plus", 20261121, "AOI_5_Khartoum", ROOT / "experiments/P5_02/training/20261121/AOI_5_Khartoum/training_log.json"),
        ("DeepLabV3Plus", 20261122, "AOI_3_Paris", ROOT / "experiments/P5_02/training/20261122/AOI_3_Paris/training_log.json"),
        ("DeepLabV3Plus", 20261122, "AOI_5_Khartoum", ROOT / "experiments/P5_02/training/20261122/AOI_5_Khartoum/training_log.json"),
    ]
    rows = []
    for arch, seed, city, path in log_specs:
        assert path.exists(), f"Missing real training log: {path}"
        data = json.loads(path.read_text(encoding="utf-8"))
        for rec in data:
            rows.append({
                "architecture": arch,
                "seed": seed,
                "city": city,
                "epoch": int(rec["epoch"]),
                "train_loss": float(rec["train_loss"]),
                "validation_loss": float(rec["validation_loss"]),
                "checkpoint_score": float(rec["checkpoint_score"]),
                "elapsed_seconds": float(rec["elapsed_seconds"]),
            })
    df = pd.DataFrame(rows)
    assert len(df) == 144, f"Expected 12 runs * 12 epochs = 144 rows, got {len(df)}"
    return df


def build_fig_s1(df_logs: pd.DataFrame) -> None:
    """绘制 Fig. S1: 模型复杂度、训练耗时、推理耗时与峰值内存对比。"""
    eff = read_table(ROOT / "results/P5_02/tables/architecture_efficiency_comparison.csv").set_index("architecture")
    archs = ["UNet", "DeepLabV3Plus"]
    alabs = ["U-Net\n(基线)", "DeepLabV3+\n(对照)"]
    cols = [ARCH_COLORS[a] for a in archs]

    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.4))
    fig.subplots_adjust(left=0.048, right=0.988, top=0.84, bottom=0.18, wspace=0.46)

    # (a) 参数量（百万）与模型权重体积（MB）
    ax = axes[0]
    ax.spines["top"].set_visible(False)
    params_m = [float(eff.loc[a, "total_parameters"]) / 1e6 for a in archs]
    ckpt_mb = [float(eff.loc[a, "best_pt_bytes_mean"]) / (1024 ** 2) for a in archs]
    x = np.arange(2)
    ax.bar(x - 0.17, params_m, width=0.32, color=cols, edgecolor="#222", lw=0.7, label="可训练参数量 (M)")
    for xi, v in zip(x, params_m):
        ax.text(xi - 0.17, v + 0.04, f"{v:.3f}M", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1a202c")
    ax.set_xticks(x)
    ax.set_xticklabels(alabs, fontsize=8.0)
    ax.set_ylim(0, 1.95)
    ax.set_ylabel("参数量 (百万, M)", fontsize=8.2, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.bar(x + 0.17, ckpt_mb, width=0.32, color=cols, alpha=0.45, hatch="///", edgecolor="#222", lw=0.7, label="最优权重体积 (MB)")
    for xi, v in zip(x, ckpt_mb):
        ax_r.text(xi + 0.17, v + 0.5, f"{v:.1f} MB", ha="center", va="bottom", fontsize=7.3, color="#4a5568")
    ax_r.set_ylim(0, 23.0)
    ax_r.set_ylabel("best.pt 体积 (MB)", fontsize=8.0, color="#4a5568")
    ax.set_title("(a) 模型参数量与权重体积 (3.06×)", loc="left", fontsize=8.8, fontweight="bold", pad=8)

    # (b) 单次 12 轮完整训练耗时（展示全部 6 次真实运行散点 + 均值 + min-max）
    ax = axes[1]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    run_times = df_logs[df_logs["epoch"] == 12].copy()
    for i, a in enumerate(archs):
        t_mean = float(eff.loc[a, "train_seconds_mean"])
        t_min = float(eff.loc[a, "train_seconds_min"])
        t_max = float(eff.loc[a, "train_seconds_max"])
        ax.bar([i], [t_mean], width=0.46, color=cols[i], alpha=0.35, edgecolor=cols[i], lw=1.1, zorder=2)
        ax.plot([i, i], [t_min, t_max], color="#1a202c", lw=1.6, zorder=3)
        ax.plot([i - 0.12, i + 0.12], [t_min, t_min], color="#1a202c", lw=1.2, zorder=3)
        ax.plot([i - 0.12, i + 0.12], [t_max, t_max], color="#1a202c", lw=1.2, zorder=3)
        sub_t = run_times[run_times["architecture"] == a]["elapsed_seconds"].values
        jit = np.linspace(-0.12, 0.12, len(sub_t))
        ax.scatter(i + jit, sub_t, color=cols[i], edgecolor="white", s=34, lw=0.6, zorder=5)
        ax.text(i, t_max + 25, f"均值 {t_mean:.1f} s\n[{t_min:.0f}–{t_max:.0f} s]",
                ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1a202c")
    ax.set_xticks(x)
    ax.set_xticklabels(alabs, fontsize=8.0)
    ax.set_ylim(0, 1020)
    ax.set_ylabel("单次 12 轮训练总耗时 (s)", fontsize=8.2, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax.set_title("(b) 训练耗时分布 (各 n=6 次运行)", loc="left", fontsize=8.8, fontweight="bold", pad=8)

    # (c) 单景测试推理耗时（ms/景，含 12 路线均值与 min-max）
    ax = axes[2]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, a in enumerate(archs):
        ms_mean = float(eff.loc[a, "infer_scene_ms_mean"])
        ms_min = float(eff.loc[a, "infer_scene_ms_min_route_mean"])
        ms_max = float(eff.loc[a, "infer_scene_ms_max_route_mean"])
        ax.bar([i], [ms_mean], width=0.46, color=cols[i], alpha=0.82, edgecolor="#222", lw=0.7, zorder=2)
        ax.plot([i, i], [ms_min, ms_max], color="#111111", lw=1.6, zorder=4)
        ax.text(i, ms_max + 0.7, f"{ms_mean:.2f} ms/景\n[{ms_min:.2f}–{ms_max:.2f}]",
                ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#1a202c")
    ax.set_xticks(x)
    ax.set_xticklabels(alabs, fontsize=8.0)
    ax.set_ylim(0, 23.5)
    ax.set_ylabel("单景 650×650 推理耗时 (ms/景)", fontsize=8.2, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax.set_title("(c) 单景推理耗时对比 (1.82×)", loc="left", fontsize=8.8, fontweight="bold", pad=8)

    # (d) 训练进程峰值内存与评估进程峰值内存（MB）
    ax = axes[3]
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    tr_mem = [float(eff.loc[a, "train_peak_mem_mb_mean"]) for a in archs]
    tr_max = [float(eff.loc[a, "train_peak_mem_mb_max"]) for a in archs]
    ev_mem = [float(eff.loc[a, "eval_process_peak_mem_mb"]) for a in archs]
    ax.bar(x - 0.17, tr_mem, width=0.32, color=cols, edgecolor="#222", lw=0.7, label="训练峰值内存均值")
    ax.bar(x + 0.17, ev_mem, width=0.32, color="#94a3b8", edgecolor="#222", lw=0.7, label="全量评估进程峰值")
    for i in range(2):
        ax.text(i - 0.17, tr_mem[i] + 35, f"{tr_mem[i]:.0f}\n(max {tr_max[i]:.0f})", ha="center", va="bottom", fontsize=6.9, fontweight="bold")
        ax.text(i + 0.17, ev_mem[i] + 35, f"{ev_mem[i]:.0f}", ha="center", va="bottom", fontsize=7.2, color="#4a5568")
    ax.set_xticks(x)
    ax.set_xticklabels(alabs, fontsize=8.0)
    ax.set_ylim(0, 2200)
    ax.set_ylabel("进程峰值内存 (MB)", fontsize=8.2, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, ls="--")
    ax.legend(loc="upper center", fontsize=6.8, frameon=True, facecolor="white", ncol=1)
    ax.set_title("(d) 训练与评估进程峰值内存", loc="left", fontsize=8.8, fontweight="bold", pad=8)

    fig.text(
        0.5, 0.02,
        "注：全部统计来自封存表 results/P5_02/tables/architecture_efficiency_comparison.csv（各架构含 2 城市 × 3 种子 = 6 次独立训练运行及 12 条测试推理路线）；(d) 评估进程峰值内存为统一全量评估脚本进程峰值（两架构共用评估流水线，峰值均为 1,751 MB）。",
        ha="center", fontsize=7.4, color="#2d3748",
    )

    stem = "figS1_model_efficiency_and_resources"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


def build_fig_s2() -> None:
    """绘制 Fig. S2: 27 种完整阈值组合情景矩阵（跨 288 组因子实验均值与范围）。"""
    df27 = read_table(ROOT / "results/P5_03/tables/all_288_27_scenarios.csv")
    df_all = df27[df27["stratum"] == "all"].copy()
    # 核验确为 288 * 27 = 7,776 条 all 分层情景记录
    assert len(df_all) == 288 * 27, f"Expected 7776 all-stratum rows, got {len(df_all)}"

    # 构建 9 行 (net_tol × spatial_tol) × 3 列 (Q_min = 20%, 50%, 80%) 热力矩阵
    # 按 4 个代表情景展示：巴黎 A (100m, 500m) + 喀土穆 D (100m, 500m) + 喀土穆 B (100m, 500m) -> 2×3 矩阵
    net_vals = [0.01, 0.02, 0.05]
    sp_vals = [0.02, 0.05, 0.10]
    q_vals = [0.2, 0.5, 0.8]
    row_pairs = [(n, s) for n in net_vals for s in sp_vals]
    row_labels = [f"E_net≤{int(n*100)}% / E_sp≤{int(s*100)}%" + (" ★" if (n == 0.02 and s == 0.05) else (" (非嵌套)" if n > s else ""))
                  for n, s in row_pairs]

    panel_specs = [
        ("AOI_3_Paris", "A", 100, "(a) 巴黎同城 A (100 m)：隐蔽面积占比 (%)", "hidden_area_fraction_total", 100.0, "YlOrRd", 0, 26),
        ("AOI_5_Khartoum", "D", 100, "(b) 喀土穆同城 D (100 m)：隐蔽面积占比 (%)", "hidden_area_fraction_total", 100.0, "YlOrRd", 0, 42),
        ("AOI_5_Khartoum", "D", 500, "(c) 喀土穆同城 D (500 m)：隐蔽面积占比 (%)", "hidden_area_fraction_total", 100.0, "YlOrRd", 0, 42),
        ("AOI_5_Khartoum", "B", 100, "(d) 喀土穆跨城 B (100 m)：隐蔽面积占比 (%)", "hidden_area_fraction_total", 100.0, "YlOrRd", 0, 42),
        ("AOI_5_Khartoum", "D", 100, "(e) 喀土穆同城 D (100 m)：联合通过率 (%)", "joint_pass_ratio", 100.0, "YlGn", 0, 75),
        ("AOI_5_Khartoum", "D", 500, "(f) 喀土穆同城 D (500 m)：联合通过率 (%)", "joint_pass_ratio", 100.0, "YlGn", 0, 75),
    ]
    df_all["joint_pass_ratio"] = df_all["joint_pass_cells"] / df_all["cells"]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.2))
    fig.subplots_adjust(left=0.145, right=0.96, top=0.92, bottom=0.08, wspace=0.22, hspace=0.26)

    for idx, (city, rt, sc, title, col_m, mult, cmap_name, vmin, vmax) in enumerate(panel_specs):
        ax = axes[idx // 3, idx % 3]
        sub = df_all[(df_all["city"] == city) & (df_all["route"] == rt) & (df_all["scale_m"] == sc)]
        # 每个情景聚合 2 架构 × 3 种子 × 4 起点 = 24 组运行均值
        mat = np.zeros((len(row_pairs), len(q_vals)))
        for ri, (nt, st) in enumerate(row_pairs):
            for ci, qm in enumerate(q_vals):
                cell_vals = sub[(np.isclose(sub["net_tolerance"], nt)) &
                                (np.isclose(sub["spatial_tolerance"], st)) &
                                (np.isclose(sub["Q_min"], qm))][col_m].values * mult
                mat[ri, ci] = float(cell_vals.mean())

        im = ax.imshow(mat, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
        for ri in range(len(row_pairs)):
            for ci in range(len(q_vals)):
                val = mat[ri, ci]
                txt_col = "white" if val > (vmin + (vmax - vmin) * 0.58) else "#1a202c"
                is_main = (row_pairs[ri] == (0.02, 0.05) and q_vals[ci] == 0.5)
                ax.text(ci, ri, f"{val:.1f}%", ha="center", va="center",
                        fontsize=7.6, fontweight="bold" if is_main else "normal", color=txt_col)
                if is_main:
                    ax.add_patch(Rectangle((ci - 0.49, ri - 0.49), 0.98, 0.98,
                                           fill=False, edgecolor="#1d4ed8", lw=2.0, zorder=5))

        ax.set_xticks(np.arange(len(q_vals)))
        ax.set_xticklabels(["Q≥20%", "Q≥50%★", "Q≥80%"], fontsize=7.8, fontweight="bold")
        ax.set_yticks(np.arange(len(row_pairs)))
        if idx % 3 == 0:
            ax.set_yticklabels(row_labels, fontsize=7.2)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, loc="left", fontsize=8.6, fontweight="bold", pad=6)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
        cbar.ax.tick_params(labelsize=6.8)

    fig.text(
        0.5, 0.022,
        "注：基于封存表 results/P5_03/tables/all_288_27_scenarios.csv（38,880 条情景记录中全部 7,776 条全观测域分层记录）；每个热力格为同城市×路线×尺度下 2 架构 × 3 种子 × 4 起点（n=24）的均值。蓝框★标出正文主情景（E_net≤2%, E_spatial≤5%, Q≥50%）；(a–d) 显示隐蔽空间错分在 Q 阈值变化下高度稳定，而 (e–f) 显示 500 m 尺度下提高支持度门槛（Q≥50%→80%）会显著压缩联合通过率。",
        ha="center", fontsize=7.4, color="#2d3748",
    )

    stem = "figS2_threshold_sensitivity_27_scenarios"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


def build_fig_s3(df_logs: pd.DataFrame) -> None:
    """绘制 Fig. S3: 全 12 次真实训练运行（2 架构 × 2 城市 × 3 种子）的损失与验证选模曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.09, wspace=0.18, hspace=0.28)

    epochs = np.arange(1, 13)
    city_specs = [
        ("AOI_3_Paris", "巴黎训练集 (AOI_3_Paris，有效 428 景)"),
        ("AOI_5_Khartoum", "喀土穆训练集 (AOI_5_Khartoum，有效 384 景)"),
    ]

    for ci, (city_code, city_title) in enumerate(city_specs):
        ax_loss = axes[0, ci]
        ax_score = axes[1, ci]
        for ax in (ax_loss, ax_score):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.22, ls="--")
            ax.set_xticks(epochs)
            ax.tick_params(labelsize=7.6)

        sub_c = df_logs[df_logs["city"] == city_code]
        for arch in ["UNet", "DeepLabV3Plus"]:
            col = ARCH_COLORS[arch]
            for seed, (ls, mk, _) in SEED_STYLES.items():
                sub = sub_c[(sub_c["architecture"] == arch) & (sub_c["seed"] == seed)].sort_values("epoch")
                ep = sub["epoch"].values
                tl = sub["train_loss"].values
                vl = sub["validation_loss"].values
                sc = sub["checkpoint_score"].values

                # 上排：实线/虚线为验证损失，较淡细线为训练损失
                ax_loss.plot(ep, tl, color=col, ls=ls, lw=1.0, alpha=0.42)
                ax_loss.plot(ep, vl, color=col, ls=ls, lw=1.75, marker=mk, markersize=3.8, alpha=0.92)

                # 下排：验证选模综合得分 checkpoint_score，并用星标标出真实入选 best_epoch
                ax_score.plot(ep, sc, color=col, ls=ls, lw=1.7, marker=mk, markersize=3.8, alpha=0.90)
                best_i = int(np.argmax(sc))
                ax_score.scatter([ep[best_i]], [sc[best_i]], marker="*", s=90, color=col,
                                 edgecolor="#111111", lw=0.6, zorder=6)

        ax_loss.set_ylabel("BCE+Dice 复合损失 (深=验证 / 浅=训练)", fontsize=8.0, fontweight="bold")
        ax_loss.set_title(f"({'a' if ci == 0 else 'b'}) {city_title}：训练与验证损失轨迹", loc="left", fontsize=9.0, fontweight="bold", pad=6)
        ax_loss.set_ylim(0.35, 1.55)

        ax_score.set_xlabel("训练轮次 (Epoch)", fontsize=8.2, fontweight="bold")
        ax_score.set_ylabel("验证集选模得分 (checkpoint_score)", fontsize=8.0, fontweight="bold")
        ax_score.set_title(f"({'c' if ci == 0 else 'd'}) {city_title}：验证选模得分与最优轮次 (★)", loc="left", fontsize=9.0, fontweight="bold", pad=6)
        ax_score.set_ylim(0.35, 0.78)

    handles = [
        Line2D([0], [0], color=ARCH_COLORS["UNet"], lw=2.0, label="U-Net（深线=验证损失/得分，浅细线=训练损失）"),
        Line2D([0], [0], color=ARCH_COLORS["DeepLabV3Plus"], lw=2.0, label="DeepLabV3+（深线=验证损失/得分，浅细线=训练损失）"),
        Line2D([0], [0], color="#4a5568", ls="-", marker="o", markersize=4.5, lw=1.5, label="种子 20261115"),
        Line2D([0], [0], color="#4a5568", ls="--", marker="^", markersize=4.5, lw=1.5, label="种子 20261121"),
        Line2D([0], [0], color="#4a5568", ls=":", marker="s", markersize=4.5, lw=1.5, label="种子 20261122"),
        Line2D([0], [0], color="none", marker="*", markersize=8.5, markerfacecolor="#4a5568", markeredgecolor="#111", label="真实入选最佳轮次 (best.pt)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=3, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.6, columnspacing=1.2, handlelength=2.0,
    )
    fig.text(
        0.5, 0.015,
        "注：全部 12 组曲线直接读取自本地封存 training_log.json（experiments/P3_01_FIX03/、results/P5_01/training/ 与 experiments/P5_02/training/），覆盖 2 架构 × 2 源城市 × 3 随机种子 × 12 训练轮次（共 144 个真实轮次记录）。所有模型均在独立源域验证集上依据 checkpoint_score 选择最佳权重（★），测试集全程严格隔离零泄露。",
        ha="center", fontsize=7.4, color="#2d3748",
    )

    stem = "figS3_training_loss_and_selection_curves"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


def sync_main_figures_v2() -> None:
    """将已验收的 5 张 v2 正文主图（PNG/PDF/SVG）无损同步至 paper/P6_integrated_v2/figures/。"""
    src_dir = ROOT / "figures"
    stems = [
        "fig1_study_area_and_spatial_split",
        "fig2_cross_city_asymmetric_degradation",
        "fig3_multiscale_gis_error_and_cancellation",
        "fig4_dual_error_diagnosis_and_counterexample",
        "fig5_maup_grid_origin_sensitivity",
    ]
    for st in stems:
        for ext in ("png", "pdf", "svg"):
            src = src_dir / f"{st}.{ext}"
            assert src.exists(), f"Missing approved v2 main figure: {src}"
            shutil.copy2(src, OUT_DIR / f"{st}.{ext}")
    print(f"[OK] Synced 5 approved main figures (15 files) -> {OUT_DIR}")


def main() -> None:
    df_logs = load_all_12_training_logs()
    build_fig_s1(df_logs)
    build_fig_s2()
    build_fig_s3(df_logs)
    sync_main_figures_v2()


if __name__ == "__main__":
    main()
