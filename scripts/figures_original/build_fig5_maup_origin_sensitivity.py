# -*- coding: utf-8 -*-
"""
Figure 5 (v2 论文级重构): MAUP 网格起点敏感性（4 起点空间机制 + 288 组实验 + 13/21 与 8/21 稳定性）。
严格基于 P5-02 原始起点 GPKG、P5-03 平移起点 GPKG 及封存 288 组因子表。

输出（统一写入 paper/figures_v2/，不覆盖旧版 paper/figures/）：
  - fig5_maup_grid_origin_sensitivity.png (600 dpi)
  - fig5_maup_grid_origin_sensitivity.pdf (可编辑矢量文本)
  - fig5_maup_grid_origin_sensitivity.svg (可编辑矢量文本)
"""
from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import numpy as np
from src.figure_paths import read_table
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import os
ROOT = Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT_DIR = ROOT / "figures" / "regenerated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.linewidth"] = 0.8

# 与 Figure 4 保持 100% 一致的网格双误差诊断色板
DIAG_COLORS = {
    "low_both": "#74c476",
    "net_low_spatial_high": "#d7191c",
    "net_high_spatial_low": "#fdae61",
    "high_both": "#7b3294",
}

ORIGIN_META = [
    ("original", "原始起点\n(0, 0)", "#1a202c", "o"),
    ("x_half", "东移半格\n(+250, 0)", "#d7191c", "^"),
    ("y_half", "北移半格\n(0, +250)", "#2b6cb0", "s"),
    ("xy_half", "对角半格\n(+250,+250)", "#dd6b20", "D"),
]


def add_scalebar(ax: plt.Axes, length_m: float, label: str) -> None:
    """在地图右下角绘制 UTM 比例尺。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    sx1 = x1 - dx * 0.05
    sx0 = sx1 - length_m
    sy = y0 + dy * 0.040
    ax.add_patch(
        Rectangle(
            (sx0 - dx * 0.04, sy - dy * 0.018),
            length_m + dx * 0.08,
            dy * 0.056,
            facecolor="white", alpha=0.88, edgecolor="#cbd5e0", lw=0.6, zorder=9,
        )
    )
    ax.plot([sx0, sx1], [sy, sy], color="#111111", lw=1.9, solid_capstyle="butt", zorder=10)
    th = dy * 0.008
    ax.plot([sx0, sx0], [sy - th, sy + th], color="#111111", lw=1.1, zorder=10)
    ax.plot([sx1, sx1], [sy - th, sy + th], color="#111111", lw=1.1, zorder=10)
    ax.text((sx0 + sx1) / 2.0, sy + dy * 0.010, label, color="#111111", ha="center", va="bottom",
            fontsize=6.8, fontweight="bold", zorder=10)


def add_north_arrow(ax: plt.Axes) -> None:
    """在地图左侧纯白空白区绘制指北针，避免压盖东侧观测网格。"""
    ax.annotate(
        "N",
        xy=(0.14, 0.80),
        xytext=(0.14, 0.70),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center", va="center",
        color="#111111", fontsize=7.8, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.2, mutation_scale=9.0),
        zorder=10,
    )


def main() -> None:
    gpkg_orig = ROOT / "results/P5_02/gpkg/AOI_5_Khartoum_seed20261121_P5_02.gpkg"
    gpkg_shift = ROOT / "results/P5_03/gpkg/DeepLabV3Plus_AOI_5_Khartoum_seed20261121_shifted_P5_03.gpkg"
    obs_kh = gpd.read_file(gpkg_orig, layer="test_observed_region")

    layers_500 = {
        "original": gpd.read_file(gpkg_orig, layer="D_500m_original"),
        "x_half": gpd.read_file(gpkg_shift, layer="D_500m_x_half"),
        "y_half": gpd.read_file(gpkg_shift, layer="D_500m_y_half"),
        "xy_half": gpd.read_file(gpkg_shift, layer="D_500m_xy_half"),
    }

    df_rng = read_table(ROOT / "results/P5_03/tables/origin_sensitivity_ranges.csv")
    df_zero = read_table(ROOT / "results/P5_03/tables/zero_hidden_origin_stability.csv")
    df_288 = read_table(ROOT / "results/P5_03/tables/all_288_main_scenario.csv")
    df_288_all = df_288[df_288["stratum"] == "all"].copy()
    assert len(df_288_all) == 288, f"Expected 288 main-scenario rows, got {len(df_288_all)}"

    fig = plt.figure(figsize=(15.8, 8.6))
    gs_main = fig.add_gridspec(
        1, 2, width_ratios=[1.26, 1.32], wspace=0.14,
        top=0.845, bottom=0.082, left=0.042, right=0.985,
    )

    # =========================================================================
    # 左半区 (a1–a4): 同一模型/城市/尺度（DL D/20261121 500 m）四起点真实 GIS 空间诊断地图
    # =========================================================================
    gs_maps = gs_main[0].subgridspec(1, 4, wspace=0.05)
    minx, miny, maxx, maxy = obs_kh.total_bounds
    pad_x, pad_y = 380.0, 380.0

    for idx, (orig_key, orig_lbl, _, _) in enumerate(ORIGIN_META):
        ax = fig.add_subplot(gs_maps[0, idx])
        gdf = layers_500[orig_key]
        for diag_k, color in DIAG_COLORS.items():
            part = gdf[gdf["diagnosis"] == diag_k]
            if len(part) > 0:
                part.plot(ax=ax, facecolor=color, edgecolor="#4a5568", linewidth=0.45, alpha=0.92, zorder=2)
        low_q = gdf[gdf["Q"] < 0.50]
        if len(low_q) > 0:
            low_q.plot(ax=ax, facecolor="none", edgecolor="#222222", hatch="////", linewidth=0.35, alpha=0.55, zorder=3)
        obs_kh.boundary.plot(ax=ax, color="#111111", linewidth=0.95, zorder=5)

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")
        ax.set_xticks([450500, 452500])
        ax.set_xticklabels(["450.5", "452.5"], fontsize=6.7)
        ax.set_yticks([1732000, 1734000, 1736000, 1738000, 1740000])
        if idx == 0:
            ax.set_yticklabels(["1732", "1734", "1736", "1738", "1740"], fontsize=7.2)
            ax.set_ylabel("UTM 北坐标 (km, EPSG:32636)", fontsize=8.1, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("东坐标", fontsize=7.0)
        ax.grid(True, alpha=0.18, ls=":", lw=0.5)

        hid_part = gdf[gdf["diagnosis"] == "net_low_spatial_high"]
        hid_n = len(hid_part)
        hid_pct = float(hid_part["observed_m2"].sum() / gdf["observed_m2"].sum() * 100.0)
        is_max = (orig_key == "x_half")

        short_t = ["(a1) 原始", "(a2) 东移半格★", "(a3) 北移半格", "(a4) 对角半格"][idx]
        ax.set_title(short_t, loc="left", fontsize=8.4, fontweight="bold",
                     color="#b91c1c" if is_max else "#1a202c", pad=5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#d7191c" if is_max else "#4a5568")
            sp.set_linewidth(1.5 if is_max else 0.8)

        ax.text(
            0.04, 0.985,
            f"总格数: {len(gdf)}\n抵消隐蔽: {hid_n} 格\n面积占比: {hid_pct:.2f}%",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.2, fontweight="bold" if is_max else "normal",
            color="#9b1c1c" if is_max else "#1a202c",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="#fff5f5" if is_max else "white",
                alpha=0.93,
                edgecolor="#e53e3e" if is_max else "#cbd5e0",
                lw=0.75,
            ),
            zorder=8,
        )
        if idx == 0:
            add_north_arrow(ax)
        if idx == 3:
            add_scalebar(ax, length_m=1000.0, label="1 km")

    # 在左半区正上方紧贴地图绘制总标题（严格控制在左栏宽度内）
    fig.text(
        0.265, 0.745,
        "(a) 全矩阵最大极差案例：DeepLabV3+ 喀土穆 D (种子 20261121, 500 m) 四起点对比\n"
        "隐蔽高风险面积占比：19.48% (原始) → 31.07% (东移半格★)，极差 11.59 pp",
        ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#1a202c",
    )

    # =========================================================================
    # 右半区上 (b): 288 组完整因子交叉实验四起点敏感性（巴黎细轴放大 + 喀土穆全轴）
    # =========================================================================
    gs_right = gs_main[1].subgridspec(2, 1, height_ratios=[1.06, 1.04], hspace=0.33)
    gs_b = gs_right[0].subgridspec(1, 2, wspace=0.20)

    city_configs = [
        ("AOI_3_Paris", ["A", "C"], "(b1) 巴黎 144 组：四起点隐蔽面积占比（0–13% 细尺度放大）", 13.5, ["A\n同城", "C\n跨城"]),
        ("AOI_5_Khartoum", ["D", "B"], "(b2) 喀土穆 144 组：四起点隐蔽面积占比（0–35%）", 35.5, ["D\n同城", "B\n跨城"]),
    ]

    for c_idx, (city_code, routes, title_b, ymax, r_names) in enumerate(city_configs):
        ax = fig.add_subplot(gs_b[0, c_idx])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        sub_c = df_rng[df_rng["city"] == city_code].copy()
        # 横轴按 (路线 × 尺度) 分 6 组，每组内部并排 U-Net 3 种子 + DeepLabV3+ 3 种子共 6 条竖向四起点范围线
        group_centers = []
        group_labels = []
        x_cursor = 0.0

        for r_i, rt in enumerate(routes):
            r_start = x_cursor - 0.35
            for s_i, sc in enumerate([100, 250, 500]):
                sub_g = sub_c[(sub_c["route"] == rt) & (sub_c["scale_m"] == sc)].sort_values(["architecture", "seed"])
                xs_local = []
                for _, row in sub_g.iterrows():
                    v_min = float(row["hidden_area_pct_min"])
                    v_max = float(row["hidden_area_pct_max"])
                    is_unet = (row["architecture"] == "UNet")
                    line_c = "#2b6cb0" if is_unet else "#dd6b20"
                    # 绘制单次运行的 4 起点 min-max 竖条
                    ax.plot([x_cursor, x_cursor], [v_min, v_max], color=line_c, lw=1.7, alpha=0.65, zorder=2)
                    # 绘制 4 个起点的散点
                    for o_key, _, o_col, o_mk in ORIGIN_META:
                        ov = float(row[f"hidden_area_pct_{o_key}"])
                        ax.scatter([x_cursor], [ov], marker=o_mk, s=16, color=o_col, edgecolor="white", lw=0.35, zorder=4)
                    # 高亮全矩阵最大极差组
                    if (city_code == "AOI_5_Khartoum" and rt == "D" and sc == 500
                            and row["architecture"] == "DeepLabV3Plus" and int(row["seed"]) == 20261121):
                        ax.annotate(
                            "最大极差 11.59 pp\n(19.5%→31.1%)",
                            xy=(x_cursor, v_max), xytext=(x_cursor - 2.0, 32.8),
                            ha="center", va="center", fontsize=6.3, fontweight="bold", color="#b91c1c",
                            bbox=dict(boxstyle="round,pad=0.14", facecolor="#fff5f5", edgecolor="#e53e3e", lw=0.65),
                            arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9),
                            zorder=7,
                        )
                    elif (city_code == "AOI_3_Paris" and rt == "A" and sc == 500
                          and row["architecture"] == "DeepLabV3Plus" and int(row["seed"]) == 20261121):
                        ax.annotate(
                            "巴黎最大极差 8.70 pp\n(2.7%→11.4%)",
                            xy=(x_cursor, v_max), xytext=(x_cursor + 4.8, 11.4),
                            ha="center", va="center", fontsize=6.3, fontweight="bold", color="#1e40af",
                            bbox=dict(boxstyle="round,pad=0.14", facecolor="#ebf8ff", edgecolor="#3182ce", lw=0.65),
                            arrowprops=dict(arrowstyle="->", color="#1e40af", lw=0.9),
                            zorder=7,
                        )
                    xs_local.append(x_cursor)
                    x_cursor += 0.42
                group_centers.append(float(np.mean(xs_local)))
                group_labels.append(f"{rt}\n{sc}m")
                x_cursor += 0.72
            r_end = x_cursor - 0.72 + 0.35
            bg_col = "#ebf8ff" if r_i == 0 else "#fff5f5"
            ax.axvspan(r_start, r_end, facecolor=bg_col, alpha=0.35, zorder=0)
            if r_i == 0:
                ax.axvline(r_end + 0.28, color="#94a3b8", ls="--", lw=0.85, zorder=1)
                x_cursor += 0.56

        ax.set_xticks(group_centers)
        ax.set_xticklabels(group_labels, fontsize=7.2)
        ax.set_ylim(-0.6, ymax)
        ax.set_ylabel("抵消隐蔽总观测面积占比 (%)", fontsize=7.9, fontweight="bold")
        ax.tick_params(labelsize=7.2)
        ax.grid(True, axis="y", alpha=0.22, ls="--")
        ax.set_title(title_b, loc="left", fontsize=8.4, fontweight="bold", pad=6)

        if c_idx == 0:
            ax.text(
                0.76, 0.48,
                "巴黎 C 绝大多数\n四起点维持 0%\n（少数边界格跳变\n至 0.05%–0.51%）",
                transform=ax.transAxes, ha="center", va="center", fontsize=6.3, color="#742a2a",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#feb2b2", lw=0.6, alpha=0.92),
            )

    # =========================================================================
    # 右半区下 (c): 原始起点 21 组零隐蔽例的 4 起点稳定性热力格阵（13/21 恒 0 vs 8/21 反转）
    # =========================================================================
    ax_c = fig.add_subplot(gs_right[1])
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # 按稳定性排序：先放 8 组平移后非零反转反例（醒目置顶），再放 13 组四起点恒 0
    z_sorted = df_zero.sort_values(
        ["zero_across_all_4_origins", "route", "architecture", "scale_m", "seed"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    # 用横向 21 列 × 4 行 矩阵排版（极其宽敞，一眼读清全部 21 组×4 起点的每一个格数！）
    orig_cols = ["hidden_cells_original", "hidden_cells_x_half", "hidden_cells_y_half", "hidden_cells_xy_half"]
    orig_ylabels = ["原始起点 (0,0)", "东移半格 (+½,0)", "北移半格 (0,+½)", "对角半格 (+½,+½)"]

    mat = z_sorted[orig_cols].values.T  # shape (4, 21)
    n_cases = len(z_sorted)
    n_flip = int((z_sorted["zero_across_all_4_origins"] == 0).sum())
    n_stable = int((z_sorted["zero_across_all_4_origins"] == 1).sum())
    assert n_flip == 8 and n_stable == 13, f"Expected 8 flip / 13 stable, got {n_flip}/{n_stable}"

    # 绘制状态格子：0=淡绿(稳定恒0组) 或 浅灰(反转组中的0起点)，>=1=深红(反转非零格)
    for r in range(4):
        for c in range(n_cases):
            v = int(mat[r, c])
            is_flip_col = (c < n_flip)
            if v > 0:
                fc = "#d7191c"
                tc = "white"
                fw = "bold"
            elif is_flip_col:
                fc = "#fff5f5"
                tc = "#9b1c1c"
                fw = "normal"
            else:
                fc = "#e6fffa"
                tc = "#234e52"
                fw = "normal"
            ax_c.add_patch(
                Rectangle((c - 0.45, 3 - r - 0.43), 0.90, 0.86,
                          facecolor=fc, edgecolor="#94a3b8", lw=0.6, zorder=2)
            )
            ax_c.text(c, 3 - r, str(v), ha="center", va="center", fontsize=7.3, color=tc, fontweight=fw, zorder=3)

    # 在 8 组反转与 13 组恒零之间绘制醒目分界竖线与顶部括号标签
    ax_c.axvline(n_flip - 0.5, color="#1a202c", lw=1.6, ls="--", zorder=5)
    ax_c.text(
        (n_flip - 1) / 2.0, 3.72,
        f"平移后出现非零反例：{n_flip}/21 (38.1%)\n（全部位于巴黎 A/C，单次跳变 1–3 格）",
        ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#fff5f5", edgecolor="#e53e3e", lw=0.7),
    )
    ax_c.text(
        n_flip + (n_stable - 1) / 2.0, 3.72,
        f"全部 4 起点恒维持 0 格：{n_stable}/21 (61.9%)\n（12 组巴黎 C + 1 组巴黎 A 500 m；喀土穆无 0 例）",
        ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#234e52",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#e6fffa", edgecolor="#319795", lw=0.7),
    )

    x_labels_z = []
    for _, row in z_sorted.iterrows():
        a_s = "U" if row["architecture"] == "UNet" else "DL"
        sd_s = str(int(row["seed"]))[-2:]
        x_labels_z.append(f"{a_s}-{row['route']}\n{int(row['scale_m'])}\ns{sd_s}")

    ax_c.set_xlim(-0.6, n_cases - 0.4)
    ax_c.set_ylim(-0.55, 4.45)
    ax_c.set_xticks(np.arange(n_cases))
    ax_c.set_xticklabels(x_labels_z, fontsize=6.1)
    ax_c.set_yticks([3, 2, 1, 0])
    ax_c.set_yticklabels(orig_ylabels, fontsize=7.5, fontweight="bold")
    ax_c.set_title("(c) 原始起点 21 组零隐蔽例的 4 起点稳定性矩阵（格内数字为抵消隐蔽网格数，全矩阵 288 组实验）",
                   loc="left", fontsize=8.5, fontweight="bold", pad=6)

    # 计算池化指标最大浮点舍入极差，在底注动态写明（绝不误作稳健性证据）
    max_pool_bias_rng = float(df_rng["weighted_cover_bias_range"].max())
    max_pool_esp_rng = float(df_rng["E_spatial_pooled_range"].max())
    max_pool_rng = max(max_pool_bias_rng, max_pool_esp_rng)

    # =========================================================================
    # 顶部统一分组图例：【左栏网格诊断类别】 + 【右栏架构与起点符号】
    # =========================================================================
    leg_diag = [
        mpatches.Patch(facecolor=DIAG_COLORS["low_both"], edgecolor="#444", lw=0.5, label="双低 / 联合通过 (Q≥50%)"),
        mpatches.Patch(facecolor=DIAG_COLORS["net_low_spatial_high"], edgecolor="#444", lw=0.5, label="净误差低 / 空间错误高 (抵消隐蔽)"),
        mpatches.Patch(facecolor=DIAG_COLORS["net_high_spatial_low"], edgecolor="#444", lw=0.5, label="净误差高 / 空间错误低"),
        mpatches.Patch(facecolor=DIAG_COLORS["high_both"], edgecolor="#444", lw=0.5, label="双高超限"),
        mpatches.Patch(facecolor="white", edgecolor="#222", hatch="////", lw=0.6, label="低支持度 Q<50%"),
    ]
    leg_orig = [
        Line2D([0], [0], color="#2b6cb0", lw=1.8, label="U-Net 4起点范围"),
        Line2D([0], [0], color="#dd6b20", lw=1.8, label="DeepLabV3+ 4起点范围"),
        Line2D([0], [0], color="none", marker="o", markersize=5.0, markerfacecolor="#1a202c", markeredgecolor="white", label="原始(0,0)"),
        Line2D([0], [0], color="none", marker="^", markersize=5.2, markerfacecolor="#d7191c", markeredgecolor="white", label="东移(+½,0)"),
        Line2D([0], [0], color="none", marker="s", markersize=4.8, markerfacecolor="#2b6cb0", markeredgecolor="white", label="北移(0,+½)"),
        Line2D([0], [0], color="none", marker="D", markersize=4.8, markerfacecolor="#dd6b20", markeredgecolor="white", label="对角(+½,+½)"),
    ]

    l1 = fig.legend(
        handles=leg_diag, title="【左栏 (a)：500 m 网格双误差诊断类别（与图 4 同色系）】",
        loc="upper left", bbox_to_anchor=(0.038, 0.992),
        ncol=3, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.0, title_fontsize=7.3, columnspacing=0.9, handlelength=1.4,
    )
    l1.get_title().set_fontweight("bold")

    l2 = fig.legend(
        handles=leg_orig, title="【右栏 (b)：288 组因子实验架构色与四起点标记】",
        loc="upper right", bbox_to_anchor=(0.985, 0.992),
        ncol=3, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.0, title_fontsize=7.3, columnspacing=0.9, handlelength=1.5,
    )
    l2.get_title().set_fontweight("bold")
    fig.add_artist(l1)

    fig.text(
        0.5, 0.014,
        f"注：288 组完整因子实验覆盖 2 架构 × 2 城市 × 2 路线 × 3 随机种子 × 3 尺度 (100/250/500 m) × 4 离散半格起点。左栏 (a) 呈现全矩阵最大极差案例，半格平移导致边界重切分、隐蔽高风险面积占比波动 11.59 pp；右栏 (c) 表明原始起点 21 组零例全部来自巴黎（13/21 四起点恒零、8/21 平移后出现 1–3 格非零反例）。需严格区分：全域池化偏差与 E_spatial 的四起点最大极差仅 {max_pool_rng:.2e}，属于面积守恒代数恒等式，不能作为网格分区对 MAUP 稳健的依据。",
        ha="center", fontsize=7.3, color="#2d3748",
    )

    stem = "fig5_maup_grid_origin_sensitivity"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


if __name__ == "__main__":
    main()
