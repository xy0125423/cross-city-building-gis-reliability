# -*- coding: utf-8 -*-
"""
Figure 3 (v2 论文级重构): 多尺度 GIS 建筑覆盖误差、FP/FN 空间抵消与覆盖密度分层。
严格基于 P4-01 封存多尺度 GeoPackage、密度分层表与 P5-02 跨架构多随机种子 F4 源表。

输出（统一写入 paper/figures_v2/，不覆盖旧版 paper/figures/）：
  - fig3_multiscale_gis_error_and_cancellation.png (600 dpi)
  - fig3_multiscale_gis_error_and_cancellation.pdf (可编辑矢量文本)
  - fig3_multiscale_gis_error_and_cancellation.svg (可编辑矢量文本)
"""
from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import numpy as np
from src.figure_paths import read_table
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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

ARCH_COLORS = {"UNet": "#2b6cb0", "DeepLabV3Plus": "#dd6b20"}
ROUTE_STYLES = {
    ("AOI_3_Paris", "A"): ("#1d4ed8", "-", "o", "巴黎同城 A"),
    ("AOI_3_Paris", "C"): ("#60a5fa", "--", "^", "巴黎跨城 C"),
    ("AOI_5_Khartoum", "D"): ("#c05621", "-", "s", "喀土穆同城 D"),
    ("AOI_5_Khartoum", "B"): ("#e53e3e", "--", "D", "喀土穆跨城 B"),
}


def add_scalebar(ax: plt.Axes, length_m: float, label: str) -> None:
    """在地图右下角绘制 UTM 比例尺。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    sx1 = x1 - dx * 0.06
    sx0 = sx1 - length_m
    sy = y0 + dy * 0.042
    ax.add_patch(
        Rectangle(
            (sx0 - dx * 0.04, sy - dy * 0.018),
            length_m + dx * 0.08,
            dy * 0.055,
            facecolor="white", alpha=0.88, edgecolor="#cbd5e0", lw=0.6, zorder=9,
        )
    )
    ax.plot([sx0, sx1], [sy, sy], color="#111111", lw=1.9, solid_capstyle="butt", zorder=10)
    th = dy * 0.008
    ax.plot([sx0, sx0], [sy - th, sy + th], color="#111111", lw=1.1, zorder=10)
    ax.plot([sx1, sx1], [sy - th, sy + th], color="#111111", lw=1.1, zorder=10)
    ax.text((sx0 + sx1) / 2.0, sy + dy * 0.010, label, color="#111111", ha="center", va="bottom",
            fontsize=7.0, fontweight="bold", zorder=10)


def add_north_arrow(ax: plt.Axes) -> None:
    """在地图右上角绘制指北针。"""
    ax.annotate(
        "N",
        xy=(0.90, 0.96),
        xytext=(0.90, 0.865),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center", va="center",
        color="#111111", fontsize=8.0, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.25, mutation_scale=9.5),
        zorder=10,
    )


def main() -> None:
    gpkg_kh = ROOT / "results/P4_01/gpkg/AOI_5_Khartoum_P4_01.gpkg"
    obs_kh = gpd.read_file(gpkg_kh, layer="test_observed_region")
    grids = {
        100: gpd.read_file(gpkg_kh, layer="D_grid_100m"),
        250: gpd.read_file(gpkg_kh, layer="D_grid_250m"),
        500: gpd.read_file(gpkg_kh, layer="D_grid_500m"),
    }
    df_f4 = read_table(ROOT / "results/P5_02/figures/F4_source.csv")
    df_fn = read_table(ROOT / "results/P4_01/tables/FN_concentration_vs_GT_exposure.csv")

    fig = plt.figure(figsize=(15.6, 9.6))
    gs_main = fig.add_gridspec(
        1, 2, width_ratios=[1.12, 1.32], wspace=0.16,
        top=0.925, bottom=0.075, left=0.048, right=0.982,
    )

    # =========================================================================
    # 左半区 (a)-(c): 完整喀土穆同城 D 三尺度真实 GIS 网格有符号覆盖率偏差空间地图
    # =========================================================================
    gs_left = gs_main[0].subgridspec(2, 1, height_ratios=[1.0, 0.048], hspace=0.10)
    gs_maps = gs_left[0].subgridspec(1, 3, wspace=0.06)

    minx, miny, maxx, maxy = obs_kh.total_bounds
    pad_x, pad_y = 320.0, 320.0
    norm = TwoSlopeNorm(vmin=-18.0, vcenter=0.0, vmax=18.0)
    cmap = mpl.colormaps["RdBu_r"]

    # 从 F4 提取 P4-01 对应的 U-Net seed=20261115 喀土穆 D 汇总统计用于精确信息盒标注
    f4_u15_d = df_f4[(df_f4["architecture"] == "UNet") & (df_f4["seed"] == 20261115) & (df_f4["route"] == "D")].set_index("scale_m")

    map_axes = []
    for idx, sc in enumerate([100, 250, 500]):
        ax = fig.add_subplot(gs_maps[0, idx])
        map_axes.append(ax)
        gdf = grids[sc].copy()
        gdf["bias_pp"] = gdf["cover_bias"] * 100.0

        gdf.plot(
            ax=ax, column="bias_pp", cmap=cmap, norm=norm,
            edgecolor="#4a5568" if sc == 500 else "#94a3b8",
            linewidth=0.55 if sc == 500 else (0.35 if sc == 250 else 0.16),
            zorder=2,
        )
        obs_kh.boundary.plot(ax=ax, color="#111111", linewidth=0.95, zorder=4)

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_aspect("equal")
        ax.set_xticks([450500, 451500, 452500])
        ax.set_xticklabels(["450.5", "451.5", "452.5"], fontsize=7.0)
        ax.set_yticks([1732000, 1734000, 1736000, 1738000, 1740000])
        if idx == 0:
            ax.set_yticklabels(["1732", "1734", "1736", "1738", "1740"], fontsize=7.3)
            ax.set_ylabel("UTM 北坐标 (km, EPSG:32636)", fontsize=8.2, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("UTM 东坐标 (km)", fontsize=7.4)
        ax.grid(True, alpha=0.20, ls=":", lw=0.55)

        panel_lbl = ["(a)", "(b)", "(c)"][idx]
        ax.set_title(f"{panel_lbl} {sc} m 网格 (喀土穆同城 D)", loc="left", fontsize=8.8, fontweight="bold", pad=5)

        n_cells = len(gdf)
        q_med = float(gdf["observed_fraction"].median()) * 100.0
        part_pct = float(gdf["partial_cell"].mean()) * 100.0
        mae_pp = float(f4_u15_d.loc[sc, "weighted_cover_MAE"]) * 100.0
        canc_pct = float(f4_u15_d.loc[sc, "cancel_pct"])

        stat_box = (
            f"格数: {n_cells} (部分覆盖 {part_pct:.0f}%)\n"
            f"支持度 Q 中位: {q_med:.1f}%\n"
            f"加权MAE: {mae_pp:.2f} pp\n"
            f"格内抵消率 C: {canc_pct:.1f}%"
        )
        ax.text(
            0.03, 0.985, stat_box,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4, color="#1a202c", fontweight="bold" if sc == 500 else "normal",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="#fff5f5" if sc == 500 else "white",
                alpha=0.92,
                edgecolor="#e53e3e" if sc == 500 else "#cbd5e0",
                lw=0.75,
            ),
            zorder=8,
        )
        if idx == 2:
            add_scalebar(ax, length_m=1000.0, label="1 km")
            add_north_arrow(ax)

    # 左半区底部共享发散色带
    cax = fig.add_subplot(gs_left[1])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("网格有符号建筑覆盖率偏差 Pred − GT (百分点, pp) | 蓝=低估, 红=高估, 留白=未观测区", fontsize=7.6, fontweight="bold")
    cb.ax.tick_params(labelsize=7.3)
    cb.set_ticks([-18, -12, -6, 0, 6, 12, 18])

    # =========================================================================
    # 右半区上 (d): 多尺度加权覆盖率 MAE 与 汇总格内抵消率 C（跨架构×3种子）
    # =========================================================================
    gs_right = gs_main[1].subgridspec(2, 1, height_ratios=[1.02, 1.05], hspace=0.30)
    gs_d = gs_right[0].subgridspec(1, 2, wspace=0.22)

    scales = [100, 250, 500]
    x_sc = np.array([0, 1, 2])

    # (d1) 多尺度面积加权覆盖率 MAE (%)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d1.spines["top"].set_visible(False)
    ax_d1.spines["right"].set_visible(False)

    # (d2) 多尺度格内 FP/FN 空间抵消率 C (%)
    ax_d2 = fig.add_subplot(gs_d[0, 1])
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_visible(False)

    for (city, rt), (col, ls, mk, lbl) in ROUTE_STYLES.items():
        # 按尺度聚合两架构×3种子（共 6 组运行）均值与真实 min-max 范围，同时以不同符号画个体点
        sub_r = df_f4[(df_f4["city"] == city) & (df_f4["route"] == rt)]
        for ax_cur, col_metric, scale_mult in [
            (ax_d1, "weighted_cover_MAE", 100.0),
            (ax_d2, "cancel_pct", 1.0),
        ]:
            means, lows, highs = [], [], []
            for si, sc in enumerate(scales):
                vals = sub_r[sub_r["scale_m"] == sc][col_metric].values * scale_mult
                means.append(float(vals.mean()))
                lows.append(float(vals.min()))
                highs.append(float(vals.max()))
                # 叠加个体运行点（U-Net 实心, DeepLabV3+ 空心）
                for arch, fc_mode, dx_a in [("UNet", col, -0.055), ("DeepLabV3Plus", "white", 0.055)]:
                    v_a = sub_r[(sub_r["scale_m"] == sc) & (sub_r["architecture"] == arch)][col_metric].values * scale_mult
                    ax_cur.scatter(
                        np.full_like(v_a, si + dx_a, dtype=float), v_a,
                        marker=mk, s=22, facecolor=fc_mode, edgecolor=col, lw=0.9, alpha=0.85, zorder=4,
                    )
            ax_cur.fill_between(x_sc, lows, highs, color=col, alpha=0.12, zorder=1)
            ax_cur.plot(x_sc, means, color=col, ls=ls, lw=1.9, marker=mk, markersize=5.2,
                        markerfacecolor=col, markeredgecolor="white", markeredgewidth=0.6, label=lbl, zorder=5)

    # 动态提取各组精确极差数值，确保图内标注与底注 100% 与 F4_source.csv 一致
    kh_b_all = df_f4[(df_f4["city"] == "AOI_5_Khartoum") & (df_f4["route"] == "B")]
    kh_b_mae_lo = float(kh_b_all["weighted_cover_MAE"].min()) * 100.0
    kh_b_mae_hi = float(kh_b_all["weighted_cover_MAE"].max()) * 100.0

    kh_d_500 = df_f4[(df_f4["city"] == "AOI_5_Khartoum") & (df_f4["route"] == "D") & (df_f4["scale_m"] == 500)]
    pa_a_500 = df_f4[(df_f4["city"] == "AOI_3_Paris") & (df_f4["route"] == "A") & (df_f4["scale_m"] == 500)]
    pa_c_all = df_f4[(df_f4["city"] == "AOI_3_Paris") & (df_f4["route"] == "C")]
    in_500_lo = min(float(kh_d_500["cancel_pct"].min()), float(pa_a_500["cancel_pct"].min()))
    in_500_hi = max(float(kh_d_500["cancel_pct"].max()), float(pa_a_500["cancel_pct"].max()))
    pa_c_lo = float(pa_c_all["cancel_pct"].min())
    pa_c_hi = float(pa_c_all["cancel_pct"].max())

    ax_d1.set_xticks(x_sc)
    ax_d1.set_xticklabels(["100 m", "250 m", "500 m"], fontsize=8.0)
    ax_d1.set_ylim(-0.3, 16.2)
    ax_d1.set_ylabel("面积加权覆盖率 MAE (pp)", fontsize=8.1, fontweight="bold")
    ax_d1.set_title("(d1) 多尺度覆盖率 MAE（均值线 + 全 6 次运行极差带）", loc="left", fontsize=8.6, fontweight="bold", pad=6)
    ax_d1.tick_params(labelsize=7.5)
    ax_d1.grid(True, axis="y", alpha=0.22, ls="--")
    ax_d1.text(
        1.0, 15.6,
        f"喀土穆 B 跨种子离散度极大\n(MAE {kh_b_mae_lo:.1f}–{kh_b_mae_hi:.1f} pp，含过估反例)",
        ha="center", va="top", fontsize=6.5, color="#9b1c1c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#fff5f5", edgecolor="#feb2b2", lw=0.6),
    )

    ax_d2.set_xticks(x_sc)
    ax_d2.set_xticklabels(["100 m", "250 m", "500 m"], fontsize=8.0)
    ax_d2.set_ylim(-3.0, 90.0)
    ax_d2.set_ylabel("汇总格内 FP/FN 空间抵消率 C (%)", fontsize=8.1, fontweight="bold")
    ax_d2.set_title("(d2) 多尺度格内误差抵消率 C（同城陡升 vs 巴黎 C 近零）", loc="left", fontsize=8.6, fontweight="bold", pad=6)
    ax_d2.tick_params(labelsize=7.5)
    ax_d2.grid(True, axis="y", alpha=0.22, ls="--")
    ax_d2.annotate(
        f"同城 A/D 500 m 抵消达 {in_500_lo:.1f}–{in_500_hi:.1f}%\n净误差收敛主要由格内互抵驱动",
        xy=(2.0, float(kh_d_500["cancel_pct"].mean())), xytext=(0.15, 79.5),
        fontsize=6.4, color="#9c4221", fontweight="bold", ha="left",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#fffaf0", edgecolor="#fbd38d", lw=0.6),
        arrowprops=dict(arrowstyle="->", color="#c05621", lw=0.9),
    )
    ax_d2.text(
        1.05, 8.8,
        f"巴黎 C 单向全漏检\nC 仅 {pa_c_lo:.1f}%–{pa_c_hi:.1f}%",
        ha="center", va="bottom", fontsize=6.5, color="#1e40af", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ebf8ff", edgecolor="#90cdf4", lw=0.6),
    )

    # (d1)/(d2) 共享紧凑图例置于右区顶部
    leg_d = [
        Line2D([0], [0], color="#1d4ed8", ls="-", lw=1.8, marker="o", markersize=5, label="巴黎同城 A"),
        Line2D([0], [0], color="#60a5fa", ls="--", lw=1.8, marker="^", markersize=5, label="巴黎跨城 C"),
        Line2D([0], [0], color="#c05621", ls="-", lw=1.8, marker="s", markersize=5, label="喀土穆同城 D"),
        Line2D([0], [0], color="#e53e3e", ls="--", lw=1.8, marker="D", markersize=5, label="喀土穆跨城 B"),
        Line2D([0], [0], color="none", marker="o", markersize=4.8, markerfacecolor="#4a5568", markeredgecolor="#4a5568", label="实心=U-Net"),
        Line2D([0], [0], color="none", marker="o", markersize=4.8, markerfacecolor="white", markeredgecolor="#4a5568", label="空心=DeepLabV3+"),
    ]
    fig.legend(
        handles=leg_d, loc="upper right", bbox_to_anchor=(0.982, 0.992),
        ncol=6, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.1, columnspacing=0.9, handlelength=1.8,
    )

    # =========================================================================
    # 右半区下 (e): 100 m 覆盖密度分层（GT 暴露份额 vs FN 漏检份额 + 条件漏检率 FN/GT）
    # =========================================================================
    gs_e = gs_right[1].subgridspec(1, 2, wspace=0.22)
    bands = ["0-5%", "5-15%", "15-30%", ">30%"]
    xb = np.arange(len(bands))
    w = 0.34

    for p_idx, (city_c, rt_c, title_e, bar_gt, bar_fn, line_col) in enumerate([
        ("AOI_3_Paris", "C", "(e1) 巴黎跨城 C：100 m 密度分层暴露与漏检", "#90cdf4", "#2b6cb0", "#9b1c1c"),
        ("AOI_5_Khartoum", "B", "(e2) 喀土穆跨城 B：100 m 密度分层暴露与漏检", "#fbd38d", "#c05621", "#9b1c1c"),
    ]):
        ax = fig.add_subplot(gs_e[0, p_idx])
        ax.spines["top"].set_visible(False)
        sub = df_fn[(df_fn["city"] == city_c) & (df_fn["route"] == rt_c)].set_index("band")
        gt_sh = [float(sub.loc[b, "GT_area_share"]) * 100.0 for b in bands]
        fn_sh = [float(sub.loc[b, "FN_area_share"]) * 100.0 for b in bands]
        cond_fn = [float(sub.loc[b, "conditional_FN_fraction"]) * 100.0 for b in bands]
        n_c = [int(sub.loc[b, "cells"]) for b in bands]

        ax.bar(xb - w / 2, gt_sh, width=w, color=bar_gt, edgecolor="#2d3748", lw=0.6, label="GT份额: 层内GT / 全区总GT")
        ax.bar(xb + w / 2, fn_sh, width=w, color=bar_fn, edgecolor="#2d3748", lw=0.6, label="FN份额: 层内FN / 全区总FN")
        for i in range(len(bands)):
            ax.text(xb[i] - w / 2, gt_sh[i] + 0.9, f"{gt_sh[i]:.1f}", ha="center", va="bottom", fontsize=6.3, color="#2d3748")
            ax.text(xb[i] + w / 2, fn_sh[i] + 0.9, f"{fn_sh[i]:.1f}", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#1a202c")

        ax.set_xticks(xb)
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bands, n_c)], fontsize=7.3)
        ax.set_ylim(0, 68.0)
        if p_idx == 0:
            ax.set_ylabel("占全区总 GT 或总 FN 份额 (%)", fontsize=7.9, fontweight="bold")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.3)
        ax.grid(True, axis="y", alpha=0.22, ls="--")
        ax.set_title(title_e, loc="left", fontsize=8.6, fontweight="bold", pad=6)

        # 右轴：条件漏检率 = 层内 FN / 层内 GT (%) —— 仅右侧 (e2) 显示右轴标题，彻底消除中间碰撞
        ax_r = ax.twinx()
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_color(line_col)
        ax_r.plot(xb, cond_fn, color=line_col, ls="-", lw=1.75, marker="D", markersize=5.0,
                  markerfacecolor="#fff5f5", markeredgecolor=line_col, markeredgewidth=1.2,
                  label="条件漏检率: 层内FN / 层内GT (右轴)", zorder=6)
        for i, cv in enumerate(cond_fn):
            if p_idx == 0:
                dy_c, va_c = 1.15, "bottom"
            else:
                dy_c, va_c = (1.15, "bottom") if i != 3 else (-1.8, "top")
            ax_r.text(xb[i], cv + dy_c, f"{cv:.1f}%", ha="center", va=va_c,
                      fontsize=6.5, fontweight="bold", color=line_col,
                      bbox=dict(boxstyle="square,pad=0.05", facecolor="white", alpha=0.84, edgecolor="none"),
                      zorder=7)
        ax_r.set_ylim(74.0, 106.5)
        if p_idx == 1:
            ax_r.set_ylabel("条件漏检率 FN / GT (%)", fontsize=7.9, fontweight="bold", color=line_col)
            ax_r.tick_params(axis="y", labelsize=7.2, colors=line_col)
        else:
            ax_r.set_yticklabels([])
            ax_r.tick_params(axis="y", length=2, colors=line_col)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left" if p_idx == 1 else "upper right",
                  bbox_to_anchor=(0.01, 0.99) if p_idx == 1 else (0.99, 0.78),
                  fontsize=6.2, frameon=True, facecolor="white", framealpha=0.94, borderpad=0.3)

        if p_idx == 1:
            ax_r.annotate(
                "高密层条件漏检更低\n(89.8% < 97.2%)",
                xy=(3.0, cond_fn[3]), xytext=(0.68, 86.2),
                ha="center", va="center", fontsize=6.4, fontweight="bold", color="#9b1c1c",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#fff5f5", edgecolor="#feb2b2", lw=0.65),
                arrowprops=dict(arrowstyle="->", color="#9b1c1c", lw=0.9, connectionstyle="arc3,rad=0.12"),
                zorder=8,
            )

    mae100 = float(f4_u15_d.loc[100, "weighted_cover_MAE"]) * 100.0
    mae500 = float(f4_u15_d.loc[500, "weighted_cover_MAE"]) * 100.0
    c100 = float(f4_u15_d.loc[100, "cancel_pct"])
    c500 = float(f4_u15_d.loc[500, "cancel_pct"])
    q500 = float(grids[500]["observed_fraction"].median()) * 100.0

    fig.text(
        0.5, 0.014,
        f"注：左栏 (a–c) 基于 P4-01 喀土穆同城 D（U-Net, 种子 20261115）完整有效观测域真实多尺度 GIS 网格；尺度从 100 m 放粗至 500 m 时，表观加权 MAE 由 {mae100:.2f} pp 降至 {mae500:.2f} pp，格内 FP/FN 抵消率 C 由 {c100:.1f}% 升至 {c500:.1f}%，且 500 m 全部 64 格均为部分覆盖（Q 中位数 {q500:.1f}%）。右栏 (e) 排除参考空网格（巴黎 364 格、喀土穆 215 格）按 100 m GT 覆盖率分层，明确三项不同分母：GT 份额 = 层内 GT / 全区总 GT；FN 份额 = 层内 FN / 全区总 FN；条件漏检率 = 层内 FN / 层内 GT。",
        ha="center", fontsize=7.25, color="#2d3748",
    )

    stem = "fig3_multiscale_gis_error_and_cancellation"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


if __name__ == "__main__":
    main()
