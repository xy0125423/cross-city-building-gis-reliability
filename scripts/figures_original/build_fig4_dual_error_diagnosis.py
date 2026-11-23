#!/usr/bin/env python3
"""
Figure 4（v2 最终候选精修版）：E_net 与 E_spatial 双误差空间诊断（宏观 GIS 诊断地图 + 微观抵消网格解剖
+ U-Net B/20261121 (+61.38%) 过估反例真实遥感对比 + 连续决策空间与双指标反例量化）。
精修要点：
1. 顶部图例拆分为两个独立逻辑组：【组一：网格级双误差诊断类别（100 m GIS 网格）】与【组二：像元级误差分类叠层（0.3 m 原生栅格）】；
2. 在双低通过图例中明确写出 Q≥50% 联合通过条件（区分 Q<50% 斜线影纹低支持网格）；
3. 在面板 (c) 与底部图注中明确面积抵消量计算定义：2min(FP,FN) 及恒等式 E_spatial − E_net = 2min(FP,FN)/S；
4. 优化 (e) 下半面板为双 Y 轴对照：左轴柱图展示【相对 GT 建筑面积偏差 (P−A)/A (%)，分母为 A】，右轴独立黑框菱形折线展示【全域总空间错分率 E_spatial = (FP+FN)/S (%)，分母为 S】，明确两者属于不同量纲与分母指标；
5. 完整保留 4521_17342 典型抵消网格与 img1028 跨城市过估反例。
输出：paper/figures_v2/fig4_dual_error_diagnosis_and_counterexample.(pdf|svg|png[600dpi])
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from src.figure_paths import read_table
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

import os
ROOT = Path(os.environ.get('FIGURE_INPUT_ROOT', str(Path(__file__).resolve().parents[2])))
OUT_DIR = ROOT / "figures" / "regenerated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
})

DIAG_COLORS = {
    "low_both": "#74c476",              # 柔绿：双低通过
    "net_low_spatial_high": "#d7191c",  # 醒目红：净误差低 / 空间错误高（核心焦点）
    "net_high_spatial_low": "#fdae61",  # 浅橙：净误差高 / 空间错误低
    "high_both": "#8c6bb1",             # 灰紫：双高超限
}

ERR_RGBA = {
    1: (0.11, 0.62, 0.47, 0.56),  # TP：翠绿半透明
    2: (0.91, 0.16, 0.54, 0.62),  # FP：品红半透明
    3: (0.95, 0.75, 0.05, 0.66),  # FN：明黄半透明
    4: (0.00, 0.00, 0.00, 0.00),  # TN：完全透明，保留真实遥感地物纹理
}


def warp_scene_and_error_utm(
    rgb_path: Path,
    support_path: Path,
    err_path: Path,
    dst_crs: str = "EPSG:32636",
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """将单景真实 RGB 影像与原生 TP/FP/FN/TN 误差栅格同步重投影至 UTM 米制坐标（分类值严格最近邻）。"""
    with rasterio.open(rgb_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        rgb_utm = np.zeros((3, height, width), dtype=np.float32)
        for b in range(3):
            reproject(
                source=rasterio.band(src, b + 1),
                destination=rgb_utm[b],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0.0,
            )

    with rasterio.open(support_path) as sup_src:
        sup_utm = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(sup_src, 1),
            destination=sup_utm,
            src_transform=sup_src.transform,
            src_crs=sup_src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )

    with rasterio.open(err_path) as err_src:
        err_utm = np.full((height, width), 255, dtype=np.uint8)
        reproject(
            source=rasterio.band(err_src, 1),
            destination=err_utm,
            src_transform=err_src.transform,
            src_crs=err_src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=255,
            dst_nodata=255,
        )

    valid = sup_utm == 1
    img = np.transpose(rgb_utm, (1, 2, 0))
    disp = np.zeros_like(img, dtype=np.float32)
    for b in range(3):
        vals = img[:, :, b][valid]
        lo, hi = np.percentile(vals, [2, 98.5])
        disp[:, :, b] = np.clip((img[:, :, b] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    disp = np.clip(disp * 0.88 + 0.04, 0.0, 1.0)
    disp[~valid] = 0.08

    overlay = np.zeros((height, width, 4), dtype=np.float32)
    for code, rgba in ERR_RGBA.items():
        mask = (err_utm == code) & valid
        overlay[mask] = rgba

    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return disp, overlay, (left, right, bottom, top)


def add_scalebar(ax: plt.Axes, length_m: float, label: str, loc: str = "lower left", color: str = "white", bg: bool = True) -> None:
    """米制比例尺。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    sx = x0 + 0.045 * w if loc == "lower left" else x1 - 0.045 * w - length_m
    sy = y0 + 0.06 * h
    if bg:
        ax.add_patch(
            Rectangle(
                (sx - 0.02 * w, sy - 0.025 * h),
                length_m + 0.04 * w,
                0.095 * h,
                facecolor="black" if color == "white" else "white",
                alpha=0.65 if color == "white" else 0.90,
                edgecolor="#cbd5e0" if color != "white" else "none",
                lw=0.5,
                zorder=9,
            )
        )
    ax.plot([sx, sx + length_m], [sy, sy], color=color, lw=2.3, solid_capstyle="butt", zorder=10)
    ax.plot([sx, sx], [sy - 0.012 * h, sy + 0.012 * h], color=color, lw=1.1, zorder=10)
    ax.plot([sx + length_m, sx + length_m], [sy - 0.012 * h, sy + 0.012 * h], color=color, lw=1.1, zorder=10)
    ax.text(sx + length_m / 2, sy + 0.018 * h, label, color=color, ha="center", va="bottom",
            fontsize=7.6, fontweight="bold", zorder=10)


def add_north_arrow(ax: plt.Axes, color: str = "#111111", bg: bool = False) -> None:
    """指北针。"""
    if bg:
        ax.add_patch(
            Rectangle((0.885, 0.80), 0.09, 0.175, transform=ax.transAxes,
                      facecolor="black", alpha=0.55, edgecolor="none", zorder=9)
        )
    ax.annotate(
        "N", xy=(0.93, 0.945), xytext=(0.93, 0.825),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", color=color, fontsize=8.3, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=10),
        zorder=10,
    )


def main() -> None:
    g_d100 = gpd.read_file(ROOT / "results/P4_02/gpkg/AOI_5_Khartoum_P4_02.gpkg", layer="D_diagnosis_100m")
    obs_reg = gpd.read_file(ROOT / "results/P4_02/gpkg/AOI_5_Khartoum_P4_02.gpkg", layer="test_observed_region")
    own = gpd.read_file(ROOT / "results/P4_01/gpkg/AOI_5_Khartoum_P4_01.gpkg", layer="scene_ownership")
    manifest = read_table(ROOT / "configs/P3_01_FIX03_effective_manifest.csv")
    assets_1028 = read_table(ROOT / "results/P5_02/diagnostics/img1028_GIS_assets.csv")
    df_f3 = read_table(ROOT / "results/P5_02/figures/F3_source.csv")

    fig = plt.figure(figsize=(15.4, 9.6))
    gs_outer = fig.add_gridspec(
        2, 3, width_ratios=[0.95, 1.30, 1.15], height_ratios=[1.0, 1.05],
        wspace=0.19, hspace=0.27, top=0.865, bottom=0.055, left=0.045, right=0.98
    )

    # =========================================================================
    # Panel (a): 喀土穆同城 D 路线 100 m 网格双误差空间诊断地图
    # =========================================================================
    ax_a = fig.add_subplot(gs_outer[:, 0])
    for diag_key, color in DIAG_COLORS.items():
        part = g_d100[g_d100["diagnosis"] == diag_key]
        if len(part) > 0:
            part.plot(ax=ax_a, facecolor=color, edgecolor="#ffffff", linewidth=0.22, alpha=0.92, zorder=2)

    low_q = g_d100[g_d100["Q"] < 0.50]
    if len(low_q) > 0:
        low_q.plot(ax=ax_a, facecolor="none", edgecolor="#222222", hatch="////", linewidth=0.25, alpha=0.55, zorder=3)

    obs_reg.boundary.plot(ax=ax_a, color="#111111", linewidth=0.9, zorder=4)

    c_cell = g_d100[g_d100["grid_id"] == "4521_17342"].iloc[0]
    cx0, cy0, cx1, cy1 = c_cell.geometry.bounds
    ax_a.add_patch(Rectangle((cx0 - 60, cy0 - 60), (cx1 - cx0) + 120, (cy1 - cy0) + 120,
                             fill=False, edgecolor="#00e5ff", lw=2.2, zorder=7))
    ax_a.annotate(
        "① 典型抵消网格\n4521_17342\n(E_net=0.9%, E_sp=22.0%)",
        xy=(cx1 + 60, (cy0 + cy1) / 2),
        xytext=(450200, 1735150),
        fontsize=7.2, fontweight="bold", color="#003d4d",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#e6fcff", edgecolor="#00acc1", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#00838f", lw=1.2),
        zorder=8,
    )

    sc_1028 = own[own["image_id"] == "AOI_5_Khartoum_img1028"].iloc[0]
    s0, s1, s2, s3 = sc_1028.geometry.bounds
    ax_a.add_patch(Rectangle((s0 - 40, s1 - 40), (s2 - s0) + 80, (s3 - s1) + 80,
                             fill=False, edgecolor="#111111", lw=2.0, ls="--", zorder=7))
    ax_a.annotate(
        "② 反例场景 img1028\n(B/20261121 过估)",
        xy=(s0 - 40, (s1 + s3) / 2),
        xytext=(450180, 1732550),
        fontsize=7.2, fontweight="bold", color="#7a0036",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#fff0f6", edgecolor="#e7298a", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#980043", lw=1.2),
        zorder=8,
    )

    minx, miny, maxx, maxy = g_d100.total_bounds
    ax_a.set_xlim(minx - 350, maxx + 350)
    ax_a.set_ylim(miny - 300, maxy + 300)
    ax_a.set_aspect("equal")
    ax_a.set_xticks([450500, 451500, 452500, 453500])
    ax_a.set_xticklabels(["450.5", "451.5", "452.5", "453.5"], fontsize=7.6)
    yticks_a = np.arange(1732000, 1740500, 1000)
    ax_a.set_yticks(yticks_a)
    ax_a.set_yticklabels([f"{y/1000:.0f}" for y in yticks_a], fontsize=7.6)
    ax_a.set_xlabel("UTM 东坐标 (km, EPSG:32636)", fontsize=8.5)
    ax_a.set_ylabel("UTM 北坐标 (km)", fontsize=8.5)
    ax_a.grid(True, alpha=0.20, ls=":", lw=0.6)
    ax_a.set_title("(a) 喀土穆同城 D 双误差诊断地图（100 m）", loc="left", fontsize=9.6, fontweight="bold", pad=6)
    joint_n = int(g_d100["joint_scenario_pass"].sum())
    ax_a.text(
        0.03, 0.985,
        f"主情景: E_net≤2%, E_spatial≤5%, Q≥50%\n共 781 格 | 净误差过关 457 格 (联合通过 {joint_n} 格)\n净误差低/空间错误高: 77 格 (占过关格 16.85%)\n覆盖总观测面积 12.10% | 留白=未观测区",
        transform=ax_a.transAxes, ha="left", va="top", fontsize=7.0, color="#1a202c",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.92, edgecolor="#cbd5e0", lw=0.7),
        zorder=9,
    )
    add_scalebar(ax_a, length_m=1000.0, label="1 km", loc="lower right", color="#111111", bg=True)
    add_north_arrow(ax_a, color="#111111", bg=False)

    # =========================================================================
    # Panel (b) & (c): 典型抵消网格 4521_17342（img937）真实影像叠加 TP/FP/FN + 2min(FP,FN) 预算
    # =========================================================================
    r937 = manifest[manifest["image_id"] == "AOI_5_Khartoum_img937"].iloc[0]
    disp937, ov937, ext937 = warp_scene_and_error_utm(
        Path(r937["rgb_path"]),
        Path(r937["support_path"]),
        ROOT / "results/P3_02/error_maps/D/AOI_5_Khartoum_img937_errors.tif",
        dst_crs="EPSG:32636",
    )

    ax_b = fig.add_subplot(gs_outer[0, 1])
    ax_b.imshow(disp937, extent=ext937, origin="upper", interpolation="bilinear")
    ax_b.imshow(ov937, extent=ext937, origin="upper", interpolation="nearest")
    ax_b.add_patch(Rectangle((452100.0, 1734200.0), 100.0, 100.0, fill=False, edgecolor="white", lw=1.1, ls=":", zorder=5))
    gpd.GeoSeries([c_cell.geometry], crs="EPSG:32636").boundary.plot(ax=ax_b, color="#00e5ff", linewidth=2.2, zorder=6)
    ax_b.set_xlim(452072.0, 452225.0)
    ax_b.set_ylim(1734222.0, 1734360.0)
    ax_b.set_aspect("equal")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    ax_b.set_title("(b) ① 抵消网格 4521_17342 真实影像与错分叠加", loc="left", fontsize=9.4, fontweight="bold", pad=5)
    ax_b.text(
        0.025, 0.97,
        "青框=实际观测支持 S=6,863 m² (Q=68.6% ≥ 50%)\n净误差 E_net=0.89% ≤ 2%（单指标基线误判通过）\n总空间错分 E_spatial=21.99%（抵消率 C=96.0%）",
        transform=ax_b.transAxes, ha="left", va="top", fontsize=7.1, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.72, edgecolor="#00e5ff", lw=0.8),
        zorder=8,
    )
    zx0, zx1, zy0, zy1 = 452110.0, 452192.0, 1734235.0, 1734296.0
    ax_b.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="#ffea00", lw=1.6, ls="--", zorder=7))
    add_scalebar(ax_b, length_m=25.0, label="25 m", loc="lower left", color="white", bg=True)
    add_north_arrow(ax_b, color="white", bg=True)

    gs_c = gs_outer[0, 2].subgridspec(2, 1, height_ratios=[1.52, 0.75], hspace=0.26)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c1.imshow(disp937, extent=ext937, origin="upper", interpolation="nearest")
    ax_c1.imshow(ov937, extent=ext937, origin="upper", interpolation="nearest")
    gpd.GeoSeries([c_cell.geometry], crs="EPSG:32636").boundary.plot(ax=ax_c1, color="#00e5ff", linewidth=1.8, zorder=6)
    ax_c1.set_xlim(zx0, zx1)
    ax_c1.set_ylim(zy0, zy1)
    ax_c1.set_aspect("equal")
    ax_c1.set_xticks([]); ax_c1.set_yticks([])
    for sp in ax_c1.spines.values():
        sp.set_edgecolor("#d69e2e")
        sp.set_linewidth(1.3)
    ax_c1.set_title("(c) 网格内 FP / FN 空间互抵特写与 2min(FP,FN) 预算", loc="left", fontsize=9.3, fontweight="bold", pad=5)
    add_scalebar(ax_c1, length_m=20.0, label="20 m", loc="lower left", color="white", bg=True)

    ax_c2 = fig.add_subplot(gs_c[1, 0])
    ax_c2.spines["top"].set_visible(False)
    ax_c2.spines["right"].set_visible(False)
    cats = ["净面积误差\n|FP − FN|", "总空间错分\nFP + FN"]
    y_pos = [0, 1]
    net_val = float(c_cell["abs_bias_m2"])
    tot_val = float(c_cell["error_m2"])
    cancel_val = tot_val - net_val  # 2 * min(FP, FN)
    ax_c2.barh(0, net_val, height=0.52, color="#2b6cb0", edgecolor="#222222", lw=0.6)
    ax_c2.barh(1, net_val, height=0.52, color="#2b6cb0", edgecolor="#222222", lw=0.6, label="净面积差 |FP−FN| (61 m²)")
    ax_c2.barh(1, cancel_val, left=net_val, height=0.52, color="#d7191c", alpha=0.82, edgecolor="#222222", lw=0.6, label="抵消量 2min(FP,FN) (1,448 m²)")
    ax_c2.text(net_val + 35, 0, "61 m² (E_net=0.89%)", va="center", fontsize=7.5, fontweight="bold", color="#2b6cb0")
    ax_c2.text(tot_val + 35, 1, "1,509 m² (E_sp=21.99%)", va="center", fontsize=7.5, fontweight="bold", color="#b91c1c")
    ax_c2.set_yticks(y_pos)
    ax_c2.set_yticklabels(cats, fontsize=7.8)
    ax_c2.set_xlim(0, 2320)
    ax_c2.set_xlabel("面积 (m²) | FP=785, FN=724 → 2min(FP,FN)=1,448 m²", fontsize=7.6)
    ax_c2.tick_params(axis="x", labelsize=7.4)
    ax_c2.grid(True, axis="x", alpha=0.25, ls="--")
    ax_c2.legend(fontsize=6.7, loc="lower right", frameon=True, facecolor="white", framealpha=0.94)

    # =========================================================================
    # Panel (d): ② 反例场景 img1028（同城 D vs 跨城 U-Net B/20261121 过估反例）
    # =========================================================================
    gs_d = gs_outer[1, 1].subgridspec(1, 2, wspace=0.06)
    rgb1028 = Path(assets_1028.iloc[0]["RGB"])
    sup1028 = Path(assets_1028.iloc[0]["support"])
    err_d21 = ROOT / assets_1028[(assets_1028["architecture"] == "UNet") & (assets_1028["seed"] == 20261121) & (assets_1028["route"] == "D") & (assets_1028["kind"] == "error")].iloc[0]["path"]
    err_b21 = ROOT / assets_1028[(assets_1028["architecture"] == "UNet") & (assets_1028["seed"] == 20261121) & (assets_1028["route"] == "B") & (assets_1028["kind"] == "error")].iloc[0]["path"]

    disp1028, ov_d21, ext1028 = warp_scene_and_error_utm(rgb1028, sup1028, err_d21, dst_crs="EPSG:32636")
    _, ov_b21, _ = warp_scene_and_error_utm(rgb1028, sup1028, err_b21, dst_crs="EPSG:32636")

    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d1.imshow(disp1028, extent=ext1028, origin="upper", interpolation="bilinear")
    ax_d1.imshow(ov_d21, extent=ext1028, origin="upper", interpolation="nearest")
    ax_d1.set_xticks([]); ax_d1.set_yticks([])
    ax_d1.set_aspect("equal")
    ax_d1.set_title("(d) ② 反例区 img1028：同城 D vs 跨城 B/20261121", loc="left", fontsize=9.3, fontweight="bold", pad=5)
    ax_d1.text(
        0.03, 0.97, "U-Net 同城 D (20261121)\nIoU=35.8% | Recall=62.9%\nFP=2,882 m² | FN=1,412 m²",
        transform=ax_d1.transAxes, ha="left", va="top", fontsize=6.9, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.68, edgecolor="none"),
    )
    add_scalebar(ax_d1, length_m=50.0, label="50 m", loc="lower left", color="white", bg=True)

    ax_d2 = fig.add_subplot(gs_d[0, 1])
    ax_d2.imshow(disp1028, extent=ext1028, origin="upper", interpolation="bilinear")
    ax_d2.imshow(ov_b21, extent=ext1028, origin="upper", interpolation="nearest")
    ax_d2.set_xticks([]); ax_d2.set_yticks([])
    ax_d2.set_aspect("equal")
    for sp in ax_d2.spines.values():
        sp.set_edgecolor("#e7298a")
        sp.set_linewidth(1.5)
    ax_d2.text(
        0.03, 0.97, "U-Net 跨城 B (20261121 反例)\nIoU=11.9% | Recall=72.9% ↑\nFP=19,528 m²（连片过估）",
        transform=ax_d2.transAxes, ha="left", va="top", fontsize=6.9, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#7a0036", alpha=0.82, edgecolor="#ff80ab", lw=0.7),
    )
    add_north_arrow(ax_d2, color="white", bg=True)

    # =========================================================================
    # Panel (e): 连续双误差决策空间 + 双指标独立坐标轴量化 B 路线过估反例
    # =========================================================================
    gs_e = gs_outer[1, 2].subgridspec(2, 1, height_ratios=[1.28, 0.92], hspace=0.36)
    ax_e1 = fig.add_subplot(gs_e[0, 0])
    ax_e1.spines["top"].set_visible(False)
    ax_e1.spines["right"].set_visible(False)

    en = g_d100["E_net"].values * 100.0
    es = g_d100["E_spatial"].values * 100.0
    qv = g_d100["Q"].values * 100.0

    ax_e1.add_patch(Rectangle((0, 5.0), 2.0, 35.0, facecolor="#d7191c", alpha=0.11, zorder=1))
    ax_e1.add_patch(Rectangle((0, 0.0), 2.0, 5.0, facecolor="#74c476", alpha=0.12, zorder=1))
    xx = np.linspace(0, 26, 100)
    ax_e1.plot(xx, xx, color="#555555", ls=":", lw=1.0, zorder=2)
    ax_e1.axvline(2.0, color="#111111", ls="--", lw=1.05, zorder=3)
    ax_e1.axhline(5.0, color="#111111", ls="--", lw=1.05, zorder=3)

    sc = ax_e1.scatter(en, es, c=qv, cmap="viridis_r", s=11, alpha=0.75, edgecolors="none", vmin=0, vmax=100, zorder=4)
    ax_e1.scatter([c_cell["E_net"] * 100], [c_cell["E_spatial"] * 100], s=84, marker="*",
                  color="#00e5ff", edgecolor="#111111", lw=0.8, zorder=6)
    ax_e1.annotate(
        "① 4521_17342\n(0.9%, 22.0%)",
        xy=(c_cell["E_net"] * 100, c_cell["E_spatial"] * 100),
        xytext=(12.2, 26.8),
        fontsize=7.1, fontweight="bold", color="#005066",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#e6fcff", edgecolor="#00acc1", lw=0.75),
        arrowprops=dict(arrowstyle="->", color="#00838f", lw=1.0),
        zorder=7,
    )
    ax_e1.text(2.4, 34.8, "净误差低 / 空间错误高\n(77 格被净差掩盖)", fontsize=7.0, fontweight="bold", color="#b91c1c", va="top")
    ax_e1.set_xlim(-0.3, 24.0)
    ax_e1.set_ylim(-0.5, 36.8)
    ax_e1.set_xlabel("净覆盖率误差 E_net (%)", fontsize=8.0)
    ax_e1.set_ylabel("总空间错分率 E_spatial (%)", fontsize=8.0)
    ax_e1.tick_params(labelsize=7.4)
    ax_e1.grid(True, alpha=0.22, ls="--")
    ax_e1.set_title("(e) 双误差决策空间与跨城过估反例双指标对照", loc="left", fontsize=9.2, fontweight="bold", pad=5)
    cbar = fig.colorbar(sc, ax=ax_e1, fraction=0.046, pad=0.02)
    cbar.set_label("支持度 Q (%)", fontsize=7.4)
    cbar.ax.tick_params(labelsize=6.9)

    # 下半面板 (e2): 双 Y 轴明确分离【左轴柱：相对 GT 面积偏差 (P−A)/A】与【右轴菱形线：总空间错分率 E_spatial=(FP+FN)/S】
    ax_e2 = fig.add_subplot(gs_e[1, 0])
    ax_e2.spines["top"].set_visible(False)
    df_kb = df_f3[(df_f3["city"] == "AOI_5_Khartoum") & (df_f3["route"] == "B")].copy()
    order = [
        ("UNet", 20261115, "U-15"),
        ("UNet", 20261121, "U-21★"),
        ("UNet", 20261122, "U-22"),
        ("DeepLabV3Plus", 20261115, "DL-15"),
        ("DeepLabV3Plus", 20261121, "DL-21"),
        ("DeepLabV3Plus", 20261122, "DL-22"),
    ]
    vals, cols, xlabs, esps = [], [], [], []
    for arch, sd, lbl in order:
        r = df_kb[(df_kb["architecture"] == arch) & (df_kb["seed"] == sd)].iloc[0]
        v = float(r["rel_area_pct"])
        esp_v = (float(r["fp_m2"]) + float(r["fn_m2"])) / float(r["observed_m2"]) * 100.0
        vals.append(v)
        esps.append(esp_v)
        xlabs.append(lbl)
        if arch == "UNet" and sd == 20261121:
            cols.append("#d7191c")
        elif arch == "UNet":
            cols.append("#2b6cb0")
        else:
            cols.append("#dd6b20")
    x2 = np.arange(len(vals))
    ax_e2.bar(x2 - 0.14, vals, width=0.44, color=cols, edgecolor="#222222", lw=0.6,
              label="柱(左轴): 相对GT偏差 (P−A)/A")
    ax_e2.axhline(0, color="#111111", lw=0.85)
    for xi, v in zip(x2, vals):
        if xi == 1:
            # 红色过估反例柱：+61.4% 置于红柱中上部白色粗体，彻底避开上方紫菱形 21.7%
            ax_e2.text(xi - 0.14, v * 0.52, f"{v:+.1f}%", ha="center", va="center",
                       fontsize=6.6, fontweight="bold", color="white", zorder=6)
        elif xi == 3:
            # DL-15 (+88.4%)：置于橙柱中上部白色粗体，避开上方紫菱形 24.0%
            ax_e2.text(xi - 0.14, v * 0.56, f"{v:+.1f}%", ha="center", va="center",
                       fontsize=6.5, fontweight="bold", color="white", zorder=6)
        else:
            va = "bottom" if v >= 0 else "top"
            dy = 4.5 if v >= 0 else -5.0
            ax_e2.text(xi - 0.14, v + dy, f"{v:+.1f}%", ha="center", va=va,
                       fontsize=6.5, color="#1a202c", zorder=6)
    ax_e2.set_xticks(x2)
    ax_e2.set_xticklabels(xlabs, fontsize=7.3)
    ax_e2.set_ylim(-122, 152)
    ax_e2.set_ylabel("相对GT偏差 (P−A)/A (%)", fontsize=7.4, color="#1a202c")
    ax_e2.tick_params(axis="y", labelsize=7.0)
    ax_e2.grid(True, axis="y", alpha=0.20, ls="--")

    # 右轴：全域总空间错分率 E_spatial = (FP+FN)/S (%)
    ax_e2_r = ax_e2.twinx()
    ax_e2_r.spines["top"].set_visible(False)
    ax_e2_r.spines["right"].set_color("#553c9a")
    ax_e2_r.plot(x2 + 0.18, esps, color="#553c9a", ls="--", marker="D", markersize=4.6, lw=1.35,
                 label="◆(右轴): 总空间错分 E_sp=(FP+FN)/S", zorder=7)
    esp_offsets = {
        0: (0.18, 1.8, "bottom"),
        1: (0.22, 1.9, "bottom"),
        2: (0.18, 1.8, "bottom"),
        3: (0.24, 1.9, "bottom"),
        4: (0.18, 1.8, "bottom"),
        5: (0.18, -2.4, "top"),
    }
    for xi, ev in zip(x2, esps):
        dx_e, dy_e, va_e = esp_offsets[xi]
        ax_e2_r.text(
            xi + dx_e, ev + dy_e, f"{ev:.1f}%",
            ha="center", va=va_e,
            fontsize=6.2, fontweight="bold" if xi in (1, 3) else "normal",
            color="#553c9a",
            bbox=dict(boxstyle="square,pad=0.06", facecolor="white", alpha=0.82, edgecolor="none"),
            zorder=8,
        )
    ax_e2_r.set_ylim(0, 38)
    ax_e2_r.set_ylabel("E_spatial = (FP+FN)/S (%)", fontsize=7.3, color="#553c9a")
    ax_e2_r.tick_params(axis="y", labelsize=7.0, colors="#553c9a")

    h1, l1 = ax_e2.get_legend_handles_labels()
    h2, l2 = ax_e2_r.get_legend_handles_labels()
    ax_e2.legend(h1 + h2, l1 + l2, fontsize=6.1, loc="upper left", frameon=True, facecolor="white", framealpha=0.94, borderpad=0.35)

    # =========================================================================
    # 顶栏分两组逻辑图例盒：【网格级双误差诊断类别】 vs 【像元级 TP/FP/FN 分类】
    # =========================================================================
    leg_group1 = [
        mpatches.Patch(facecolor=DIAG_COLORS["low_both"], edgecolor="#444", lw=0.5, label="双低 / 联合通过 (E_net≤2%, E_spatial≤5%, Q≥50%)"),
        mpatches.Patch(facecolor=DIAG_COLORS["net_low_spatial_high"], edgecolor="#444", lw=0.5, label="净误差低 / 空间错误高 (E_net≤2%, E_spatial>5%，抵消隐蔽)"),
        mpatches.Patch(facecolor=DIAG_COLORS["net_high_spatial_low"], edgecolor="#444", lw=0.5, label="净误差高 / 空间错误低 (E_net>2%, E_spatial≤5%)"),
        mpatches.Patch(facecolor=DIAG_COLORS["high_both"], edgecolor="#444", lw=0.5, label="双高超限 (E_net>2%, E_spatial>5%)"),
        mpatches.Patch(facecolor="white", edgecolor="#222", hatch="////", lw=0.6, label="低观测支持 (Q<50%，斜线影纹，不计入联合通过)"),
    ]
    leg_group2 = [
        mpatches.Patch(facecolor=(0.11, 0.62, 0.47, 0.72), edgecolor="#222", lw=0.5, label="TP 正确检测建筑物"),
        mpatches.Patch(facecolor=(0.91, 0.16, 0.54, 0.78), edgecolor="#222", lw=0.5, label="FP 假阳性（相对参考多出预测）"),
        mpatches.Patch(facecolor=(0.95, 0.75, 0.05, 0.80), edgecolor="#222", lw=0.5, label="FN 假阴性（漏检参考建筑）"),
        mpatches.Patch(facecolor="#e2e8f0", edgecolor="#4a5568", lw=0.5, label="TN 正确背景（透明展示真实遥感底图）"),
    ]

    leg1 = fig.legend(
        handles=leg_group1,
        title="【组一：网格级双误差诊断类别（100 m GIS 网格）】",
        loc="upper left", bbox_to_anchor=(0.015, 0.990),
        ncol=2, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.3, title_fontsize=7.6, columnspacing=1.0, handlelength=1.5
    )
    leg1.get_title().set_fontweight("bold")

    leg2 = fig.legend(
        handles=leg_group2,
        title="【组二：像元级误差分类叠层（0.3 m 原生遥感栅格）】",
        loc="upper right", bbox_to_anchor=(0.985, 0.990),
        ncol=2, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.3, title_fontsize=7.6, columnspacing=1.0, handlelength=1.5
    )
    leg2.get_title().set_fontweight("bold")
    fig.add_artist(leg1)

    fig.text(
        0.5, -0.018,
        "注：在同一有效观测支持面积 S 上，净覆盖率误差 E_net = |P−A|/S = |FP−FN|/S，总空间错分率 E_spatial = (FP+FN)/S；被净差抵消的空间错分面积定义为 2min(FP,FN)，满足恒等式 E_spatial − E_net = 2min(FP,FN)/S。面板 (e) 下图中相对 GT 面积偏差 (P−A)/A（左轴柱，分母为参考建筑面积 A）与全域 E_spatial（右轴◆线，分母为有效观测面积 S）属于不同分母与物理含义的评价指标。",
        ha="center", fontsize=7.6, color="#2d3748"
    )

    stem = "fig4_dual_error_diagnosis_and_counterexample"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


if __name__ == "__main__":
    main()
