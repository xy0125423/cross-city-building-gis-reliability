#!/usr/bin/env python3
"""
Figure 1（v2 最终候选精修版）：研究区空间隔离划分、宏观地理定位插图、真实高分辨率遥感影像、
建筑物参考轮廓及局部放大。
精修要点：
1. 核实并显式注明喀土穆 Train 数量：P3-00 原划分为 385 景，FIX03 排除 img894（1 景）后有效训练为 384 景，并在地图上标出 img894 位置；
2. 明确两条红色虚线切线与每条切线两侧 ±100 m 切线缓冲带的关系（区分切线间距 2.0/3.5 km 的 Val 条带与 ±100 m 缓冲禁区）；
3. 基于本机真实 Natural Earth 矢量（naturalearth_lowres.shp）增加巴黎（法国）与喀土穆（苏丹/欧非跨纬度）宏观地理定位插图；
4. 保留真实 PS-RGB 重投影影像、GT 轮廓、3 倍局部放大与米制比例尺。
输出：paper/figures_v2/fig1_study_area_and_spatial_split.(pdf|svg|png[600dpi])
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd
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
    "xtick.direction": "out",
    "ytick.direction": "out",
})

SPLIT_COLORS = {
    "train": "#2b6cb0",          # 沉稳钢蓝（有效训练集）
    "validation": "#38a169",     # 翠绿（验证集）
    "test": "#dd6b20",           # 焦橙（测试集）
    "buffer": "#9e9e9e",         # 中灰（跨切线/缓冲整组隔离场景）
    "excluded": "#e2e6ea",       # 极浅灰（质量排除 <80% / 开发留出）
    "img894": "#805ad5",         # 醒目紫（FIX03 单独排除的 img894，原属 P3-00 Train）
}


def find_natural_earth_shp() -> Path | None:
    """Use the explicitly packaged public-domain locator boundaries."""
    candidate = ROOT / "figure_inputs/naturalearth/naturalearth_lowres.shp"
    return candidate if candidate.is_file() else None


def reproject_rgb_to_utm(rgb_path: Path, support_path: Path, dst_crs: str) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """将原生 EPSG:4326 RGB 影像重投影至米制 UTM（还原真实长宽比），并仅在有效支持区做 2%–98.5% 对比度拉伸。"""
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
                src_nodata=None,
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

    valid = sup_utm == 1
    img = np.transpose(rgb_utm, (1, 2, 0))
    disp = np.zeros_like(img, dtype=np.float32)
    for b in range(3):
        vals = img[:, :, b][valid]
        lo, hi = np.percentile(vals, [2, 98.5])
        disp[:, :, b] = np.clip((img[:, :, b] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    disp[~valid] = 0.08
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return disp, (left, right, bottom, top)


def add_scalebar(ax: plt.Axes, length_m: float, label: str, loc: str = "lower left", color: str = "white", bg: bool = True) -> None:
    """在 UTM 米制地图或影像面板上绘制精确物理比例尺。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    if loc == "lower left":
        sx = x0 + 0.05 * w
        sy = y0 + 0.065 * h
    elif loc == "lower right":
        sx = x1 - 0.05 * w - length_m
        sy = y0 + 0.065 * h
    else:
        sx = x0 + 0.36 * w
        sy = y0 + 0.065 * h
    if bg:
        pad_x = 0.02 * w
        pad_y = 0.025 * h
        ax.add_patch(
            Rectangle(
                (sx - pad_x, sy - pad_y),
                length_m + 2 * pad_x,
                0.088 * h,
                facecolor="black" if color == "white" else "white",
                alpha=0.62 if color == "white" else 0.90,
                edgecolor="#cbd5e0" if color != "white" else "none",
                lw=0.5,
                zorder=9,
            )
        )
    ax.plot([sx, sx + length_m], [sy, sy], color=color, lw=2.4, solid_capstyle="butt", zorder=10)
    ax.plot([sx, sx], [sy - 0.012 * h, sy + 0.012 * h], color=color, lw=1.2, zorder=10)
    ax.plot([sx + length_m, sx + length_m], [sy - 0.012 * h, sy + 0.012 * h], color=color, lw=1.2, zorder=10)
    ax.text(
        sx + length_m / 2,
        sy + 0.018 * h,
        label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        zorder=10,
    )


def add_north_arrow(ax: plt.Axes, color: str = "#111111", bg: bool = False) -> None:
    """在右上角绘制简洁清晰的指北针。"""
    if bg:
        ax.add_patch(
            Rectangle(
                (0.885, 0.80), 0.09, 0.175,
                transform=ax.transAxes, facecolor="black", alpha=0.55, edgecolor="none", zorder=9
            )
        )
    ax.annotate(
        "N",
        xy=(0.93, 0.94),
        xytext=(0.93, 0.825),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        color=color,
        fontsize=8.5,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, mutation_scale=10),
        zorder=10,
    )


def add_macro_locator_inset(ax: plt.Axes, world: gpd.GeoDataFrame, mode: str) -> None:
    """在空间划分地图的空白角嵌入基于真实 Natural Earth 矢量的宏观地理定位插图。"""
    if mode == "paris":
        # 置于 (a) 巴黎地图左下空白区
        ins = ax.inset_axes([0.025, 0.035, 0.35, 0.31], zorder=8)
        world.plot(ax=ins, facecolor="#edf2f7", edgecolor="#a0aec0", linewidth=0.4)
        fr = world[world["name"] == "France"]
        fr.plot(ax=ins, facecolor="#bee3f8", edgecolor="#2b6cb0", linewidth=0.8)
        ins.scatter([2.35], [48.85], marker="*", s=55, color="#e53e3e", edgecolor="white", lw=0.5, zorder=5)
        ins.text(2.35, 47.7, "巴黎\n(48.8°N)", fontsize=6.1, fontweight="bold", color="#9b1c1c", ha="center", va="top")
        ins.set_xlim(-6.5, 11.0)
        ins.set_ylim(41.2, 52.2)
        ins.set_title("法国宏观定位", fontsize=6.5, fontweight="bold", pad=2.0, color="#1a202c")
    else:
        # 置于 (d) 喀土穆地图右下纯白空白区：同时呈现欧非跨纬度双城关系
        ins = ax.inset_axes([0.565, 0.03, 0.415, 0.365], zorder=8)
        world.plot(ax=ins, facecolor="#edf2f7", edgecolor="#a0aec0", linewidth=0.35)
        world[world["name"] == "France"].plot(ax=ins, facecolor="#bee3f8", edgecolor="#2b6cb0", linewidth=0.65)
        world[world["name"] == "Sudan"].plot(ax=ins, facecolor="#feebc8", edgecolor="#dd6b20", linewidth=0.75)
        ins.scatter([2.35], [48.85], marker="o", s=24, color="#2b6cb0", edgecolor="white", lw=0.5, zorder=5)
        ins.scatter([32.56], [15.55], marker="*", s=58, color="#e53e3e", edgecolor="white", lw=0.5, zorder=5)
        # 双向跨城迁移连接虚线
        ins.annotate(
            "", xy=(31.5, 16.8), xytext=(3.8, 47.5),
            arrowprops=dict(arrowstyle="<->", color="#9b1c1c", lw=0.95, ls="--", connectionstyle="arc3,rad=-0.18"),
            zorder=4,
        )
        ins.text(3.8, 50.4, "巴黎", fontsize=6.0, fontweight="bold", color="#2b6cb0")
        ins.text(28.5, 6.2, "喀土穆 (15.6°N)", fontsize=5.8, fontweight="bold", color="#9b1c1c", ha="center")
        ins.text(21.5, 34.0, "跨纬度~33°\n(~4,600 km)", fontsize=5.6, color="#742a2a", ha="center",
                 bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.86, edgecolor="none"))
        ins.set_xlim(-10.0, 46.0)
        ins.set_ylim(2.5, 56.0)
        ins.set_title("欧—非跨区域定位", fontsize=6.4, fontweight="bold", pad=1.8, color="#1a202c")

    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor("#d6eaf8")
    for sp in ins.spines.values():
        sp.set_edgecolor("#4a5568")
        sp.set_linewidth(0.75)


def plot_spatial_split_map(
    ax: plt.Axes,
    gpkg_path: Path,
    manifest: pd.DataFrame,
    world: gpd.GeoDataFrame | None,
    city: str,
    title: str,
    cut1: float,
    cut2: float,
    sample_id: str,
    extra_mark_id: str | None,
    min_dist_text: str,
    latlon_text: str,
) -> None:
    """绘制真实 UTM 空间划分地图（含全部场景、双切线、±100 m 切线缓冲带与代表场景精准定位）。"""
    scenes = gpd.read_file(gpkg_path, layer="scenes")
    sub_m = manifest[manifest["city"] == city][["image_id", "split", "effective_split"]].copy()
    gdf = scenes.merge(sub_m, on="image_id", how="left")

    def map_cat(row) -> str:
        sp = row["effective_split"]
        if sp in ("train", "validation", "test", "buffer"):
            return sp
        if sp == "excluded_ambiguous_edge_partial_no_pixel_center":
            return "img894"
        return "excluded"

    gdf["cat"] = gdf.apply(map_cat, axis=1)

    draw_order = [
        ("excluded", SPLIT_COLORS["excluded"], "#bfc6ce", 0.35, 0.90),
        ("buffer", SPLIT_COLORS["buffer"], "#6e6e6e", 0.45, 0.88),
        ("train", SPLIT_COLORS["train"], "#ffffff", 0.35, 0.90),
        ("validation", SPLIT_COLORS["validation"], "#ffffff", 0.35, 0.90),
        ("test", SPLIT_COLORS["test"], "#ffffff", 0.35, 0.92),
        ("img894", SPLIT_COLORS["img894"], "#111111", 0.9, 1.0),
    ]
    for cat, fc, ec, lw, alpha in draw_order:
        part = gdf[gdf["cat"] == cat]
        if len(part) > 0:
            part.plot(ax=ax, facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=2 if cat != "img894" else 7)

    minx, miny, maxx, maxy = gdf.total_bounds
    dx = maxx - minx
    dy = maxy - miny
    if dy / max(dx, 1.0) > 1.6:
        target_dx = dy * 0.86
        extra_x = (target_dx - dx) / 2.0
        ax.set_xlim(minx - extra_x * 0.65, maxx + extra_x * 1.35)
        ax.set_ylim(miny - dy * 0.05, maxy + dy * 0.10)
        is_narrow = True
    else:
        ax.set_xlim(minx - dx * 0.14, maxx + dx * 0.08)
        ax.set_ylim(miny - dy * 0.07, maxy + dy * 0.08)
        is_narrow = False

    xlim0, xlim1 = ax.get_xlim()
    ylim0, ylim1 = ax.get_ylim()

    # 绘制两条切线及其各自两侧真实的 ±100 m（共 200 m 宽）半透明切线缓冲禁区
    cut_specs = [
        (cut1, f"切线 y={cut1/1000:.1f} km (±100 m缓冲带)", "top" if is_narrow else "bottom"),
        (cut2, f"切线 y={cut2/1000:.1f} km (±100 m缓冲带)", "bottom"),
    ]
    for cut_y, lbl, va_mode in cut_specs:
        ax.axhspan(cut_y - 100.0, cut_y + 100.0, color="#e53e3e", alpha=0.34, zorder=3)
        ax.axhline(cut_y, color="#9b1c1c", ls="--", lw=1.25, zorder=4)
        offset = (-220 if va_mode == "top" else 220) if is_narrow else 120
        ax.text(
            xlim0 + (xlim1 - xlim0) * 0.022,
            cut_y + offset,
            lbl,
            ha="left",
            va=va_mode,
            fontsize=6.7,
            fontweight="bold",
            color="#9b1c1c",
            bbox=dict(boxstyle="square,pad=0.10", facecolor="white", alpha=0.90, edgecolor="none"),
            zorder=6,
        )

    # 在两条切线之间用真实双向箭头与清晰标签标出 Validation 条带宽度，避免被误认为 2 km / 3.5 km 缓冲带
    mid_y = (cut1 + cut2) / 2.0
    band_km = (cut2 - cut1) / 1000.0
    arr_x = xlim1 - (xlim1 - xlim0) * (0.37 if is_narrow else 0.34)
    ax.annotate(
        "", xy=(arr_x, cut2 - 110.0), xytext=(arr_x, cut1 + 110.0),
        arrowprops=dict(arrowstyle="<->", color="#22543d", lw=1.05),
        zorder=6,
    )
    ax.text(
        xlim1 - (xlim1 - xlim0) * 0.025,
        mid_y,
        f"Val条带 (宽{band_km:.1f} km)\n跨切线整块归Buffer",
        ha="right", va="center", fontsize=6.4, color="#22543d", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#f0fff4", alpha=0.93, edgecolor="#38a169", lw=0.6),
        zorder=6,
    )

    # 高亮代表遥感影像样本精确位置
    s_row = gdf[gdf["image_id"] == sample_id]
    if len(s_row) > 0:
        s_row.boundary.plot(ax=ax, color="#ffea00", linewidth=2.2, zorder=7)
        cx, cy = s_row.geometry.iloc[0].centroid.coords[0]
        ax.scatter([cx], [cy], marker="*", s=75, color="#ffea00", edgecolor="#111111", lw=0.7, zorder=8)
        if is_narrow:
            tx, ty = xlim0 + (xlim1 - xlim0) * 0.035, cy - dy * 0.02
        else:
            tx, ty = xlim0 + (xlim1 - xlim0) * 0.03, cy + dy * 0.08
        ax.annotate(
            f"影像样例\n({sample_id.split('_')[-1]})",
            xy=(cx, cy),
            xytext=(tx, ty),
            ha="left",
            fontsize=7.1,
            fontweight="bold",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="#fffde7", edgecolor="#b7791f", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#111111", lw=1.0),
            zorder=9,
        )

    # 如为喀土穆，标注图4反例区 img1028 及 FIX03 排除的 img894（解释 385 -> 384）
    if extra_mark_id is not None:
        e_row = gdf[gdf["image_id"] == extra_mark_id]
        if len(e_row) > 0:
            ex, ey = e_row.geometry.iloc[0].centroid.coords[0]
            ax.scatter([ex], [ey], marker="D", s=28, color="#e7298a", edgecolor="white", lw=0.6, zorder=8)
            ax.annotate(
                "图4反例区\n(img1028)",
                xy=(ex, ey),
                xytext=(xlim1 - (xlim1 - xlim0) * 0.03, ey + dy * 0.085),
                ha="right",
                fontsize=6.9,
                color="#980043",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff0f6", edgecolor="#e7298a", lw=0.75),
                arrowprops=dict(arrowstyle="->", color="#980043", lw=0.95),
                zorder=9,
            )
        r894 = gdf[gdf["image_id"] == "AOI_5_Khartoum_img894"]
        if len(r894) > 0:
            x8, y8 = r894.geometry.iloc[0].centroid.coords[0]
            ax.scatter([x8], [y8], marker="o", s=36, facecolor="none", edgecolor="#6b46c1", lw=1.6, zorder=8)
            ax.annotate(
                "FIX03排除: img894\n(Train 385→384)",
                xy=(x8, y8),
                xytext=(xlim0 + (xlim1 - xlim0) * 0.03, y8 + dy * 0.04),
                ha="left",
                fontsize=6.5,
                color="#553c9a",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.16", facecolor="#faf5ff", edgecolor="#805ad5", lw=0.7),
                arrowprops=dict(arrowstyle="->", color="#553c9a", lw=0.9),
                zorder=9,
            )

    # 宏观地理定位插图
    if world is not None:
        add_macro_locator_inset(ax, world, mode="khartoum" if is_narrow else "paris")

    ax.set_aspect("equal")
    if is_narrow:
        ax.set_xticks([444000, 448000, 452000, 456000])
        ax.set_xticklabels(["444", "448", "452", "456"], fontsize=7.6)
        ax.set_yticks([1715000, 1720000, 1725000, 1730000, 1735000, 1740000])
        ax.set_yticklabels(["1715", "1720", "1725", "1730", "1735", "1740"], fontsize=7.6)
    else:
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticks([t for t in xticks if xlim0 + 400 <= t <= xlim1 - 400])
        ax.set_yticks([t for t in yticks if ylim0 + 400 <= t <= ylim1 - 400])
        ax.set_xticklabels([f"{t/1000:.0f}" for t in ax.get_xticks()], fontsize=7.6)
        ax.set_yticklabels([f"{t/1000:.0f}" for t in ax.get_yticks()], fontsize=7.6)
    ax.set_xlabel("UTM 东坐标 (km)", fontsize=8.5)
    ax.set_ylabel("UTM 北坐标 (km)", fontsize=8.5)
    ax.grid(True, alpha=0.22, ls=":", lw=0.6)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=5)

    ax.text(
        0.025, 0.978,
        f"{latlon_text}\n{min_dist_text}",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=6.65 if is_narrow else 7.0, color="#1a202c",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", alpha=0.92, edgecolor="#cbd5e0", lw=0.7),
        zorder=9,
    )
    add_scalebar(
        ax, length_m=2000.0, label="2 km",
        loc="lower left" if is_narrow else "lower right",
        color="#111111", bg=True
    )
    add_north_arrow(ax, color="#111111", bg=False)


def plot_imagery_and_zoom(
    ax_full: plt.Axes,
    ax_zoom: plt.Axes,
    manifest: pd.DataFrame,
    image_id: str,
    dst_crs: str,
    title_full: str,
    title_zoom: str,
    zoom_frac: tuple[float, float, float, float],
    gsd_note: str,
) -> None:
    """绘制真实 UTM 重投影遥感影像 + 真实 GeoJSON 建筑参考轮廓 + 局部 3 倍放大视窗。"""
    row = manifest[manifest["image_id"] == image_id].iloc[0]
    rgb_disp, (left, right, bottom, top) = reproject_rgb_to_utm(
        Path(row["rgb_path"]), Path(row["support_path"]), dst_crs
    )
    gt = gpd.read_file(row["geojson_path"])
    if gt.crs is None:
        gt = gt.set_crs("EPSG:4326")
    gt_utm = gt.to_crs(dst_crs)
    gt_utm = gt_utm[~gt_utm.is_empty & gt_utm.geometry.notna()]

    ax_full.imshow(rgb_disp, extent=[left, right, bottom, top], origin="upper", interpolation="bilinear")
    gt_utm.boundary.plot(ax=ax_full, color="#00f5d4", linewidth=0.95, zorder=4)
    ax_full.set_xlim(left, right)
    ax_full.set_ylim(bottom, top)
    ax_full.set_aspect("equal")
    ax_full.set_xticks([])
    ax_full.set_yticks([])
    ax_full.set_title(title_full, loc="left", fontsize=9.4, fontweight="bold", pad=5)

    w, h = right - left, top - bottom
    zx0 = left + zoom_frac[0] * w
    zx1 = left + zoom_frac[1] * w
    zy0 = bottom + zoom_frac[2] * h
    zy1 = bottom + zoom_frac[3] * h

    ax_full.add_patch(
        Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, edgecolor="#ff3b30", lw=1.8, zorder=6)
    )
    add_scalebar(ax_full, length_m=50.0, label="50 m", loc="lower left", color="white", bg=True)
    add_north_arrow(ax_full, color="white", bg=True)

    ax_zoom.imshow(rgb_disp, extent=[left, right, bottom, top], origin="upper", interpolation="nearest")
    gt_utm.boundary.plot(ax=ax_zoom, color="#00f5d4", linewidth=1.45, zorder=4)
    ax_zoom.set_xlim(zx0, zx1)
    ax_zoom.set_ylim(zy0, zy1)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    for sp in ax_zoom.spines.values():
        sp.set_edgecolor("#ff3b30")
        sp.set_linewidth(1.4)
    ax_zoom.set_title(title_zoom, loc="left", fontsize=9.4, fontweight="bold", pad=5)
    ax_zoom.text(
        0.03, 0.96, gsd_note,
        transform=ax_zoom.transAxes, ha="left", va="top",
        fontsize=7.4, color="white", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="black", alpha=0.65, edgecolor="none"),
        zorder=8,
    )
    add_scalebar(ax_zoom, length_m=20.0, label="20 m", loc="lower left", color="white", bg=True)

    for (p1, p2) in [((zx1, zy1), (zx0, zy1)), ((zx1, zy0), (zx0, zy0))]:
        con = ConnectionPatch(
            xyA=p1, coordsA=ax_full.transData,
            xyB=p2, coordsB=ax_zoom.transData,
            color="#ff3b30", lw=1.0, ls="--", alpha=0.85
        )
        ax_full.figure.add_artist(con)


def main() -> None:
    manifest = read_table(ROOT / "configs/P3_01_FIX03_effective_manifest.csv")
    ne_path = find_natural_earth_shp()
    world = gpd.read_file(ne_path) if ne_path is not None else None

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.36, 1.0, 0.92],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.28,
    )

    # 第 1 行：巴黎（AOI_3_Paris）
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    plot_spatial_split_map(
        ax_a,
        ROOT / "figure_inputs/P1_01/AOI_3_Paris_P1_01.gpkg",
        manifest,
        world=world,
        city="AOI_3_Paris",
        title="(a) 巴黎研究区空间隔离划分（EPSG:32631）",
        cut1=5430500.0,
        cut2=5432500.0,
        sample_id="AOI_3_Paris_img516",
        extra_mark_id=None,
        min_dist_text="Train–Test 最小间距: 2.93 km",
        latlon_text="巴黎 (48.8°N, 2.3°E) | 428 / 150 / 111 景",
    )
    plot_imagery_and_zoom(
        ax_b, ax_c, manifest,
        image_id="AOI_3_Paris_img516",
        dst_crs="EPSG:32631",
        title_full="(b) 巴黎真实 PS-RGB 影像与 GT 轮廓（img516）",
        title_zoom="(c) 巴黎局部放大（欧式坡屋顶肌理）",
        zoom_frac=(0.20, 0.72, 0.35, 0.75),
        gsd_note="实测像元: 0.20 m × 0.30 m\n82 栋参考建筑 (支持度 100%)",
    )

    # 第 2 行：喀土穆（AOI_5_Khartoum）
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])
    plot_spatial_split_map(
        ax_d,
        ROOT / "figure_inputs/P1_01/AOI_5_Khartoum_P1_01.gpkg",
        manifest,
        world=world,
        city="AOI_5_Khartoum",
        title="(d) 喀土穆研究区空间隔离划分（EPSG:32636）",
        cut1=1728000.0,
        cut2=1731500.0,
        sample_id="AOI_5_Khartoum_img952",
        extra_mark_id="AOI_5_Khartoum_img1028",
        min_dist_text="Train–Test 最小间距: 4.66 km",
        latlon_text="喀土穆 (15.6°N, 32.5°E) | 384(原385)/141/131",
    )
    plot_imagery_and_zoom(
        ax_e, ax_f, manifest,
        image_id="AOI_5_Khartoum_img952",
        dst_crs="EPSG:32636",
        title_full="(e) 喀土穆真实 PS-RGB 影像与 GT 轮廓（img952）",
        title_zoom="(f) 喀土穆局部放大（干旱区平顶肌理）",
        zoom_frac=(0.25, 0.65, 0.25, 0.65),
        gsd_note="实测像元: 0.29 m × 0.30 m\n107 栋参考建筑 (支持度 100%)",
    )

    # 顶栏统一紧凑图例（全部数量均严格按“巴黎 / 喀土穆”顺序标明）
    handles = [
        mpatches.Patch(facecolor=SPLIT_COLORS["train"], edgecolor="#555555", lw=0.5, label="有效训练集 Train（巴黎 428 / 喀土穆 384 [P3-00原385]）"),
        mpatches.Patch(facecolor=SPLIT_COLORS["validation"], edgecolor="#555555", lw=0.5, label="验证集 Val（巴黎 150 / 喀土穆 141，两切线间条带）"),
        mpatches.Patch(facecolor=SPLIT_COLORS["test"], edgecolor="#555555", lw=0.5, label="测试集 Test（巴黎 111 / 喀土穆 131）"),
        mpatches.Patch(facecolor=SPLIT_COLORS["buffer"], edgecolor="#666666", lw=0.5, label="空间隔离场景 Buffer（巴黎 300 / 喀土穆 71，跨切线整块不拆分）"),
        mpatches.Patch(facecolor=SPLIT_COLORS["excluded"], edgecolor="#adb5bd", lw=0.5, label="质量排除+开发留出（巴黎 159 [144+15] / 喀土穆 284 [262+22]）"),
        mpatches.Patch(facecolor=SPLIT_COLORS["img894"], edgecolor="#111111", lw=0.6, label="FIX03 单景排除（喀土穆 img894: 边缘残片无像元中心，1 景）"),
        plt.Line2D([0], [0], color="#9b1c1c", ls="--", lw=1.4, label="空间切线及每条切线两侧 ±100 m 切线缓冲带（红半透明窄带）"),
        plt.Line2D([0], [0], color="#00f5d4", lw=1.8, label="SpaceNet 2 建筑物参考轮廓 (GT)"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.998),
        ncol=4, frameon=False, fontsize=7.5, columnspacing=0.85, handlelength=1.5
    )
    fig.text(
        0.5, -0.015,
        "注：全部场景矢量与训练/验证/测试划分严格对应 configs/P3_01_FIX03_effective_manifest.csv（共 2,160 景）；每条红色虚线切线两侧各设 ±100 m 切线缓冲带（图中 200 m 宽红半透明窄带），完整 500 m 块与连通组不拆分。",
        ha="center", fontsize=7.8, color="#333333"
    )

    stem = "fig1_study_area_and_spatial_split"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


if __name__ == "__main__":
    main()
