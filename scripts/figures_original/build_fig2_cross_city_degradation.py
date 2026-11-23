# -*- coding: utf-8 -*-
"""
Figure 2 (v2 论文级重构): 跨城市建筑物提取性能与非对称退化。
严格基于 P3-02 封存误差栅格、空间组 Bootstrap 配对差值表及 P5-02 跨架构多随机种子真实评估数据。

输出（统一写入 paper/figures_v2/，不覆盖旧版 paper/figures/）：
  - fig2_cross_city_asymmetric_degradation.png (600 dpi)
  - fig2_cross_city_asymmetric_degradation.pdf (可编辑矢量文本)
  - fig2_cross_city_asymmetric_degradation.svg (可编辑矢量文本)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from src.figure_paths import read_table
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
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

ARCH_COLORS = {"UNet": "#2b6cb0", "DeepLabV3Plus": "#dd6b20"}
ARCH_MARKERS = {"UNet": "o", "DeepLabV3Plus": "s"}


def stretch_rgb_uint16(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """对真实 16 位 PS-RGB 影像执行 2%–98% 百分位线性拉伸。"""
    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.float32)
    for c in range(3):
        band = rgb[c].astype(np.float32)
        vals = band[valid]
        if vals.size > 0:
            lo, hi = np.percentile(vals, [2, 98])
            if hi <= lo:
                lo, hi = float(vals.min()), float(vals.max()) + 1e-6
            out[:, :, c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    out[~valid] = 1.0
    return out


def warp_scene_and_error_utm(
    rgb_path: Path,
    sup_path: Path,
    err_path: Path,
    dst_crs: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], dict[str, float]]:
    """将真实 RGB 与分类错误栅格重投影至目标 UTM 坐标系，并返回像元统计。"""
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3])
        src_crs = src.crs
        src_transform = src.transform
        src_w, src_h = src.width, src.height
        src_bounds = src.bounds
    with rasterio.open(sup_path) as s_sup:
        sup = s_sup.read(1) > 0
    with rasterio.open(err_path) as s_err:
        err = s_err.read(1)

    tp = int((err == 1).sum())
    fp = int((err == 2).sum())
    fn = int((err == 3).sum())
    iou = tp / max(tp + fp + fn, 1) * 100.0
    prec = tp / max(tp + fp, 1) * 100.0
    rec = tp / max(tp + fn, 1) * 100.0
    stats = {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "prec": prec, "rec": rec}

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, src_w, src_h, *src_bounds
    )
    dst_rgb = np.zeros((3, dst_h, dst_w), dtype=np.uint16)
    for c in range(3):
        reproject(
            source=rgb[c],
            destination=dst_rgb[c],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    dst_sup = np.zeros((dst_h, dst_w), dtype=np.uint8)
    reproject(
        source=sup.astype(np.uint8),
        destination=dst_sup,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    dst_err = np.full((dst_h, dst_w), 255, dtype=np.uint8)
    reproject(
        source=err,
        destination=dst_err,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )

    disp = stretch_rgb_uint16(dst_rgb, dst_sup > 0)
    overlay = np.zeros((dst_h, dst_w, 4), dtype=np.float32)
    # 统一采用与 Figure 4 完全一致的色盲友好像元错分色板
    overlay[dst_err == 1] = [0.11, 0.62, 0.47, 0.72]  # TP 青绿
    overlay[dst_err == 2] = [0.91, 0.16, 0.54, 0.78]  # FP 洋红
    overlay[dst_err == 3] = [0.95, 0.75, 0.05, 0.80]  # FN 明黄

    x0 = dst_transform.c
    y1 = dst_transform.f
    x1 = x0 + dst_w * dst_transform.a
    y0 = y1 + dst_h * dst_transform.e
    return disp, overlay, (x0, x1, y0, y1), stats


def add_scalebar(ax: plt.Axes, length_m: float, label: str) -> None:
    """在遥感影像左下角绘制带深色半透明底框的真实 UTM 比例尺。"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    sx0 = x0 + dx * 0.045
    sy = y0 + dy * 0.075
    ax.add_patch(
        Rectangle(
            (sx0 - dx * 0.018, sy - dy * 0.032),
            length_m + dx * 0.036,
            dy * 0.105,
            facecolor="black", alpha=0.60, edgecolor="none", zorder=9,
        )
    )
    ax.plot([sx0, sx0 + length_m], [sy, sy], color="white", lw=2.1, solid_capstyle="butt", zorder=10)
    tick_h = dy * 0.016
    ax.plot([sx0, sx0], [sy - tick_h, sy + tick_h], color="white", lw=1.2, zorder=10)
    ax.plot([sx0 + length_m, sx0 + length_m], [sy - tick_h, sy + tick_h], color="white", lw=1.2, zorder=10)
    ax.text(sx0 + length_m / 2.0, sy + dy * 0.020, label, color="white", ha="center", va="bottom",
            fontsize=7.3, fontweight="bold", zorder=10)


def add_north_arrow(ax: plt.Axes) -> None:
    """在遥感影像右上角绘制指北针。"""
    ax.add_patch(
        Rectangle(
            (0.865, 0.79), 0.11, 0.185,
            transform=ax.transAxes, facecolor="black", alpha=0.56, edgecolor="none", zorder=9,
        )
    )
    ax.annotate(
        "N",
        xy=(0.92, 0.945),
        xytext=(0.92, 0.815),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center", va="center",
        color="white", fontsize=8.4, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.3, mutation_scale=10),
        zorder=10,
    )


def main() -> None:
    cat = read_table(ROOT / "results/P3_02/GIS_output_catalog.csv")
    df_f1 = read_table(ROOT / "results/P5_02/figures/F1_source.csv")
    df_diff = read_table(ROOT / "results/P3_02/metrics/paired_differences.csv")

    fig = plt.figure(figsize=(15.4, 9.5))
    gs_outer = fig.add_gridspec(
        2, 1, height_ratios=[1.02, 1.12], hspace=0.25,
        top=0.905, bottom=0.068, left=0.055, right=0.985,
    )

    # =========================================================================
    # 上排 (a)-(d): 4 幅真实 PS-RGB + 像元错分 UTM 地图（按真实物理宽高比紧排）
    # =========================================================================
    gs_top = gs_outer[0].subgridspec(1, 4, width_ratios=[0.74, 0.74, 1.04, 1.04], wspace=0.10)
    scene_specs = [
        ("AOI_3_Paris_img27", "A", "EPSG:32631", "(a) 巴黎同城 A (img27)", "#2b6cb0", "欧式坡屋顶规整提取"),
        ("AOI_3_Paris_img27", "C", "EPSG:32631", "(b) 巴黎跨城 C (img27)", "#d7191c", "喀土穆→巴黎：近全漏检崩溃"),
        ("AOI_5_Khartoum_img1385", "D", "EPSG:32636", "(c) 喀土穆同城 D (img1385)", "#2b6cb0", "干旱区平顶群落稳健检出"),
        ("AOI_5_Khartoum_img1385", "B", "EPSG:32636", "(d) 喀土穆跨城 B (img1385)", "#dd6b20", "巴黎→喀土穆：高精低召回局部保留"),
    ]

    for idx, (img_id, route, dst_crs, title, border_col, subtitle) in enumerate(scene_specs):
        row = cat[(cat["image_id"] == img_id) & (cat["route"] == route)].iloc[0]
        disp, ov, ext, st = warp_scene_and_error_utm(
            Path(row["rgb_path"]), Path(row["support_path"]), ROOT / row["error_path"], dst_crs
        )
        ax = fig.add_subplot(gs_top[0, idx])
        ax.imshow(disp, extent=ext, origin="upper", interpolation="bilinear")
        ax.imshow(ov, extent=ext, origin="upper", interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(1.35)
        ax.set_title(title, loc="left", fontsize=9.1, fontweight="bold", pad=5)

        info_text = (
            f"{subtitle}\n"
            f"IoU={st['iou']:.1f}% | P={st['prec']:.1f}% | R={st['rec']:.1f}%"
        )
        box_fc = "#742a2a" if route == "C" else ("#7b341e" if route == "B" else "#111111")
        ax.text(
            0.03, 0.97, info_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=box_fc, alpha=0.78, edgecolor="none"),
            zorder=8,
        )
        add_scalebar(ax, length_m=50.0, label="50 m")
        if idx in (0, 2):
            add_north_arrow(ax)

    # =========================================================================
    # 下排左 (e): 双架构×3种子在四路线上的 4 项核心像素指标（散点 + min-max + 均值横线）
    # =========================================================================
    gs_bot = gs_outer[1].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.20)
    gs_e = gs_bot[0].subgridspec(2, 2, wspace=0.16, hspace=0.32)

    metric_specs = [
        ("IoU", "Micro IoU (%)", 0, 0),
        ("F1", "Micro F1 (%)", 0, 1),
        ("Precision", "Micro Precision (%)", 1, 0),
        ("Recall", "Micro Recall (%)", 1, 1),
    ]
    route_order = ["A", "C", "D", "B"]
    route_labels = ["A\n(巴黎同城)", "C\n(喀→巴跨城)", "D\n(喀土穆同城)", "B\n(巴→喀跨城)"]
    x_base = np.arange(len(route_order))

    for m_col, m_label, ri, ci in metric_specs:
        ax = fig.add_subplot(gs_e[ri, ci])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 区分巴黎目标区(A/C)与喀土穆目标区(D/B)
        ax.axvspan(-0.5, 1.5, facecolor="#ebf8ff", alpha=0.45, zorder=0)
        ax.axvspan(1.5, 3.5, facecolor="#fffaf0", alpha=0.50, zorder=0)
        ax.axvline(1.5, color="#a0aec0", ls="--", lw=0.85, zorder=1)

        for a_idx, arch in enumerate(["UNet", "DeepLabV3Plus"]):
            dx = -0.17 if arch == "UNet" else 0.17
            col = ARCH_COLORS[arch]
            mk = ARCH_MARKERS[arch]
            for r_idx, rt in enumerate(route_order):
                sub = df_f1[(df_f1["architecture"] == arch) & (df_f1["route"] == rt)].sort_values("seed")
                vals = sub[m_col].values * 100.0
                xp = r_idx + dx
                vmin, vmax, vmean = float(vals.min()), float(vals.max()), float(vals.mean())
                # 绘制真实 min-max 实线极差（零负下界越界）
                ax.plot([xp, xp], [vmin, vmax], color=col, lw=1.6, alpha=0.75, zorder=3)
                # 均值短横线
                ax.plot([xp - 0.09, xp + 0.09], [vmean, vmean], color="#111111", lw=1.8, zorder=5)
                # 3 个随机种子个体散点（轻微水平抖动防止重叠）
                jit = np.array([-0.032, 0.0, 0.032])
                for jv, sd, val in zip(jit, sub["seed"].values, vals):
                    is_outlier = (arch == "UNet" and rt == "B" and sd == 20261121)
                    ax.scatter(
                        [xp + jv], [val],
                        marker="*" if is_outlier else mk,
                        s=62 if is_outlier else 26,
                        color="#d7191c" if is_outlier else col,
                        edgecolor="white" if not is_outlier else "#111111",
                        lw=0.55,
                        zorder=6 if is_outlier else 4,
                    )

        # 在 IoU 子图添加简洁数字标示非对称性
        if m_col == "IoU":
            ax.text(1.0, 8.5, "C全模型\n0.0–0.2%", ha="center", va="bottom", fontsize=6.6,
                    color="#9b1c1c", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#fff5f5", edgecolor="#feb2b2", lw=0.6))
            ax.text(3.0, 33.5, "B跨种子\n5.5–22.5%", ha="center", va="bottom", fontsize=6.6,
                    color="#9c4221", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#fffaf0", edgecolor="#fbd38d", lw=0.6))
        elif m_col == "Recall":
            out_row = df_f1[
                (df_f1["architecture"] == "UNet") & (df_f1["route"] == "B") & (df_f1["seed"] == 20261121)
            ].iloc[0]
            out_r = float(out_row["Recall"]) * 100.0
            out_p = float(out_row["Precision"]) * 100.0
            ax.annotate(
                f"U-Net B/21 过估反例\n(Recall={out_r:.1f}%, P={out_p:.1f}%)",
                xy=(3 - 0.17, out_r),
                xytext=(1.0, 56.0),
                fontsize=6.2, color="#b91c1c", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=0.9, connectionstyle="arc3,rad=-0.10"),
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#fff5f5", edgecolor="#feb2b2", lw=0.6),
            )

        ax.set_xticks(x_base)
        if ri == 1:
            ax.set_xticklabels(route_labels, fontsize=7.4)
        else:
            ax.set_xticklabels(["A", "C", "D", "B"], fontsize=7.6)
        ax.set_ylim(-2.5, 86.0)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_ylabel(m_label, fontsize=8.0, fontweight="bold")
        ax.tick_params(labelsize=7.3)
        ax.grid(True, axis="y", alpha=0.22, ls="--")
        if ri == 0 and ci == 0:
            ax.set_title("(e) 双架构跨 3 随机种子四路线提取性能（散点 + min–max 极差 + 均值横线）",
                         loc="left", fontsize=9.1, fontweight="bold", pad=8)

    # =========================================================================
    # 下排右 (f): 同目标城市“同城 − 跨城”性能退化量 + P3-02 空间组 Bootstrap 95% CI
    # =========================================================================
    ax_f = fig.add_subplot(gs_bot[1])
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    m_keys = [
        ("micro_IoU", "IoU", "Δ Micro IoU"),
        ("micro_F1", "F1", "Δ Micro F1"),
        ("micro_Precision", "Precision", "Δ Precision"),
        ("micro_Recall", "Recall", "Δ Recall"),
    ]
    y_pos = np.arange(len(m_keys))[::-1]

    # 先画 P3-02 空间组 Bootstrap 95% CI 横向误差条（U-Net 基准，2000 次空间组重采样）
    for idx, (diff_m, f1_m, lbl) in enumerate(m_keys):
        yp = y_pos[idx]
        for c_off, city_code, r_in, r_cr, col_c, city_lbl in [
            (0.17, "AOI_3_Paris", "A", "C", "#1d4ed8", "巴黎 (A−C，22 空间组)"),
            (-0.17, "AOI_5_Khartoum", "D", "B", "#c05621", "喀土穆 (D−B，55 空间组)"),
        ]:
            dr = df_diff[(df_diff["target_city"] == city_code) & (df_diff["metric"] == diff_m)].iloc[0]
            pt = float(dr["within_minus_cross"]) * 100.0
            lo = float(dr["lower_95"]) * 100.0
            hi = float(dr["upper_95"]) * 100.0

            # 粗误差条表示 P3-02 空间组 Bootstrap 95% 区间
            ax_f.plot([lo, hi], [yp + c_off, yp + c_off], color=col_c, lw=3.0, alpha=0.45,
                      solid_capstyle="round", zorder=2)
            ax_f.scatter([pt], [yp + c_off], marker="D", s=46, color=col_c, edgecolor="white", lw=0.7, zorder=5)

            # 叠加 P5-02 两个架构×3种子的实际同目标退化值（小空心/实心点）
            for arch, mkr in [("UNet", "o"), ("DeepLabV3Plus", "s")]:
                s_in = df_f1[(df_f1["architecture"] == arch) & (df_f1["route"] == r_in)].set_index("seed")[f1_m]
                s_cr = df_f1[(df_f1["architecture"] == arch) & (df_f1["route"] == r_cr)].set_index("seed")[f1_m]
                gaps = (s_in - s_cr).values * 100.0
                y_sc = yp + c_off + (0.045 if arch == "UNet" else -0.045)
                ax_f.scatter(
                    gaps, np.full_like(gaps, y_sc),
                    marker=mkr, s=22,
                    facecolor="white" if arch == "DeepLabV3Plus" else col_c,
                    edgecolor=col_c, lw=0.95, alpha=0.92, zorder=4,
                )

            # 标注 P3-02 精确点估计与区间（全部严格避开散点）
            if diff_m == "micro_Precision" and city_code == "AOI_5_Khartoum":
                ax_f.text(1.2, yp + c_off - 0.16, f"+{pt:.1f} pp [+{lo:.1f}, +{hi:.1f}]",
                          va="top", ha="left", fontsize=6.6, fontweight="bold", color=col_c)
            elif diff_m == "micro_Precision" and city_code == "AOI_3_Paris":
                ax_f.text(pt - 2.0, yp + c_off + 0.13, f"+{pt:.1f} pp [+{lo:.1f}, +{hi:.1f}]",
                          va="bottom", ha="center", fontsize=6.6, fontweight="bold", color=col_c)
            else:
                ax_f.text(hi + 1.8, yp + c_off, f"+{pt:.1f} pp",
                          va="center", ha="left", fontsize=6.8, fontweight="bold", color=col_c)

    ax_f.axvline(0.0, color="#111111", lw=0.9, ls=":")
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels([x[2] for x in m_keys], fontsize=8.3, fontweight="bold")
    ax_f.set_xlim(-6.0, 92.0)
    ax_f.set_ylim(-0.62, 4.35)
    ax_f.set_xlabel("同目标城市性能退化量 Within − Cross (百分点, pp)", fontsize=8.2, fontweight="bold")
    ax_f.tick_params(labelsize=7.5)
    ax_f.grid(True, axis="x", alpha=0.24, ls="--")
    ax_f.set_title("(f) 同目标“同城−跨城”退化幅度与空间组 Bootstrap 95% 区间",
                   loc="left", fontsize=9.1, fontweight="bold", pad=8)

    # (f) 子图内顶部 2 列紧凑图例（置顶于纯白预留区 y=3.55~4.35，零遮挡全部数据条）
    f_handles = [
        Line2D([0], [0], color="#1d4ed8", lw=3.0, alpha=0.55, marker="D", markersize=5.2,
               markerfacecolor="#1d4ed8", markeredgecolor="white", label="巴黎 A−C（P3-02 点估计与 95% CI）"),
        Line2D([0], [0], color="#c05621", lw=3.0, alpha=0.55, marker="D", markersize=5.2,
               markerfacecolor="#c05621", markeredgecolor="white", label="喀土穆 D−B（P3-02 点估计与 95% CI）"),
        Line2D([0], [0], color="none", marker="o", markersize=4.8, markerfacecolor="#4a5568",
               markeredgecolor="#4a5568", label="P5-02 U-Net 3 种子退化散点"),
        Line2D([0], [0], color="none", marker="s", markersize=4.8, markerfacecolor="white",
               markeredgecolor="#4a5568", markeredgewidth=1.0, label="P5-02 DeepLabV3+ 3 种子退化散点"),
    ]
    ax_f.legend(handles=f_handles, loc="upper center", bbox_to_anchor=(0.50, 0.995),
                ncol=2, fontsize=6.4, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
                framealpha=0.96, borderpad=0.35, columnspacing=0.9)

    # =========================================================================
    # 顶栏分两组统一图例：【上排像元错分叠层】 + 【下排模型架构标记】
    # =========================================================================
    leg_px = [
        mpatches.Patch(facecolor=(0.11, 0.62, 0.47, 0.72), edgecolor="#222", lw=0.5, label="TP 正确检测建筑"),
        mpatches.Patch(facecolor=(0.91, 0.16, 0.54, 0.78), edgecolor="#222", lw=0.5, label="FP 假阳性误检"),
        mpatches.Patch(facecolor=(0.95, 0.75, 0.05, 0.80), edgecolor="#222", lw=0.5, label="FN 假阴性漏检"),
        mpatches.Patch(facecolor="#e2e8f0", edgecolor="#4a5568", lw=0.5, label="TN 背景（透明显示真彩底图）"),
    ]
    leg_arch = [
        Line2D([0], [0], color=ARCH_COLORS["UNet"], lw=1.6, marker="o", markersize=5.5, label="U-Net（3 随机种子散点与 min–max）"),
        Line2D([0], [0], color=ARCH_COLORS["DeepLabV3Plus"], lw=1.6, marker="s", markersize=5.5, label="DeepLabV3+（3 随机种子散点与 min–max）"),
        Line2D([0], [0], color="#111111", lw=1.8, label="3 种子均值"),
        Line2D([0], [0], color="none", marker="*", markersize=7.5, markerfacecolor="#d7191c",
               markeredgecolor="#111111", label="U-Net B/20261121 过估反例"),
    ]

    l1 = fig.legend(
        handles=leg_px, title="【上排 (a–d)：像元级误差叠层（P3-02 基准 U-Net）】",
        loc="upper left", bbox_to_anchor=(0.045, 0.992),
        ncol=4, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.2, title_fontsize=7.5, columnspacing=0.9, handlelength=1.4,
    )
    l1.get_title().set_fontweight("bold")

    l2 = fig.legend(
        handles=leg_arch, title="【下排 (e)：双架构跨 3 随机种子统计】",
        loc="upper right", bbox_to_anchor=(0.985, 0.992),
        ncol=2, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e0",
        fontsize=7.2, title_fontsize=7.5, columnspacing=0.9, handlelength=1.5,
    )
    l2.get_title().set_fontweight("bold")
    fig.add_artist(l1)

    fig.text(
        0.5, 0.012,
        "注：上排 (a–d) 展示了同一测试区域在同城与跨城推理下的原生 0.3 m 像元级误差栅格（UTM 重投影，P3-02 基准 U-Net，种子 20261115）；下排 (e) 对非负指标全部展示 3 个独立随机种子散点、真实 min–max 极差及均值横线（避免对称 ±SD 穿破 0% 下界）；(f) 置信区间由 P3-02 空间组（巴黎 22 组、喀土穆 55 组）2,000 次配对 Bootstrap 重采样给出。",
        ha="center", fontsize=7.5, color="#2d3748",
    )

    stem = "fig2_cross_city_asymmetric_degradation"
    for ext, dpi in [("png", 600), ("pdf", 300), ("svg", 300)]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved {OUT_DIR / stem}.(png[600dpi]|pdf|svg)")


if __name__ == "__main__":
    main()
