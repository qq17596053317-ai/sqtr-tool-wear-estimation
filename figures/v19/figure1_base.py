from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as MplPath
from PIL import Image


OUTPUT_DIR = Path(__file__).resolve().parent
OUT_SVG = OUTPUT_DIR / "Figure1_sqtr_routing.svg"
OUT_PDF = OUTPUT_DIR / "Figure1_sqtr_routing.pdf"
OUT_PNG = OUTPUT_DIR / "Figure1_sqtr_routing.png"
OUT_TIFF = OUTPUT_DIR / "Figure1_sqtr_routing.tiff"
PREVIEW = OUTPUT_DIR / "Figure1_sqtr_routing_preview.png"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


BLUE = "#087DB8"
BLUE_EDGE = "#7EC6E6"
BLUE_FILL = "#E7F4FA"
ORANGE = "#D65A00"
ORANGE_EDGE = "#F3AF8C"
ORANGE_FILL = "#FFF3EC"
GRAY = "#8A8A8A"
GRAY_EDGE = "#D4DADD"
GRAY_FILL = "#F5F7F8"
DARK = "#252525"


def rounded_box(ax, x, y, w, h, face, edge, lw=1.15, radius=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=8, weight="normal", color=DARK,
          ha="center", va="center", style="normal", linespacing=1.25, zorder=5):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        fontstyle=style,
        linespacing=linespacing,
        zorder=zorder,
    )


def arrow(ax, start, end, color, lw=1.45, mutation=10, zorder=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def path_arrow(ax, vertices, color, lw=1.45, mutation=10, zorder=3):
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 1)
    patch = FancyArrowPatch(
        path=MplPath(vertices, codes),
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def stage_title(ax, number, cx, title_x, title):
    ax.add_patch(Circle((cx, 28), 12.0, facecolor=BLUE, edgecolor="none", zorder=4))
    label(ax, cx, 28, str(number), size=9.0, color="white")
    label(ax, title_x, 28, title, size=10.5, weight="bold", ha="left")


def build_figure():
    fig = plt.figure(figsize=(7.10, 3.22), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1080)
    ax.set_ylim(490, 0)
    ax.axis("off")

    # Stage titles
    stage_title(ax, 1, 22, 44, "Record representation")
    stage_title(ax, 2, 344, 366, "Channel-quality assessment")
    stage_title(ax, 3, 684, 706, "Conditional branch prediction")

    # Main boxes
    rounded_box(ax, 14, 104, 270, 100, GRAY_FILL, GRAY_EDGE)
    rounded_box(ax, 14, 254, 270, 112, ORANGE_FILL, ORANGE_EDGE)
    rounded_box(ax, 346, 104, 238, 100, GRAY_FILL, GRAY_EDGE)
    rounded_box(ax, 346, 264, 224, 94, ORANGE_FILL, ORANGE_EDGE)
    rounded_box(ax, 770, 105, 216, 102, BLUE_FILL, BLUE_EDGE)
    rounded_box(ax, 770, 253, 216, 113, ORANGE_FILL, ORANGE_EDGE)
    rounded_box(ax, 1003, 195, 71, 100, BLUE, BLUE)
    rounded_box(ax, 14, 410, 1060, 61, "#F2F6F8", GRAY_EDGE)

    # Prediction inputs
    label(ax, 149, 124, "Prediction inputs", size=8.5, weight="bold")
    label(ax, 149, 148, r"Process settings  $\mathbf{p}_i$", size=7.7)
    label(ax, 149, 171, r"Cumulative time  $t_i,\ t_i^2$", size=7.7)
    label(ax, 149, 194, r"Sensor features  $s_i^1,\ldots,s_i^m$", size=7.7)

    # Quality descriptors
    label(ax, 149, 275, "Label-free quality descriptors", size=8.3, weight="bold")
    label(ax, 149, 296, "(routing only)", size=7.4, style="italic")
    label(ax, 149, 325, "Rail occupancy · zero/flat ratios", size=7.4)
    label(ax, 149, 348, "Difference statistics", size=7.4)

    # Training reference and scores
    label(ax, 465, 126, "Training-health reference", size=8.3, weight="bold")
    label(ax, 465, 153, r"Median and $1.4826\times\mathrm{MAD}$", size=7.6)
    label(ax, 465, 178, "computed per channel", size=7.4)

    label(ax, 458, 283, "Channel deviation scores", size=8.2, weight="bold")
    label(ax, 458, 314, r"$d_{ij}=\max\,|\mathrm{robust}\ z\mathrm{-score}|$", size=7.6)
    label(ax, 458, 340, r"$d_i^{\max}=\max_j d_{ij}$", size=7.4)

    # Decision diamond
    diamond = Polygon(
        [[662, 249], [724, 311], [662, 373], [600, 311]],
        closed=True,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.35,
        zorder=2,
    )
    ax.add_patch(diamond)
    label(ax, 662, 282, "Abnormal", size=7.2)
    label(ax, 662, 302, "channel?", size=7.2)
    label(ax, 662, 329, r"$d_i^{\max}>\tau$", size=7.5)

    # Candidate branches
    label(ax, 878, 134, "All-sensor branch", size=8.4, weight="bold")
    label(ax, 878, 163, "Ridge model using", size=7.4)
    label(ax, 878, 183, "all channels", size=7.4)

    label(ax, 878, 275, "Channel-exclusion branch", size=8.1, weight="bold")
    label(ax, 878, 306, r"Excludes smcDC (NASA)", size=7.2)
    label(ax, 878, 329, r"or detected channel $j^\ast$", size=7.2)
    label(ax, 878, 350, "(external datasets)", size=7.0)

    # Output
    label(ax, 1038.5, 219, "One wear", size=7.4, weight="bold", color="white")
    label(ax, 1038.5, 241, "estimate", size=7.4, weight="bold", color="white")
    label(ax, 1038.5, 272, r"$\hat{y}_i$", size=10.0, weight="bold", color="white")

    # Footer
    label(
        ax,
        544,
        440.5,
        "All preprocessing, thresholds, and models are fitted within the training fold.",
        size=9.3,
        weight="bold",
    )

    # Information-flow arrows
    arrow(ax, (284, 154), (346, 154), GRAY)
    arrow(ax, (284, 310), (346, 310), ORANGE)
    arrow(ax, (465, 204), (465, 264), ORANGE)
    arrow(ax, (570, 311), (600, 311), ORANGE)

    # Prediction inputs feed both pre-fitted candidate branches.
    path_arrow(ax, [(284, 154), (315, 154), (315, 78), (748, 78), (748, 156), (770, 156)], GRAY)
    path_arrow(ax, [(748, 156), (748, 229), (800, 229), (800, 253)], GRAY)

    # Router output.
    path_arrow(ax, [(662, 249), (696, 216), (733, 183), (770, 169)], BLUE)
    label(ax, 703, 222, "No", size=7.2, color=BLUE)
    arrow(ax, (724, 311), (770, 311), ORANGE)
    label(ax, 746, 296, "Yes", size=7.2, color=ORANGE)

    # Candidate predictions converge to a single estimate.
    path_arrow(ax, [(986, 156), (998, 170), (1010, 195)], BLUE)
    path_arrow(ax, [(986, 310), (998, 304), (1014, 294)], ORANGE)

    return fig


def save_outputs():
    fig = build_figure()
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SVG, format="svg", facecolor="white")
    fig.savefig(OUT_PDF, format="pdf", facecolor="white")
    fig.savefig(OUT_PNG, format="png", dpi=600, facecolor="white")
    fig.savefig(PREVIEW, format="png", dpi=300, facecolor="white")
    plt.close(fig)

    with Image.open(OUT_PNG) as image:
        image.save(
            OUT_TIFF,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(600, 600),
        )

    print(OUT_SVG)
    print(OUT_PDF)
    print(OUT_PNG)
    print(OUT_TIFF)
    print(PREVIEW)


if __name__ == "__main__":
    save_outputs()


