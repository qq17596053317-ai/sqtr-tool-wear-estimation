"""Regenerate Supplementary Figure S3 in vector and 600-dpi formats."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.97, 1.79))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.005, 0.955, "Randomised degradation pressure test", fontsize=10.5,
            weight="bold", va="top", color="#111111")

    centers = [0.10, 0.30, 0.50, 0.70, 0.90]
    widths = [0.18] * 5
    y0, height = 0.34, 0.34
    fills = ["#e8f3fa", "#ffffff", "#fff0e6", "#ffffff", "#e8f3fa"]
    headings = ["2 datasets", "3 channels", "4 modes", "Random severity", "30 repeats"]
    details = [
        "NUAA\nPHM 2010",
        "Fz\nvibration X / Y",
        "plateau clipping · dropout\nbias · noise",
        "severity and affected\nsegment sampled",
        "within each\ncondition",
    ]

    for x, width, fill, heading, detail in zip(centers, widths, fills, headings, details):
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y0), width, height,
            boxstyle="round,pad=0.006,rounding_size=0.022",
            linewidth=0.9, edgecolor="#aaaaaa", facecolor=fill,
        ))
        ax.text(x, y0 + 0.235, heading, ha="center", va="center",
                fontsize=7.8, weight="bold", color="#222222")
        ax.text(x, y0 + 0.105, detail, ha="center", va="center",
                fontsize=7.1, color="#222222", linespacing=1.1)

    for left, right in zip(centers[:-1], centers[1:]):
        ax.add_patch(FancyArrowPatch(
            (left + 0.093, y0 + height / 2),
            (right - 0.093, y0 + height / 2),
            arrowstyle="-|>", mutation_scale=9, linewidth=1.0, color="#777777",
        ))

    ax.text(0.5, 0.215, "Independent inference: 9 NUAA experiments + 3 PHM tools",
            ha="center", va="center", fontsize=7.7, weight="bold", color="#222222")
    ax.text(0.5, 0.105,
            "The 30 repeats quantify within-unit variability and are not independent inferential units.",
            ha="center", va="center", fontsize=6.6, color="#777777")

    stem = OUT / "Supplementary_Figure_S3_stress_test_design"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    svg_path = stem.with_suffix(".svg")
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
