from __future__ import annotations

"""Generate the v19 Figure 1 with method-consistent routing evidence.

The production drawing is inherited from the audited Python Figure 1 source.
Only the routing-evidence wording and output locations are changed here.
"""

import importlib.util
from pathlib import Path

import matplotlib as mpl


WIDTH_MM = 180.3
RASTER_DPI = 600
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


SOURCE = Path(__file__).resolve().parent / "v19" / "figure1_base.py"
OUTPUT = Path(__file__).resolve().parent / "v19"


def load_source():
    spec = importlib.util.spec_from_file_location("sqtr_figure1_v18_source", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Figure 1 source: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_source()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    module.OUT_SVG = OUTPUT / "Figure1_sqtr_routing_v19.svg"
    module.OUT_PDF = OUTPUT / "Figure1_sqtr_routing_v19.pdf"
    module.OUT_PNG = OUTPUT / "Figure1_sqtr_routing_v19.png"
    module.OUT_TIFF = OUTPUT / "Figure1_sqtr_routing_v19.tiff"
    module.PREVIEW = OUTPUT / "Figure1_sqtr_routing_v19_preview.png"

    original_label = module.label

    def consistent_label(ax, x, y, text, *args, **kwargs):
        if text == "Label-free quality descriptors":
            text = "Routing evidence"
        elif text == "(routing only)":
            text = "Dedicated quality descriptors\nand selected channel statistics"
            y = 313
            kwargs["style"] = "normal"
            kwargs["size"] = 7.3
            kwargs["linespacing"] = 1.22
        elif text in {
            "Rail occupancy · zero/flat ratios",
            "Difference statistics",
        }:
            return None
        return original_label(ax, x, y, text, *args, **kwargs)

    module.label = consistent_label
    module.save_outputs()


if __name__ == "__main__":
    main()
