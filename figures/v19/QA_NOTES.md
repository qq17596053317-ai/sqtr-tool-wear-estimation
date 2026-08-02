# Figure 1 v19 QA notes

- Core conclusion: SQTR selects one pre-fitted regression branch from wear-label-free routing evidence.
- Evidence mapping: process variables, sensor features and time feed the candidate branches; dedicated quality descriptors and selected channel statistics feed the quality detector; the hard gate returns one branch output.
- Figure archetype: schematic-led composite.
- Backend: Python/matplotlib only.
- Final size: 180.3 mm wide; raster exports are 600 dpi; SVG and PDF preserve editable text.
- Data integrity: Figure 1 is a method schematic and contains no experimental observations, sampling or exclusions.
- Method-consistency change: the routing-evidence box now distinguishes dedicated quality descriptors from selected channel statistics. This matches the external-dataset implementation, in which seven signal statistics are reused by the detector while the four dedicated descriptors remain outside the regression covariates.
- Visual QA: the PNG/TIFF, SVG and PDF were inspected at final size. Text remains within its boxes, arrows do not overlap, and the route labels remain legible.
- Static preflight: `validate_figure.py --strict` returned 14 PASS, 0 WARN and 0 FAIL.
