PHM2010 timestamp segmentation audit

Segmentation inputs: experiment_tag and timestamp only.
The target tool_wear field is never read when constructing groups.
Reference threshold: timestamp increment < 0.1 x cutter-specific median positive cadence.
Expected protocol count: 315 cuts per cutter (c1, c4, c6).
Official challenge page: https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/
The wear field is used only after grouping to audit boundary correspondence.
Stable multiplier interval producing an identical grouping: 0.001 to 0.7.
All three cutters match 315 cuts: 1.
All wear changes occur at a timestamp boundary: 1.
