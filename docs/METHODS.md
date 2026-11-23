# Frozen methods and engineering adaptations

The numerical routines are extracted from the executed FIX03, P4 and P5 sources.
`PROVENANCE.json` records their source names, original hashes and extracted symbols.
The original research workspace is unchanged. Historical per-stage authorization
and seal-verification wrappers are not public execution dependencies; the new CLI
uses the frozen effective manifest and preserves source/test separation.

Support class 1 alone is observed. Boundary-connected all-zero RGB/PAN/pansharpened
MUL components require ≥64 pixels, eroded interior agreement ≥0.99 with native MUL
zero evidence, and a one-pixel guard. Uncertain/invalid regions are excluded.
The actual rules are retained in `src/p1_spatial.py`. Quality screening, buffers,
development exposure and the single FIX03 exclusion are retained row-wise in the
manifest. No model scores enter a new split search.

Both architectures use 128×128 input, random initialization, no augmentation,
Adam (lr .001, betas .9/.999, eps 1e-8, weight decay 0), batch 4 and 12 epochs.
Source-training P1/P99 is estimated lexicographically using every 16th valid native
pixel; it is never re-estimated on target data. RGB uses bilinear resize with
align_corners=False, labels nearest and Support adaptive pooling ≥.999999.
Loss is masked probability BCE plus batch-global soft Dice with smoothing 1.
Checkpoint selection uses the original group-balanced positive IoU / empty-scene
false-positive criterion; score rounding is 12 decimals and ties select the earliest
epoch. Sigmoid probabilities are upsampled bilinearly to native size before the .5
threshold. Original parameter YAML files remain as reference snapshots; the CLI
uses the frozen values rather than historical private output paths.

Native confusion areas use WGS84 ellipsoidal row areas. GIS uses native class
polygonization, edge densification at 1e-5 degree and EPSG:32631 / EPSG:32636.
Overlaps are assigned to the lexicographically first image ID for both models and
reference labels. Grid indices are anchored at absolute UTM multiples of 100, 250
or 500 m. Half-cell shifts are east, north or both, with boundary neighbors added
to preserve support. Only observed cell intersections enter the denominator.

For observed area S, building FP/FN areas and grid side L:

```text
P − A = FP − FN
E_net = |FP − FN| / S
E_spatial = (FP + FN) / S
Q = S / L²
C = 1 − |FP − FN| / (FP + FN), undefined when FP + FN = 0
E_spatial − E_net = 2 min(FP,FN) / S
```

Scenarios: net tolerances .01/.02/.05; spatial tolerances .02/.05/.10; Q minima
.20/.50/.80. Pass boundaries are inclusive. The main scenario is .02/.05/.50.
These are descriptive, post-hoc GT-dependent tolerances, not a calibrated
unlabelled-deployment classifier. Pixels, grids, seeds and cities are not treated
as independent replicates for a new significance test.

Engineering adaptations replace absolute paths, output directories and historical
approval wrappers. The public training loop retains the numerical order and
checkpoint selection but uses a new portable checkpoint/output schema. Exact
equivalence against full historical outputs remains to be checked; see BUILD_STATUS.
