# Local release candidate acceptance

Date: 2026-11-23. The source project has not been deleted, moved or rewritten.
The candidate is under `GitHub_Release/`. Private email-bearing DOCX/PDF/Markdown
review copies are in the separate `GitHub_Author_Review/` folder and must not be
published. Public main and supplement PDFs contain the sole author Mengle Xue,
retain 测绘地理信息学院, omit the university and private email, and leave ORCID,
city and postal code unfilled. The main PDF has 11 pages and the supplement 8.
Word-generated PDFs were rendered with the bundled PDF runtime after the generic
DOCX renderer reported missing LibreOffice. Page overviews and changed author
pages were inspected. The original manuscript and formal figure bytes are preserved.

## Observed checks

- A fresh `.venv` installed official CPU PyTorch 2.7.1 and all declared dependencies.
  `pip check` passed. The exact installed versions are in the Windows/Python 3.11 lock.
- Nine targeted tests passed inside that environment with PYTHONPATH
  cleared. The two-model synthetic disk-to-model-to-native-inference-to-GIS smoke
  passed there too, including checkpoint reloading, overlap ownership and all
  3 scale × 4 origin support-conservation checks.
- All 2,160 actual data scenes passed frozen RGB/label asset hashes and regenerated
  Support array hashes. The four recorded derived label repairs are included with
  data-license attribution. Preparation did not train a model or change source data.
- Five formal main plots and three supplementary plots ran against the packaged
  recorded source statistics, selected derived GIS layers and external imagery.
- The new GIS compatibility exporter recomputed all 288 recorded cell combinations
  and matched all 1,440 main-scenario rows and their numeric columns (rtol 1e-9,
  atol 1e-7). This is a code/schema check using existing results, not new training.
- Initial candidate scanning found no personal absolute paths, either known private
  email, common secret-token patterns, broken local Markdown links or ≥100 MiB files.
  The final public audit and machine-readable checks are in `docs/validation/`.

## Scope and limits

The complete public-data reproduction commands and new-output-to-main-figure
conversion code are supplied. Their components and schemas have been exercised;
the 12 formal models have not been freshly retrained and compared end to end.
Do not claim all formal results have been independently reproduced, peer reviewed,
or externally validated. The archived timing measurements remain specific to the
original machine and cannot be inferred from a different machine's execution.

No raw SpaceNet imagery, private weights, Skills source, credentials or the private
review folder are Git candidates. The author has selected MIT for original project
code; the LICENSE scope notice excludes research documents, remote-sensing imagery,
derived third-party media and third-party code. The upstream U-Net MIT license and
SpaceNet CC BY-SA terms retain their own scope.

The public repository was created under the authenticated GitHub profile xy0125423.
For the final release, verify that the remote default-branch HEAD exactly matches the local HEAD.

Engineering intermediates remain outside the public candidate. An early plotting
probe created a new workspace-level `figures/regenerated/` folder before its absolute
path was corrected; it did not replace pre-existing research artifacts. No cleanup
of the original workspace was performed.


The full main-figure converter was additionally exercised on an archive-output
fixture arranged in the portable run schema. After fixing missing diagnostic and
critical-scene catalog fields, all five main plots ran in the clean environment
using that exported directory. This verifies exporter IO and plot-schema integration;
the fixture contains recorded experiment outputs, not newly trained models.
