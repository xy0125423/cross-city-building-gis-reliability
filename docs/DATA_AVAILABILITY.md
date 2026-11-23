# Data availability and input layout

Official source: https://spacenet.ai/spacenet-buildings-dataset-v2/
The publisher lists Paris and Khartoum building-training tarballs under the AWS
SpaceNet public dataset. Follow the publisher's current AWS access instructions;
no raw-data download was performed for this release build.

Required products for both `AOI_3_Paris` and `AOI_5_Khartoum`:

- `RGB-PanSharpen/RGB-PanSharpen_<image_id>.tif`
- `PAN/PAN_<image_id>.tif`
- `MUL-PanSharpen/MUL-PanSharpen_<image_id>.tif`
- `MUL/MUL_<image_id>.tif` (native-resolution evidence for Support)
- `geojson/buildings/buildings_<image_id>.geojson`

An extracted public tarball tree is accepted when each original filename is unique
under `--data-root`. The historical relative tree is also accepted:

```text
DATA_ROOT/
  interim/spacenet2/AOI_3_Paris/AOI_3_Paris_Train/{products...}
  interim/spacenet2/AOI_5_Khartoum/AOI_5_Khartoum_Train/{products...}
```

Data-root resolution does not require a fixed drive or OS. Duplicate basenames are
rejected rather than silently choosing an asset. Four small derived label-repair
GeoJSONs are in `configs/label_repairs/`; all original labels remain external.
Repairs use GEOS make_valid(linework), polygon extraction and union. These are
recorded preprocessing outputs, not author-drawn new building annotations.

The publisher states CC BY-SA 4.0. Attribute SpaceNet Partners and consult the
publisher's terms before redistributing imagery or adapted material. Original
imagery/labels are not supplied in this repository; figure panels and the four
derived repairs retain the corresponding data-license obligations. Public
manifests contain relative paths, frozen membership and asset hashes.
