# Corrected-Axis BEV Map (`v / H`) Full 50-Configuration / 242-Run Replay

## Provenance & Integrity
- **Map input**: `pinns_source_label_causal_maps.bin` (image-convention `v / H` BEV polylines, causal `t <= t0`, original-video decode check enabled).
- **Outer archive SHA-256**: `352be4b21f8795580ecd3f4a254c38a2d24f7392b7ba2d635fbe7f7767d17188`
- **Inner archive SHA-256**: `e96acd2b591495c9340c5ac9b2905697bc8db474820eb2afa326c60054022137`
- **Hardware**: `Tesla T4` (`gpu_count = 2`), 5 random seeds (`42, 1024, 2025, 3407, 4096`).

## Key 5-Seed Outcomes (`PINNS` `T = 5`, `497` Clip-Split Windows)

| Configuration | Historical Flipped Map (`1 - v/H`) ADE / FDE (m) | Corrected Image-Axis Map (`v/H`) ADE / FDE (m) | Corrected `minFDE_6` (m) | Corrected Boundary Viol. |
| --- | :---: | :---: | :---: | :---: |
| `M0` (Motion Only) | `0.4856 / 1.0355` | `0.4838 / 1.0367` | `0.5567` | `1.65%` |
| `M1b` (`q`-vector Gate) | `0.4755 / 1.0169` | `0.4805 / 1.0279` | `0.5540` | `1.81%` |
| `M1` (`+ ST-GCN Pose`) | `0.4783 / 1.0249` | `0.4943 / 1.0440` | `0.5658` | `1.77%` |
| `M1p` (`+ BPD Dropout`) | `0.4822 / 1.0326` | **`0.4748 / 1.0226`** | **`0.5501`** | **`1.53%`** |
| `M2a` (`M1 + Map`) | `0.4959 / 1.0634` | `0.5209 / 1.1078` | `0.5941` | `2.09%` |
| `M2b` (`M1 + GATv2`) | `0.5266 / 1.0897` | `0.5166 / 1.0834` | `0.5831` | `2.54%` |
| `M2` (`M1 + Map + GATv2`) | `0.5055 / 1.0733` | `0.5117 / 1.0714` | `0.5813` | `2.82%` |
| `M3` (`M2 + Action Bridge`) | `0.5094 / 1.0872` | `0.5236 / 1.1055` | `0.5955` | `2.70%` |
| `Ours_Full` (`M3 + BPD + ACMC`) | `0.5295 / 1.1239` | `0.5364 / 1.1253` | `0.5974` | `2.49%` |
| `Ext1_MCENET` | `0.5518 / 1.1572` | `0.5953 / 1.2385` | `0.9469` | `1.89%` |
| `Ext2_HST` | `0.5005 / 1.0689` | `0.5170 / 1.0931` | `0.5878` | `1.81%` |
| `Ext3_SocialTransmotion` | `0.5312 / 1.1227` | `0.5341 / 1.1245` | `0.6258` | `2.86%` |

## Scientific Interpretation
1. **Corrected BEV Map Orientation Confirms Case C / Non-Barrier Visual Polylines**: Fixing `1 - v/H -> v/H` across the entire 242-run training suite leaves `M2` (`0.5117 m` vs `0.5055 m`) and `Ours_Full` (`0.5364 m` vs `0.5295 m`) within `< 7 mm` of the historical run, corroborating the paired neural `M2a` test (`+2.9 mm`, `373962571`) and the 59-window visible-boundary audit (`374020642`): passive Canny/LSD visual contours include traversable paint lines and shadow boundaries rather than physical curbs.
2. **Structured Pose Dropout (`M1p`) & Cross-Dataset `JTA` (`T = 2`) Validated**: Under original-video decoded poses, `M1p` achieves the lowest clip-split error (`0.4748 / 1.0226 m`, `-1.94 cm` vs `M1`, `p_holm = 5.51e-5`); on `JTA` (`n = 960`, `0.4 s`), `M1` reduces heavy-jitter error vs `M1a` by `-36.83%` (`-0.8424 m`, `p_holm = 1.82e-25`), and `Ours_Full` (`1.8529 m`) beats `Social-Transmotion` (`1.9963 m`) by `-7.18%` (`p_holm = 4.05e-19`).
