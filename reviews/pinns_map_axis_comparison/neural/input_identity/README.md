# PINNS motion and historical image-source identity

This check reconstructs the 7,283 development windows directly from the published 47 tracking text files. It reads columns 9 and 10 as metric ground X/Y and retains ordinal source frames 1, 4, 7, …, following the frozen 10 Hz convention. It uses the 22 calibration files for source identity. The script uses NumPy and the Python standard library.

From the repository root, run the following with your local PINNS metadata directory. The directory must contain `labels/` and `videos/<condition>/calib.json`. The optional private archive comes from the source-label image map builder.

```sh
python reviews/pinns_map_axis_comparison/neural/input_identity/verify.py "$METADATA_ROOT" \
  --maps "$PRIVATE_MAPS_NPZ" --output verified.json
```

Omit `--maps` to reproduce both motion digests and the source-eligibility counts from labels alone. Each digest includes the ordered clip, person, decision frame and split. The original digest covers the five-frame observed history and 30-step scoring future in coordinates relative to the final observation. The second covers the full 20-frame history and 30-step future in absolute source coordinates, including masks and padding.

The script reconstructs the original split and stride, then tests each window for a historical image source from the same clip. The source requires at least seven eligible training frames in its early reference period and must end before the prediction frame. The optional archive check compares every window key, split and source bit and counts map tensors with at least one valid image point. These checks use no RGB inference, neural weights or forecast scores. [Reference aggregate](verified.json)

## Original video decode

With OpenCV installed and the 47 original `videos/<condition>/videos/ori_<number>.mp4` files present, run the independent decode check from the repository root.

```sh
python reviews/pinns_map_axis_comparison/neural/input_identity/verify_original_decode.py \
  "$METADATA_ROOT" --output original_decode.json
```

The check reads the registered video bytes in full. It calls OpenCV `grab` once per declared raw frame and `retrieve` once for each distinct frame requested by the extractor’s person-label rule, across all splits. It compares decoded image dimensions with the container and calibration, then checks that one further `grab` fails at the declared end. All 26,296 frames and all 8,274 requested images passed with OpenCV 5.1.0 and FFmpeg. All 47 containers reported 1536 × 864 pixels at 30 Hz. The check exports only counts; decoder behavior on a different backend and person-to-image annotation alignment are separate questions. [Decode counts by site](original_decode.json)
