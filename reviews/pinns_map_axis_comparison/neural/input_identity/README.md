# PINNS motion and historical image-source identity

This check reconstructs the 7,283 development windows directly from the published 47 tracking text files. It reads columns 9 and 10 as metric ground X/Y and retains ordinal source frames 1, 4, 7, …, following the frozen 10 Hz convention. It uses the 22 calibration files for source identity. The script uses NumPy and the Python standard library.

From the repository root, run the following with your local PINNS metadata directory. The directory must contain `labels/` and `videos/<condition>/calib.json`. The optional private archive comes from the source-label image map builder.

```sh
python reviews/pinns_map_axis_comparison/neural/input_identity/verify.py "$METADATA_ROOT" \
  --maps "$PRIVATE_MAPS_NPZ" --output verified.json
```

Omit `--maps` to reproduce both motion digests and the source-eligibility counts from labels alone. Each digest includes the ordered clip, person, decision frame and split. The original digest covers the five-frame observed history and 30-step scoring future in coordinates relative to the final observation. The second covers the full 20-frame history and 30-step future in absolute source coordinates, including masks and padding.

The script reconstructs the original split and stride, then tests each window for a historical image source from the same clip. The source requires at least seven eligible training frames in its early reference period and must end before the prediction frame. The optional archive check compares every window key, split and source bit and counts map tensors with at least one valid image point. These checks use no RGB inference, neural weights or forecast scores. [Reference aggregate](verified.json)
