# SURF Feature Matching with OpenCV

This project demonstrates the use of the **Speeded-Up Robust Features (SURF)** algorithm for detecting and matching keypoints between an original image and a transformed version. It highlights SURF's robustness to **rotation**, **scaling**, **viewpoint changes**, and **illumination variation**.

---

## Description

The workflow involves:

1. Loading an original image (`original.jpg`)
2. Applying geometric and photometric transformations:
   - Rotation & Scaling
   - Perspective warp (simulated viewpoint change)
   - Illumination reduction
3. Detecting keypoints using **SURF**
4. Matching descriptors using **Brute-Force Matcher (L2 norm)**
5. Visualizing:
   - Original vs. Transformed images
   - Keypoints on each image
   - Top 50 matching features

---

## Dependencies

Make sure you have a custom-built OpenCV with `OPENCV_ENABLE_NONFREE=ON`.

### Install Requirements:
```bash
pip install numpy matplotlib Pillow
```

### Confirm OpenCV with SURF:
```python
import cv2
cv2.xfeatures2d.SURF_create()
```

If this raises an error, rebuild OpenCV with nonfree modules enabled.

---

## Usage

1. Place your base image as `original.jpg` in the project directory.
2. Run the script:
```bash
python3 surf.py
```
3. The script will:
   - Create a transformed image (`transformed.jpg`)
   - Display matched keypoints using matplotlib

---

## File Structure

```
├── surf.py         	 # Main script
├── original.jpg         # Input image (grayscale or color)
├── transformed.jpg      # Generated, transformed image
└── README.md            # This file
```

---

## Output

- Image 1: Original
- Image 2: Transformed (rotated, scaled, warped, darkened)
- Visuals:
  - Keypoints on both images
  - Top 50 feature matches with lines

---

## Notes

- `hessianThreshold` in `SURF_create()` controls keypoint sensitivity.
- SURF is patented and not available in the default OpenCV wheels. You must build OpenCV from source to access it.

---

## References

- [OpenCV SURF Docs](https://docs.opencv.org/master/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html)
- [Bay et al., 2006 — SURF: Speeded Up Robust Features](https://link.springer.com/article/10.1007/s11263-006-0039-z)

---

