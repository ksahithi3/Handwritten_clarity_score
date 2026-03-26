"""
HANDWRITING DETECTION + CLARITY PIPELINE
=========================================
Detects handwritten vs typed regions in documents,
draws bounding boxes, measures handwriting clarity.

Saves debug images after each step so you can inspect
exactly what the pipeline sees.

INSTALL:
    pip install opencv-python numpy pillow pymupdf

USAGE:
    python handwriting_pipeline.py

OUTPUT FOLDER: debug_output/
    step1_gray.jpg          — grayscale conversion
    step2_threshold.jpg     — adaptive threshold (ink isolated)
    step3_regions.jpg       — all detected text regions
    step4_classified.jpg    — GREEN=typed, RED=handwritten, ORANGE=mixed
    step5_hw_only.jpg       — handwritten regions only highlighted
    step6_clarity_map.jpg   — clarity score overlaid on each hw region
    final_report.txt        — scores for every region
"""

import cv2
import numpy as np
from PIL import Image
import fitz
import os

# ── CONFIG ────────────────────────────────────────────
# Put the file paths you want to test here
TEST_FILES = [
    r"C:\Users\Admin\Desktop\handwritten-clarity\Panchnama_2025-06-27 13_23_51.0_318208094.jpeg",
    r"C:\Users\Admin\Desktop\handwritten-clarity\HOID-308033-Charge sheet.pdf",
    r"C:\Users\Admin\Desktop\handwritten-clarity\SEIZURE_MOMO_OF_IV (76).pdf",
]

OUTPUT_DIR = r"C:\Users\Admin\Desktop\handwritten-clarity\debug_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── THRESHOLDS (tune these if detection is off) ────────
ENTROPY_HW_THRESHOLD    = 4.0   # above = handwritten (lowered)
ENTROPY_TYPED_THRESHOLD = 3.2   # below = typed
VARIANCE_HW_THRESHOLD   = 1.2   # above = handwritten (lowered)
VARIANCE_TYPED_THRESHOLD= 0.6   # below = typed
MIN_REGION_AREA         = 400   # lowered — catches smaller handwritten words
MAX_REGION_AREA_RATIO   = 0.85  # ignore regions >85% of page (page border)


# ════════════════════════════════════════════════════════
# STEP 0 — LOAD DOCUMENT AS IMAGE
# ════════════════════════════════════════════════════════
def load_document(file_path, page_num=0):
    """Load PDF or image file as OpenCV BGR image."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        doc  = fitz.open(file_path)
        page = doc[page_num]
        mat  = fitz.Matrix(2.0, 2.0)   # 2x zoom = ~144 DPI
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read: {file_path}")
        return img


# ════════════════════════════════════════════════════════
# STEP 1 — GRAYSCALE
# ════════════════════════════════════════════════════════
def step1_grayscale(img, prefix):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step1_gray.jpg", gray)
    print(f"  [Step 1] Grayscale — shape: {gray.shape}")
    return gray


# ════════════════════════════════════════════════════════
# STEP 2 — ADAPTIVE THRESHOLD
# Isolates ink from paper, handles shadows + uneven lighting
# blockSize=21 works well for phone captures
# ════════════════════════════════════════════════════════
def step2_threshold(gray, prefix):
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=8
    )
    # Light morphological close to connect broken strokes
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step2_threshold.jpg", binary)
    print(f"  [Step 2] Adaptive threshold — ink pixels: "
          f"{np.sum(binary > 0):,}")
    return binary


# ════════════════════════════════════════════════════════
# STEP 2.5 — GLOBAL SKEW DETECTION
#
# WHY THIS IS NEEDED:
#   _edge_angle_entropy looks for angle clusters to decide
#   if a region is typed (clusters at 0/90) or handwritten
#   (random angles). But if the page is tilted 8°, typed text
#   has clusters at 8/98 — not 0/90. The entropy function sees
#   no cluster at 0/90, gets confused, and calls typed text
#   "handwritten."
#
# THE FIX — Professional way (no pixel rotation):
#   1. Detect the global skew angle of the whole page
#   2. Pass it to _edge_angle_entropy as an offset
#   3. Subtract the skew from all measured angles before
#      computing the histogram — so 8° tilted typed text
#      looks like 0° typed text to the entropy calculation
#   4. Original pixels are never touched — no interpolation blur
#
# METHOD: Hough line transform on the cleaned binary.
#   Find all long horizontal-ish lines, measure their angles,
#   take the median. This is the global skew.
# ════════════════════════════════════════════════════════
def step25_detect_skew(binary_clean, prefix):
    """
    Returns global skew angle in degrees.
    Positive = tilted clockwise, Negative = counter-clockwise.
    Returns 0.0 if skew cannot be reliably detected.
    """
    # Use Hough lines on the cleaned binary
    # HoughLinesP finds line segments — we want long ones (likely text baselines)
    h, w    = binary_clean.shape
    min_len = w // 6   # line must be at least 1/6 page wide to count

    lines = cv2.HoughLinesP(
        binary_clean,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=20
    )

    if lines is None or len(lines) < 3:
        print(f"  [Step 2.5] Skew: could not detect (too few lines) → using 0.0°")
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue   # vertical line — skip
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only use near-horizontal lines (within 20° of horizontal)
        # These are text baselines, not vertical strokes
        if abs(angle) < 20:
            angles.append(angle)

    if len(angles) < 3:
        print(f"  [Step 2.5] Skew: not enough horizontal lines → using 0.0°")
        return 0.0

    skew = float(np.median(angles))
    print(f"  [Step 2.5] Global skew detected: {round(skew, 2)}° "
          f"(from {len(angles)} lines)")
    return round(skew, 2)


# ════════════════════════════════════════════════════════
# STEP 3 — FIND TEXT REGIONS
#
# NEW APPROACH: Line-by-line sliding window
#
# The old dilation approach failed because:
#   1. Entirely handwritten docs (Panchnama) have no separate zones
#   2. Ruling lines made the whole page one giant contour
#   3. That giant contour exceeded MAX_REGION_AREA_RATIO → discarded
#
# New approach:
#   A) Remove horizontal ruling lines morphologically
#   B) Divide the page into horizontal strips (one per text line)
#      using the vertical projection profile — ink density per row
#   C) Find valleys (low ink rows = gaps between lines)
#      and peaks (high ink rows = text lines)
#   D) Each text line becomes a region for classification
#   E) Also run a block-level pass to catch large mixed zones
# ════════════════════════════════════════════════════════
def step3_find_regions(binary, img, prefix):
    h, w      = binary.shape
    page_area = h * w

    # ── A: Remove horizontal ruling lines ─────────────────
    # Pass 1: wide kernel catches full-width solid lines
    kernel_wide   = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    lines_wide    = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_wide)
    # Pass 2: medium kernel catches dotted/dashed ruling lines
    # Dotted lines have gaps, so we dilate first to connect dots,
    # then open to isolate the line, then subtract
    kernel_med    = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 6, 1))
    binary_temp   = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1)))
    lines_dotted  = cv2.morphologyEx(binary_temp, cv2.MORPH_OPEN, kernel_med)
    # Combine both line masks and subtract from binary
    lines_all     = cv2.add(lines_wide, lines_dotted)
    binary_clean  = cv2.subtract(binary, lines_all)
    # Dilate slightly to restore any ink broken by line removal
    kernel_restore = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_clean   = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_restore)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step3a_no_lines.jpg", binary_clean)

    # ── B: Vertical projection profile ────────────────────
    # Sum ink pixels per row → tells us where text lines are
    row_profile = np.sum(binary_clean > 0, axis=1).astype(np.float32)

    # Smooth the profile to avoid noise spikes
    kernel_1d   = np.ones(5) / 5
    row_profile = np.convolve(row_profile, kernel_1d, mode='same')

    # Save profile visualization
    profile_vis = np.ones((h, 300), dtype=np.uint8) * 50  # dark grey background
    max_val     = row_profile.max() + 1e-9
    for row_i, val in enumerate(row_profile):
        bar_w = int(val / max_val * 280)
        if bar_w > 0:
            cv2.line(profile_vis, (0, row_i), (bar_w, row_i), 255, 1)
        # Mark threshold line in red (approximate)
        thresh_x = int((w * 0.04) / max_val * 280)
        cv2.line(profile_vis, (thresh_x, row_i), (thresh_x, row_i), 120, 1)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step3b_profile.jpg", profile_vis)

    # ── C: Find text line boundaries ──────────────────────
    # Threshold: rows with ink > 1% of page width = text row
    ink_threshold = w * 0.04   # raised: needs 4% of row width to count as text
    in_text       = False
    line_start    = 0
    line_regions  = []

    for row_i in range(h):
        has_ink = row_profile[row_i] > ink_threshold
        if has_ink and not in_text:
            in_text    = True
            line_start = row_i
        elif not has_ink and in_text:
            in_text = False
            line_end = row_i
            # Only keep if tall enough to be real text (not noise)
            if (line_end - line_start) > 15:  # min 15px tall to be a real text line
                # Find actual left/right bounds of ink in this strip
                strip = binary_clean[line_start:line_end, :]
                cols  = np.where(np.sum(strip > 0, axis=0) > 0)[0]
                if len(cols) > 10:
                    x_start = max(0, cols[0] - 5)
                    x_end   = min(w, cols[-1] + 5)
                    line_regions.append((x_start, line_start,
                                         x_end - x_start,
                                         line_end - line_start))

    # ── D: Also run block-level pass ──────────────────────
    # Merges adjacent line regions into paragraph blocks
    # Useful for mixed docs where HW paragraphs need a single box
    kernel_block = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20))
    dilated_b    = cv2.dilate(binary_clean, kernel_block, iterations=1)
    contours_b, _ = cv2.findContours(
        dilated_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    block_regions = []
    for cnt in contours_b:
        x, y, rw, rh = cv2.boundingRect(cnt)
        area = rw * rh
        if area < MIN_REGION_AREA:
            continue
        # Only discard if it's basically the entire page (border artifact)
        if rw > w * 0.98 and rh > h * 0.98:
            continue
        block_regions.append((x, y, rw, rh))

    # ── E: Combine line + block regions ───────────────────
    all_regions = line_regions + block_regions

    # Filter out any remaining too-small regions
    regions = []
    for (x, y, rw, rh) in all_regions:
        if rw * rh < MIN_REGION_AREA:
            continue
        # Ignore scanner watermarks — typically in bottom 4% of page
        # and spanning most of the width (like "Scanned with OKEN Scanner")
        if y > h * 0.94 and rw > w * 0.3:
            continue
        regions.append((x, y, rw, rh))

    # Draw all regions in blue on a copy
    debug = img.copy()
    for (x, y, rw, rh) in regions:
        cv2.rectangle(debug, (x, y), (x+rw, y+rh), (255, 100, 0), 1)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step3_regions.jpg", debug)
    print(f"  [Step 3] Found {len(regions)} regions "
          f"({len(line_regions)} line-level, {len(block_regions)} block-level)")
    return regions
# STEP 4 — CLASSIFY EACH REGION
#
# Signal A: Edge Angle Entropy
#   Sobel gives edge direction at every pixel.
#   Entropy of the angle histogram measures how "random" they are.
#   Printed text: low entropy (edges cluster at 0°/90°)
#   Handwriting : high entropy (edges spread in all directions)
#
# Signal B: Stroke Width Variance
#   Distance transform on binary gives stroke width at each ink pixel.
#   Printed text: uniform stroke width → low variance
#   Handwriting : variable pressure/width → high variance
#
# Both signals vote → final label: typed / handwritten / mixed
# ════════════════════════════════════════════════════════
def _edge_angle_entropy(region_gray, skew_offset=0.0):
    """
    Measure randomness of edge angles. Higher = more handwritten.

    skew_offset: global page skew in degrees (from step25_detect_skew).
    Angles are shifted by -skew_offset before histogram so that
    typed text on a tilted page still clusters near 0/90,
    not at skew/90+skew.
    """
    sx          = cv2.Sobel(region_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy          = cv2.Sobel(region_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag         = np.sqrt(sx**2 + sy**2)
    angles_raw  = np.arctan2(sy, sx) * 180 / np.pi

    # ── Skew normalization ──────────────────────────────
    # Subtract global skew so typed clusters realign to 0/90
    # e.g. if page is tilted 8°, a typed horizontal stroke is
    # at 8° → after subtracting 8° it becomes 0° (correct)
    angles_norm = angles_raw - skew_offset
    # Wrap back into [-180, 180] range
    angles_norm = ((angles_norm + 180) % 360) - 180

    # ── Exclude horizontal edges (ruling lines) ──────────
    # After skew normalization, ruling lines cluster tightly at 0°
    # Exclude ±10° around 0° and 180° to remove their influence
    horiz_mask  = (np.abs(angles_norm) < 10) | (np.abs(angles_norm) > 170)
    strong_mask = mag > (mag.max() * 0.1)
    mask        = strong_mask & ~horiz_mask

    if mask.sum() < 50:
        mask = strong_mask   # fallback: use all strong edges
    if mask.sum() < 20:
        return 0.0

    angles  = angles_norm[mask]
    hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
    hist    = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return float(entropy)


def _stroke_width_variance(region_binary):
    """Measure variation in stroke width. Higher = more handwritten."""
    dist        = cv2.distanceTransform(region_binary, cv2.DIST_L2, 3)
    ink_pixels  = dist[region_binary > 0]
    if len(ink_pixels) < 20:
        return 0.0
    return float(np.std(ink_pixels))


def classify_region(region_gray, region_binary, skew_offset=0.0):
    """Returns: 'handwritten', 'typed', or 'mixed'"""
    entropy  = _edge_angle_entropy(region_gray, skew_offset=skew_offset)
    variance = _stroke_width_variance(region_binary)

    # Each signal casts a vote (0 = typed, 1 = handwritten, 0.5 = uncertain)
    hw_votes = 0.0

    if entropy >= ENTROPY_HW_THRESHOLD:
        hw_votes += 1.0
    elif entropy >= ENTROPY_TYPED_THRESHOLD:
        hw_votes += 0.5

    if variance >= VARIANCE_HW_THRESHOLD:
        hw_votes += 1.0
    elif variance >= VARIANCE_TYPED_THRESHOLD:
        hw_votes += 0.5

    if hw_votes >= 1.5:
        return "handwritten", entropy, variance
    elif hw_votes >= 0.5:
        return "mixed", entropy, variance
    else:
        return "typed", entropy, variance


def step4_classify(regions, gray, binary, img, prefix, skew_offset=0.0):
    debug    = img.copy()
    results  = []

    # Color map: typed=green, handwritten=red, mixed=orange
    colors = {
        "typed":       (0,   200,  0),
        "handwritten": (0,   0,    220),
        "mixed":       (0,   140,  255),
    }
    label_short = {
        "typed": "T", "handwritten": "HW", "mixed": "MX"
    }

    for (x, y, rw, rh) in regions:
        region_gray   = gray[y:y+rh, x:x+rw]
        region_binary = binary[y:y+rh, x:x+rw]

        label, entropy, variance = classify_region(
            region_gray, region_binary, skew_offset=skew_offset)
        color = colors[label]

        cv2.rectangle(debug, (x, y), (x+rw, y+rh), color, 2)
        cv2.putText(debug, label_short[label],
                    (x+4, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        results.append({
            "bbox":     (x, y, rw, rh),
            "label":    label,
            "entropy":  round(entropy, 3),
            "variance": round(variance, 3),
        })

    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step4_classified.jpg", debug)

    typed_count = sum(1 for r in results if r["label"] == "typed")
    hw_count    = sum(1 for r in results if r["label"] == "handwritten")
    mixed_count = sum(1 for r in results if r["label"] == "mixed")
    print(f"  [Step 4] Classification — "
          f"typed: {typed_count}, handwritten: {hw_count}, mixed: {mixed_count}")

    # Add legend
    legend_img = debug.copy()
    cv2.putText(legend_img, "GREEN=Typed  RED=Handwritten  ORANGE=Mixed",
                (10, debug.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step4_classified.jpg", legend_img)

    return results


# ════════════════════════════════════════════════════════
# STEP 5 — ISOLATE HANDWRITTEN REGIONS
# Show only handwritten + mixed regions highlighted
# ════════════════════════════════════════════════════════
def step5_hw_only(results, img, prefix):
    # Dim the full image
    dimmed = (img * 0.35).astype(np.uint8)
    output = dimmed.copy()

    hw_regions = [r for r in results
                  if r["label"] in ("handwritten", "mixed")]

    for r in hw_regions:
        x, y, rw, rh = r["bbox"]
        # Restore original brightness in handwritten regions
        output[y:y+rh, x:x+rw] = img[y:y+rh, x:x+rw]
        color = (0, 0, 220) if r["label"] == "handwritten" else (0, 140, 255)
        cv2.rectangle(output, (x, y), (x+rw, y+rh), color, 3)

    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step5_hw_only.jpg", output)
    print(f"  [Step 5] Handwritten regions isolated: {len(hw_regions)}")
    return hw_regions


# ════════════════════════════════════════════════════════
# STEP 6 — MEASURE HANDWRITING CLARITY
#
# Three metrics, only on handwritten/mixed regions:
#
# A) Stroke Fill Ratio
#    % of region that is ink after adaptive threshold.
#    Too little = faint writing (< 3%)
#    Too much   = ink bleed or heavy pen (> 20%)
#    Ideal: 3–15%
#
# B) Local Contrast in Stroke Band
#    Instead of global contrast, measure std only in a
#    dilated band around actual strokes. This captures
#    how dark the ink is vs the surrounding paper —
#    ignoring blank areas that would dilute the score.
#
# C) Stroke Continuity
#    Count connected components per 10k pixels.
#    Fewer components = more connected strokes = clearer writing.
#    Many tiny components = broken/fragmented strokes = unclear.
#
# Final score: weighted average of all three → 0–100
# ════════════════════════════════════════════════════════
def _clamp(val, lo, hi):
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))


def measure_hw_clarity(region_gray, region_binary):
    h, w = region_gray.shape
    if h < 10 or w < 10:
        return 0.0, {}

    # ── A: Stroke Fill Ratio ──────────────────────────
    ink_pixels  = np.sum(region_binary > 0)
    total_px    = h * w
    fill_pct    = ink_pixels / total_px * 100
    # Ideal: 3–15%. Score drops below 3 (too faint) and above 20 (bleed)
    if fill_pct < 3:
        fill_score = _clamp(fill_pct, 0.5, 3)
    elif fill_pct <= 15:
        fill_score = 100.0
    else:
        fill_score = _clamp(20 - fill_pct, 0, 5)

    # ── B: Local Contrast in Stroke Band ─────────────
    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stroke_band = cv2.dilate(region_binary, kernel, iterations=2)
    ink_region  = region_gray[stroke_band > 0]
    if len(ink_region) > 20:
        local_contrast = float(np.std(ink_region))
    else:
        local_contrast = 0.0
    contrast_score = _clamp(local_contrast, 10, 65)

    # ── C: Stroke Continuity ─────────────────────────
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        region_binary, connectivity=8
    )
    # Filter out tiny components (< 5px) — these are pure noise
    valid_components = sum(
        1 for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 5
    )
    density = valid_components / (total_px / 10000 + 1e-9)
    # Low density = connected = good. High = fragmented = bad.
    # Typical clear handwriting: 10–80 components per 10k px
    continuity_score = _clamp(150 - density, 0, 150)

    # ── Final weighted score ──────────────────────────
    score = (fill_score       * 0.25 +
             contrast_score   * 0.45 +
             continuity_score * 0.30)
    score = round(score, 1)

    breakdown = {
        "fill_pct":        round(fill_pct, 2),
        "fill_score":      round(fill_score, 1),
        "local_contrast":  round(local_contrast, 2),
        "contrast_score":  round(contrast_score, 1),
        "components":      valid_components,
        "continuity_score":round(continuity_score, 1),
        "clarity_score":   score,
    }
    return score, breakdown


def step6_clarity(hw_regions, gray, binary, img, results, prefix):
    debug       = img.copy()
    all_scores  = []
    report_rows = []

    for r in results:
        x, y, rw, rh    = r["bbox"]
        region_gray     = gray[y:y+rh, x:x+rw]
        region_binary   = binary[y:y+rh, x:x+rw]

        if r["label"] in ("handwritten", "mixed"):
            score, breakdown = measure_hw_clarity(region_gray, region_binary)
            all_scores.append(score)

            # Color box by clarity score
            if score >= 70:
                color = (0, 200, 0)      # green = clear
            elif score >= 45:
                color = (0, 165, 255)    # orange = moderate
            else:
                color = (0, 0, 220)      # red = unclear

            cv2.rectangle(debug, (x, y), (x+rw, y+rh), color, 3)
            cv2.putText(debug, f"{score}",
                        (x+4, y+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            report_rows.append({**r, **breakdown})
        else:
            # Typed regions — draw grey, no score
            cv2.rectangle(debug, (x, y), (x+rw, y+rh), (150, 150, 150), 1)
            report_rows.append({**r,
                                 "clarity_score": "N/A (typed)",
                                 "fill_pct": "-",
                                 "local_contrast": "-",
                                 "components": "-"})

    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_step6_clarity_map.jpg", debug)

    avg_clarity = round(np.mean(all_scores), 1) if all_scores else 0.0
    print(f"  [Step 6] Handwriting clarity scores: {[s for s in all_scores]}")
    print(f"  [Step 6] Average handwriting clarity: {avg_clarity}/100")

    return report_rows, avg_clarity


# ════════════════════════════════════════════════════════
# WRITE REPORT
# ════════════════════════════════════════════════════════
def write_report(report_rows, avg_clarity, file_name, prefix):
    path = f"{OUTPUT_DIR}/{prefix}_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"HANDWRITING ANALYSIS REPORT\n")
        f.write(f"File : {file_name}\n")
        f.write(f"{'='*65}\n\n")
        f.write(f"Average Handwriting Clarity : {avg_clarity}/100\n\n")
        f.write(f"{'─'*65}\n")
        f.write(f"{'Region':<8} {'Label':<14} {'Entropy':<10} "
                f"{'Variance':<10} {'Clarity':<10} {'Fill%':<8} "
                f"{'Contrast':<10} {'Components'}\n")
        f.write(f"{'─'*65}\n")

        for i, r in enumerate(report_rows, 1):
            f.write(
                f"{i:<8} "
                f"{r['label']:<14} "
                f"{r['entropy']:<10} "
                f"{r['variance']:<10} "
                f"{str(r.get('clarity_score', 'N/A')):<10} "
                f"{str(r.get('fill_pct', '-')):<8} "
                f"{str(r.get('local_contrast', '-')):<10} "
                f"{str(r.get('components', '-'))}\n"
            )
        f.write(f"\n{'─'*65}\n")
        f.write("Clarity Scale:\n")
        f.write("  70–100 : GREEN  — Clear, readable handwriting\n")
        f.write("  45–69  : ORANGE — Moderate clarity, may affect extraction\n")
        f.write("  0–44   : RED    — Unclear, extraction likely to struggle\n")
    print(f"  [Report] Saved: {path}")


# ════════════════════════════════════════════════════════
# MAIN — RUN FULL PIPELINE ON EACH FILE
# ════════════════════════════════════════════════════════
def run_pipeline(file_path):
    file_name = os.path.basename(file_path)
    prefix    = os.path.splitext(file_name)[0].replace(" ", "_")[:40]

    print(f"\n{'='*65}")
    print(f"  Processing: {file_name}")
    print(f"{'='*65}")

    # For PDFs, process page 0 only (extend later for multi-page)
    img    = load_document(file_path, page_num=0)
    print(f"  Image size: {img.shape[1]}x{img.shape[0]}px")

    gray         = step1_grayscale(img, prefix)
    binary       = step2_threshold(gray, prefix)

    # Step 2.5 — detect global skew before region classification
    # Need binary_clean for skew detection (ruling lines removed)
    w_img        = binary.shape[1]
    kernel_wide  = cv2.getStructuringElement(cv2.MORPH_RECT, (w_img // 3, 1))
    lines_wide   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_wide)
    binary_nodots= cv2.subtract(binary, lines_wide)
    skew_offset  = step25_detect_skew(binary_nodots, prefix)

    regions = step3_find_regions(binary, img, prefix)

    if not regions:
        print("  ⚠️  No text regions found. Check step2 threshold output.")
        return

    results        = step4_classify(regions, gray, binary, img, prefix,
                                    skew_offset=skew_offset)
    hw_regions     = step5_hw_only(results, img, prefix)
    report_rows, avg = step6_clarity(hw_regions, gray, binary,
                                      img, results, prefix)
    write_report(report_rows, avg, file_name, prefix)

    print(f"\n  ✅ Done. Debug images saved to: {OUTPUT_DIR}/")
    print(f"     Prefix used: {prefix}")


if __name__ == "__main__":
    for f in TEST_FILES:
        if os.path.exists(f):
            run_pipeline(f)
        else:
            print(f"⚠️  File not found: {f}")

    print(f"\n{'='*65}")
    print(f"All files processed. Open '{OUTPUT_DIR}/' to see debug images.")
    print(f"{'='*65}")