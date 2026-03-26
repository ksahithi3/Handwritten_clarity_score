# ✍️ Handwritten Clarity Score Pipeline

## 📌 Overview

This project implements a **rule-based computer vision pipeline** to:

- Detect handwritten vs typed regions in documents
- Isolate handwritten content
- Compute a **clarity score (0–100)** for handwriting

The goal is to estimate how readable handwritten content is in scanned or photographed documents.

---

## ⚙️ Tech Stack

- Python
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- PyMuPDF (`fitz`)

---

## 🔄 Pipeline Flow

1. **Load document**
   - Supports PDF and image formats

2. **Grayscale conversion**
   - Removes color, keeps intensity

3. **Adaptive thresholding**
   - Extracts ink from background (handles shadows)

4. **Line removal + region detection**
   - Removes form lines
   - Detects text regions using projection profile + dilation

5. **Region classification**
   - Typed / Handwritten / Mixed  
   - Based on:
     - Edge angle entropy (Sobel)
     - Stroke width variation (distance transform)

6. **Handwriting isolation**
   - Keeps only handwritten + mixed regions

7. **Clarity scoring**
   - Based on:
     - Ink density (fill %)
     - Local contrast
     - Stroke continuity

---

## 🧠 Clarity Score Explanation

The clarity score (0–100) is computed using:

### 1. Ink Density (Fill %)
- Too low → faint writing  
- Too high → ink bleed  
- Ideal range: **3% – 15%**

### 2. Local Contrast
- Measures how dark and clear the strokes are  
- Higher contrast → clearer writing

### 3. Stroke Continuity
- Measures how connected the strokes are  
- More breaks → lower clarity  

---

## 📁 Output

All outputs are saved in: debug_output/

### Generated Files

- `*_step1_gray.jpg` → grayscale image  
- `*_step2_threshold.jpg` → ink extraction  
- `*_step3_regions.jpg` → detected regions  
- `*_step4_classified.jpg` → classification  
- `*_step5_hw_only.jpg` → handwriting only  
- `*_step6_clarity_map.jpg` → clarity visualization  
- `*_report.txt` → detailed metrics  

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ksahithi3/Handwritten_clarity_score.git
cd Handwritten_clarity_score

# 2. Install dependencies
pip install opencv-python numpy pillow pymupdf

# 3. Update file paths
# Open the script and update:
TEST_FILES = [
    "path_to_your_image_or_pdf"
]

# 4. Run the pipeline
python handwriting_pipeline.py

# 5. Check output
# Go to:
debug_output/