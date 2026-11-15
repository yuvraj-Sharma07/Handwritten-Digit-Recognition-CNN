# Group 3 — Handwritten Digit Recognition with Explainable CNN

**Members**
- Dhaval Sundhesha — Student ID: `ADD_ID`
- Yuvraj Sharma — Student ID: `ADD_ID`
- Dev Shrimal — Student ID: `ADD_ID`

## Goal
Classify a handwritten digit (0–9) using a compact CNN trained on MNIST, then show a Grad-CAM heatmap that helps inspect the model's prediction.

## Requirements
Python 3.10 or 3.11 is recommended. Install packages in a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate
From this project directory:

```bash
python train_model.py
```

The script downloads MNIST through Keras on first use, trains the model, uses 5,000 examples from the official training split for validation, and evaluates on the official 10,000-image test split. It writes:
- `models/digit_cnn.keras`
- `results/metrics.json`
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/training_history.png`

**Do not copy example metrics into the report.** Run the script on your own machine and transcribe the actual values. Record laptop model, OS, Python/TensorFlow versions, and whether a CPU or GPU was used. Inference time depends strongly on hardware and runtime conditions.

## Run the demo
```bash
streamlit run Group_3_1.py
```
Choose Upload image or Draw digit. The drawing mode uses `streamlit-drawable-canvas`. Uploaded images are converted to grayscale, inverted when the background appears bright, resized and centered on a 28×28 canvas. This simple preprocessing will not work perfectly for every photograph.

## Dataset and folder structure
MNIST is downloaded by `tf.keras.datasets.mnist.load_data()`. It contains 60,000 training images and 10,000 test images, each 28×28 grayscale. We intentionally do not package the full dataset; the script downloads it when first run. This keeps the submission small and avoids redundant files. If the course LMS explicitly requires image files in `data/train`, `data/validate`, and `data/test`, ask the instructor whether a dataset-download script is acceptable; otherwise export images into the required layout.

## Model summary
- Conv2D(16) → MaxPool
- Conv2D(32) → MaxPool
- Conv2D(64) named `last_conv`
- GlobalAveragePooling → Dropout(0.25) → Dense(10, softmax)

Loss: sparse categorical cross-entropy; optimizer: Adam; batch size: 128; maximum 12 epochs with early stopping.

## Evaluation protocol
1. Keep the official test split untouched during training and model selection.
2. Report test accuracy, macro precision/recall/F1, confusion matrix, model file size, training duration, and average single-image inference latency.
3. Include examples where Grad-CAM highlights digit strokes and at least one failure case if present.
4. Report the observed results and do not claim Grad-CAM proves causal reasoning.
5. Run the demo on the same laptop intended for presentation.

## Known limitations
MNIST images are centered, low-resolution, and high contrast. A phone photo or freehand canvas image can differ in scale, polarity, and background. Grad-CAM is coarse and model-dependent. A high confidence score is not necessarily a calibrated probability of correctness.

## Research references
See the IEEE-style references in `Group_3_report_1.pdf`. Primary source papers include LeCun et al. (1998), Krizhevsky et al. (2012), Simonyan and Zisserman (2015), He et al. (2016), and Selvaraju et al. (2017).
