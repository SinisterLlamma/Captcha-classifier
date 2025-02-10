# Captcha Classifier Project

This repository contains multiple tasks for CAPTCHA generation and classification.

---

## Task 0: CAPTCHA Generation Scripts
Scripts that generate diverse CAPTCHA datasets.

- **hard.py**  
  Creates challenging CAPTCHAs with noisy, textured backgrounds and varied fonts.  
  **Run:** `python3 hard.py`
   Access Dataset at https://www.kaggle.com/datasets/eshaansharmaog/hard-captcha-data-set-50k/data

- **hard_variations.py**  
  Generates multiple variations per word by varying capitalization and spacing.  
  **Run:** `python3 hard_variations.py`
Access Dataset at https://www.kaggle.com/datasets/eshaansharmaog/hard-data-set-precog
- **easy.py**  
  Produces simple CAPTCHAs on a white background using a fixed font.  
  **Run:** `python3 easy.py`
  Access Dataset at https://www.kaggle.com/datasets/eshaansharmaog/precogtask-easy/

- **bonus.py**  
  Generates bonus CAPTCHAs with custom noisy backgrounds and special visual effects.  
  **Run:** `python3 bonus.py`

---

## Task 1: Baseline CAPTCHA Classifier
A script implementing a robust feed-forward network for CAPTCHA classification.

- **baseline_classifier.py**  
  Trains a deep MLP on CAPTCHA images, includes error analysis and visualization.  
  **Run:** `python3 baseline_classifier.py`

---

## Task 2: Segmentation & CRNN Models
Contains modules and notebooks for segmenting CAPTCHA characters and experimenting with CRNN architectures.

- **task2_segmentation_cnn.py**  
  Provides functions for segmenting images and training a character classifier.  
  **Usage:** Import functions into your training pipeline.

- **task-2-crnn.ipynb**  
  Jupyter Notebook performing experiments with a CRNN architecture for CAPTCHA decoding.

- **notebook4baf5fb2d1.ipynb**  
  Alternate CRNN experiment notebook with different training strategies.

- **added-batch-norm.ipynb**  
  CRNN notebook incorporating batch normalization layers.

- **final-attempt-task1.ipynb**  
  A notebook with a final attempt at enhancing task 1 results.

- **Easy_data_set_simple-cnn.ipynb**  
  A notebook for training a simple CNN on an easy CAPTCHA dataset.

---

## Requirements
- Python 3.x
- Pillow & numpy (for Task 0)
- PyTorch & torchvision (for Task 1 & Task 2)
- Additional packages: scikit-learn, matplotlib, tqdm (as needed)
- Verify font paths in CAPTCHA scripts match your system or update accordingly.

---

## How to Run
1. Navigate to the project directory:
   ```bash
   cd /Users/eshaan/Projects/Captcha-classifier/Task\ 0/
   ```
2. Run Task 0 scripts with:
   ```bash
   python3 hard.py
   python3 hard_variations.py
   python3 easy.py
   python3 bonus.py
   ```
3. For Task 1, run:
   ```bash
   python3 baseline_classifier.py
   ```
4. For Task 2, launch the notebooks in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

