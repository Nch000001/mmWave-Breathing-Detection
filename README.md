# mmWave Breathing Detection

Breathing detection from mmWave radar signals using signal processing and a lightweight CNN classifier.

This project implements a full pipeline for extracting breathing signals from radar data, generating training datasets, training a neural network classifier, and performing real-time inference.

---

# Project Overview

This system uses **mmWave radar data** to detect human breathing patterns without physical contact.

The pipeline combines:

- signal processing
- breathing cycle detection
- dataset generation
- deep learning classification
- real-time inference

Radar recordings are stored in `.h5` format and processed to extract respiratory signals.

---

# System Pipeline

```
Radar .h5 Data
↓
Breathing Signal Extraction
↓
Cycle Detection
↓
Segment Dataset Generation
↓
CNN Training
↓
Real-time Inference
```

---

# Project Structure

```
mmWave-Breathing-Detection
│
├── batch_cycle_v2_label_gated.py
├── make_segments_v3.py
├── train_breath_cnn.py
├── realtime_sim_batch_integrated.py
│
├── models
│
├── data
│
└── README.md
```

---

# Core Components

## 1. Breathing Cycle Detection

`batch_cycle_v2_label_gated.py`

This script processes radar `.h5` recordings and extracts breathing waveforms.

Key steps:

- ROI selection using radar energy
- bandpass filtering (breathing frequency range)
- FFT-based breathing frequency estimation
- trough detection using peak detection
- event merging and interpolation
- breathing presence gating

Outputs:

- breathing events
- estimated BPM
- signal visualization plots

---

## 2. Dataset Generation

`make_segments_v3.py`

This script converts breathing signals into training samples for deep learning.

Process:

- sliding window segmentation
- label assignment
- normalization
- dataset packaging

Output:


dataset_segments.pt


Dataset format:


X : breathing waveform segments
y : labels
---

## 3. CNN Training

`train_breath_cnn.py`

A lightweight **1D Convolutional Neural Network** is trained to classify breathing segments.

Model architecture:


Conv1D → ReLU → MaxPool
Conv1D → ReLU → MaxPool
Conv1D → ReLU → AdaptiveAvgPool
Fully Connected Layer


Training features:

- file-level train/validation split
- robust normalization
- F1-score monitoring
- best model checkpoint saving

Output model:


breathclf_best.pt


---

## 4. Real-time Simulation

`realtime_sim_batch_integrated.py`

This script simulates real-time breathing detection by feeding radar data sequentially into the trained CNN model.

Features:

- sliding window inference
- breathing classification
- prediction smoothing
- real-time output simulation

---

# Requirements

Python 3.8+

Main libraries:


numpy
torch
scipy
pandas
h5py
matplotlib


Install dependencies:


pip install numpy torch scipy pandas h5py matplotlib


---

# Training

Step 1 — Generate breathing events


python batch_cycle_v2_label_gated.py
--in_dir data/h5
--out_dir output_events


Step 2 — Create dataset


python make_segments_v3.py


Step 3 — Train CNN


python train_breath_cnn.py
--data dataset_segments.pt


---

# Inference

Run real-time simulation:


python realtime_sim_batch_integrated.py


---

# Applications

This system can be used for:

- non-contact respiratory monitoring
- healthcare sensing
- sleep monitoring
- smart environments
- radar-based human sensing

---

# Future Work

Possible improvements:

- Transformer / TCN models
- multi-person breathing detection
- real-time GUI integration
- pain index estimation
- mmWave vital sign monitoring

---

# License

MIT License
