# 🌤️ Vayu-Avekshan (SIH26073)

> **AI/ML-Based Intelligent Anomaly Detection & Self-Healing Pipeline for Automatic Weather Stations (AWS)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML: Isolation Forest](https://img.shields.io/badge/ML-Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![XAI: SHAP](https://img.shields.io/badge/XAI-SHAP-orange.svg)](https://shap.readthedocs.io/)

---

## 📌 Problem Overview & Impact

Automatic Weather Station (AWS) networks across India frequently suffer from sensor degradation, communication spikes, missing values, and calibration drifts. Traditional threshold-based validation often misidentifies genuine extreme weather events (such as heatwaves or cloudbursts) as sensor errors—or vice versa.

**Vayu-Avekshan** is an end-to-end telemetry validation system designed for **Problem Statement SIH26073** under the **Disaster Management** theme. It provides real-time fault detection, root-cause explainability, and automated data recovery without requiring additional hardware sensors.

### Key Capabilities:
* **Disambiguates Weather vs. Faults:** Uses cross-parameter and spatial consensus checks to separate true meteorologic extremes from hardware malfunction.
* **Explainable Diagnostics (XAI):** Generates instant SHAP feature-attribution scores for every flagged data point.
* **Self-Healing Telemetry:** Automatically gap-fills corrupted or missing sensor streams using statistical and historical interpolation.

---
┌─────────────────────────┐
│  1. Data Ingestion      │ ──► CSV Replay / Live Telemetry Ingest & Schema Check
└────────────┬────────────┘
│
┌────────────▼────────────┐
│  2. AI Detection Core   │ ──► Rule Engine (Physical Bounds) + Isolation Forest Model
└────────────┬────────────┘
│
┌────────────▼────────────┐
│  3. Validation & XAI    │ ──► T-H-P Cross-Parameter Check + Spatial Consensus + SHAP
└────────────┬────────────┘
│
┌────────────▼────────────┐
│  4. Dashboard & Healing │ ──► Streamlit Monitor + Severity Logs + Auto-Imputation
└─────────────────────────┘
---

## 🛠️ Tech Stack

* **Core Programming:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning & Isolation:** Scikit-Learn (`IsolationForest`)
* **Explainable AI (XAI):** SHAP (`TreeExplainer` / `KernelExplainer`)
* **Interactive Dashboard:** Streamlit, Plotly
* **Database & Logs:** SQLite

---

## 📊 Performance Metrics

* **Precision:** `0.94`
* **Recall:** `0.91`
* **F1-Score:** `0.92`
* **False Positive Rate (FPR):** `0.06`

---

## 🚀 Local Installation & Setup

Follow these steps to run the interactive Streamlit prototype locally:

### 1. Clone the Repository

git clone [https://github.com/simrandombivli-png/Vayu-Avekshan-Prototype.git](https://github.com/simrandombivli-png/Vayu-Avekshan-Prototype.git)
cd Vayu-Avekshan-Prototype
Install required dependencies
pip install -r requirements.txt
Launch Streamlit Dashboard
stramlit run app.py
The app will open automatically in your browser at http://localhost:8501.

🔗 Project Links & Submission Artifacts
Live Interactive Prototype: https://vayu-avekshan-prototype-42mjbhjgzmymenbr5hfp8i.streamlit.app/
Video Demonstration: YouTube Demo Link (Replace with your unlisted YouTube URL)
Team Name: Genesis Loop
Institute: Fr. Conceicao Rodrigues Institute of Technology (FCRIT) 
## ⚙️ System Architecture

The pipeline processes AWS telemetry through four sequential layers:
