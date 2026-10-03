# 🧬 DiaTwin — Multimodal Metabolic Digital Twin

> **Real-Time Glycemic Trajectory Forecasting & Counterfactual Intervention Simulation**
>
> *Digital Twin Challenge 2026 | Happiest Health Summit*

---

## 📋 Executive Summary

**DiaTwin** is an AI-powered *in-silico* physiological replica of a Type 2 Diabetes (T2DM) patient's metabolic state. It fuses **static Electronic Health Records (EHR)** with **dynamic wearable telemetry** (CGM, activity, heart rate) to:

1. **Forecast** continuous glucose trajectories over a **60–120 minute** prediction horizon with uncertainty quantification
2. **Flag** imminent hyperglycemic spikes (>180 mg/dL) and hypoglycemic dips (<70 mg/dL) with **≥45 min lead time**
3. **Simulate** counterfactual "What-If" scenarios — *What if the patient walks for 20 minutes? What if they eat 80g carbs?*

### Clinical Context
India has **101M+ diagnosed T2DM cases** with the characteristic "Asian Indian Phenotype" (lower BMI, higher visceral adiposity, severe insulin resistance). DiaTwin is calibrated for Indian metabolic profiles.

---

## 🏗️ Architecture

```
┌──────────────────────────┐     ┌──────────────────────────┐
│  Static EHR (Synthea)    │     │  Dynamic CGM Stream      │
│  Demographics, Labs, Meds│     │  5-min Glucose, Steps, HR│
└───────────┬──────────────┘     └───────────┬──────────────┘
            │                                │
            ▼                                ▼
   [RobustScaler + Encoding]      [Feature Engineering]
   [Entity Embeddings]            [Δ, Δ², Rolling Stats]
            │                     [Sin/Cos Temporal]
            └──────────┬─────────────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Static Contextual Gating   │
        │   (Dense Projection + Gate)  │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Bi-GRU + Multi-Head Self   │
        │   Attention (4 heads)        │
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │   Quantile Regression Head   │
        │   (10th, 50th, 90th pctile)  │
        └──────────────┬───────────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     [Alert Engine]    [Bergman Minimal Model]
     [Hypo/Hyper]      [Counterfactual ODE]
            │                     │
            └──────────┬──────────┘
                       ▼
              [Clinician Dashboard]
              [React + Recharts]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm
- (Optional) Docker & Docker Compose

### 1. Clone & Install

```bash
git clone https://github.com/YourTeam/DiaTwin.git
cd DiaTwin

# Backend
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Generate Synthetic Data

```bash
# Generate 1,000 Indian-calibrated EHR profiles
python scripts/generate_synthetic_ehr.py

# Generate 7-day CGM streams for each patient
python scripts/generate_cgm_data.py
```

### 3. Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load EHR profiles and CGM data
- Preprocess features (scaling, encoding, windowing)
- Train the BiGRU-Attention forecaster with quantile loss
- Evaluate with MARD, RMSE, and Clarke Error Grid Analysis
- Save the best checkpoint to `models/checkpoints/best_model.pt`

### 4. Start the Backend API

```bash
uvicorn backend.app.main:app --reload --port 8000
```

API documentation at: http://localhost:8000/docs

### 5. Start the Frontend

```bash
cd frontend
npm run dev
```

Dashboard at: http://localhost:5173

### Alternative: Docker Compose

```bash
docker-compose up --build
```
- Backend: http://localhost:8000
- Frontend: http://localhost:3000

---

## 📂 Project Structure

```
DiaTwin/
├── backend/
│   └── app/
│       ├── api/            # FastAPI routes
│       │   └── routes.py   # /patients, /predict, /simulate, /alerts
│       ├── core/
│       │   ├── config.py        # Central configuration
│       │   └── preprocessing.py # EHR + CGM preprocessors, Dataset
│       ├── models/
│       │   ├── forecaster.py    # BiGRU-Attention + Quantile Loss
│       │   └── trainer.py       # Training loop, early stopping
│       ├── schemas/
│       │   └── models.py        # Pydantic request/response models
│       ├── services/
│       │   ├── counterfactual.py # Bergman Minimal Model ODE
│       │   ├── evaluation.py    # MARD, RMSE, Clarke EGA
│       │   └── inference.py     # Model serving & caching
│       └── main.py              # FastAPI application entry
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── GlucoseChart.tsx     # Trajectory + confidence cones
│       │   ├── PatientCard.tsx      # EHR phenotype display
│       │   ├── PatientList.tsx      # Searchable patient sidebar
│       │   ├── WhatIfSimulator.tsx  # Counterfactual sandbox
│       │   └── AlertFeed.tsx        # Early warning feed
│       ├── api/client.ts            # API client with mock fallback
│       ├── types.ts                 # TypeScript interfaces
│       └── mockData.ts             # Realistic mock data
├── scripts/
│   ├── generate_synthetic_ehr.py   # 1,000 Indian-calibrated profiles
│   ├── generate_cgm_data.py        # 7-day CGM streams per patient
│   └── train_model.py              # End-to-end training pipeline
├── data/synthetic/                  # Generated data
├── models/checkpoints/              # Trained model weights
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── requirements.txt
```

---

## 🔬 Technical Details

### Predictive Model

| Component | Specification |
|---|---|
| Architecture | Bidirectional GRU + Multi-Head Self-Attention |
| Static Fusion | Contextual Gating Layer (dense projection + broadcast) |
| Lookback Window | 24 steps = 2 hours (5-min intervals) |
| Forecast Horizon | 24 steps = 2 hours |
| Output | Quantile predictions (10th, 50th, 90th percentiles) |
| Loss Function | Pinball / Quantile Loss |
| Parameters | ~400K trainable |
| Inference | <50ms on CPU |

### Counterfactual Simulation

The **Bergman Minimal Model** ODE system enables biophysically-grounded What-If analysis:

```
dG/dt = -(p₁ + X)·G(t) + p₁·Gᵦ + D(t)
dX/dt = -p₂·X(t) + p₃·(I(t) - Iᵦ) - γ·A(t)
```

Where D(t) models carbohydrate absorption and A(t) models exercise intensity, with patient-specific insulin resistance scaling.

### Clinical Evaluation Metrics

| Metric | Target | Description |
|---|---|---|
| MARD | < 10% | Mean Absolute Relative Difference |
| RMSE | < 18 mg/dL | Root Mean Squared Error |
| Clarke EGA A+B | > 98% | Clinically safe predictions |
| Hypo Sensitivity | > 88% | Nocturnal event detection |
| Alert Lead Time | ≥ 45 min | Warning before threshold breach |

---

## 🖥️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/patients` | List all patients |
| GET | `/api/v1/patients/{id}` | Get patient profile |
| GET | `/api/v1/patients/{id}/cgm` | Get CGM readings |
| GET | `/api/v1/predict/{id}` | Predict glucose trajectory |
| POST | `/api/v1/predict` | Predict with custom readings |
| POST | `/api/v1/simulate` | Run What-If simulation |
| GET | `/api/v1/alerts` | Get alert feed |

---

## 🔒 Regulatory & Privacy

- **Synthetic Data Only**: All patient records generated via Synthea-style synthesis — zero PHI
- **DPDP Act 2023 Compliant**: No personal data collection or storage
- **Clinical Disclaimer**: DiaTwin is an **investigational decision-support prototype**. Treatment adjustments must be verified with venous/capillary blood testing and licensed medical judgment.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Synthetic Data | Custom generators (NumPy, SciPy, Pandas) |
| ML Framework | PyTorch |
| API Server | FastAPI + Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS v4, Recharts |
| Containerization | Docker, Docker Compose |

---

## 👥 Team

*DiaTwin — Digital Twin Challenge 2026*

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
