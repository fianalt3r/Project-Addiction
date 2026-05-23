# 📱 Social Media Addiction Analysis

> Studying how excessive social media affects **productivity**, **concentration**, and **emotional health** among students.

---

## 📁 Project Structure

```
Social Media Addiction Analysis/
├── data/
│   ├── students_social_media_addiction.csv   ← Dataset
│   ├── shap_background.csv                   ← Auto generated on first run
│   └── predictions_log.csv                   ← Auto generated on each prediction
├── model/
│   └── social_media_model.pkl                ← Trained pipeline (auto generated)
├── notebook/
│   └── notebook.ipynb                        ← EDA, training, evaluation
├── src/
│   ├── backend/
│   │   └── main.py                           ← FastAPI (all logic)
│   └── frontend/
│       └── app.py                            ← Streamlit (all pages)
├── .github/
│   └── workflows/
│       └── deploy.yml                        ← CI/CD
├── Dockerfile
├── start.sh
├── requirements.txt
└── README.md
```

---

## 📋 Dataset

Place the CSV in `data/students_social_media_addiction.csv`

**705 students · 13 columns · Real survey data**

---

## ⚙️ Setup & Run

### Option A — Local (Two terminals)

```bash
# Install
pip install -r requirements.txt

# Terminal 1 — FastAPI backend (auto trains model on first run)
uvicorn src.backend.main:app --reload

# Terminal 2 — Streamlit frontend
streamlit run src/frontend/app.py
```

### Option B — Single command
```bash
chmod +x start.sh
./start.sh
```

### Option C — Docker
```bash
docker build -t social-media-addiction .
docker run -p 8000:8000 -p 8501:8501 social-media-addiction
```

---

## 🌐 URLs

| Service | URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |

---

## 📊 Dashboard Pages

| Page | Theme |
|---|---|
| 🏠 Overview | Dataset stats, risk distribution |
| 📺 Screen-Time Analysis | Usage by platform, country, academic level |
| 📚 Productivity & Concentration | Academic impact, sleep analysis, conflicts |
| 😊 Mood vs Usage | Mental health vs screen time, heatmap |
| 🤖 Model Performance | RF vs LR, confusion matrix, F1 scores |
| 🔮 Predict My Risk | Personal form → prediction + SHAP waterfall |
| 🧠 SHAP & Ethics | Global SHAP, what-if tool, digital ethics |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict risk + SHAP values |
| GET | `/model-metrics` | Accuracy, F1, confusion matrices |
| GET | `/feature-importance` | Global SHAP importance |
| GET | `/predictions-log` | Last 50 predictions made |

---

## 🤖 Tech Stack

| Tool | Role |
|---|---|
| Streamlit | Frontend — single `app.py`, sidebar navigation |
| FastAPI | Backend — single `main.py`, all logic |
| scikit-learn | Pipeline (Random Forest + Logistic Regression) |
| SHAP | Model explainability |
| Plotly | Interactive charts |
| Docker | Containerisation |
| GitHub Actions | CI/CD |
