# ML Dataset Analyser

An AI-powered web application for analyzing machine learning datasets. Upload a CSV or XLSX file and get a comprehensive quality report with model recommendations and GPT-4o-powered insights.

## Features

- **Quality Scoring** — Multi-dimensional assessment across 5 dimensions: Completeness, Consistency, Balance, Dimensionality, Separability
- **Auto-Detection** — Automatically identifies problem type (classification/regression) and target column
- **Model Recommendations** — Top 3 ML model suggestions based on dataset characteristics
- **Issue Detection** — Data leakage, class imbalance, outliers, high correlation, missing values
- **LLM Insights** — GPT-4o analysis with validated, grounded recommendations (optional)

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key (optional, for LLM insights)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.template .env          # Add your OPENAI_API_KEY
python run.py
```

API runs at `http://localhost:8000` — interactive docs at `/docs`.

### Frontend

```bash
cd frontend
npm install
npm start
```

App opens at `http://localhost:3000`.

## API

**POST** `/api/v1/analyze`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | `multipart/form-data` | Yes | CSV or XLSX file (max 100MB) |
| `target_column` | string | No | Target column name (auto-detected if omitted) |
| `problem_type` | string | No | `auto`, `classification`, or `regression` |
| `use_llm_insights` | boolean | No | Enable GPT-4o insights (default: `true`) |

**Health checks:** `GET /` and `GET /ping`

## Environment Variables

Create `backend/.env` from `backend/.env.template`:

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
LLM_MAX_TOKENS=2000
ENABLE_LLM=true
DEBUG=false
MAX_FILE_SIZE_MB=100
ALLOWED_EXTENSIONS=csv,xlsx
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Port 8000 in use | Change `PORT` in `.env` or kill the process |
| OpenAI API errors | Verify `OPENAI_API_KEY` in `.env` or set `ENABLE_LLM=false` |
| Missing dependencies (backend) | `pip install -r requirements.txt` |
| Frontend can't reach backend | Ensure backend is running on port 8000; check `CORS_ORIGINS` |
| Port 3000 in use | `npm start -- --port 3001` |
| Module not found (frontend) | `npm install` |

For more detail, see [Backend README](./backend/README.md) and [Frontend README](./frontend/README.md).
