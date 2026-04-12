# ML Dataset Analyser — Backend

FastAPI backend that handles file parsing, statistical analysis, ML model evaluation, and LLM integration.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
cp .env.template .env          # fill in OPENAI_API_KEY
python run.py
```

Server starts at `http://localhost:8000`.

## API Endpoints

### `POST /api/v1/analyze`

Upload a dataset for analysis.

**Request** (`multipart/form-data`):
- `file` *(required)* — CSV or XLSX, max 100MB
- `target_column` *(optional)* — auto-detected if omitted
- `problem_type` *(optional)* — `auto` | `classification` | `regression`
- `use_llm_insights` *(optional)* — `true` | `false`, default `true`

**Response:**
```json
{
  "dataset_info": {
    "filename": "data.csv",
    "rows": 1000,
    "columns": 20,
    "size_mb": 0.5,
    "detected_problem_type": "classification",
    "detected_target_column": "label"
  },
  "quality_score": {
    "overall": 85.5,
    "grade": "A",
    "breakdown": {
      "completeness": { "score": 90, "grade": "A" },
      "consistency": { "score": 85, "grade": "A" },
      "balance":      { "score": 75, "grade": "B" },
      "dimensionality": { "score": 88, "grade": "A" },
      "separability": { "score": 82, "grade": "A" }
    }
  },
  "issues_detected": [
    { "issue": "Class Imbalance", "severity": "high", "description": "..." }
  ],
  "model_recommendations": [
    { "model": "RandomForest", "rank": 1, "reasoning": "..." }
  ],
  "preprocessing_recommendations": [
    { "step": "Handle Missing Values", "priority": "high", "method": "forward_fill" }
  ],
  "data_profile": { ... },
  "llm_insights": "..."
}
```

### `GET /ping` — Health check  
### `GET /` — Version info and links

Interactive docs at `http://localhost:8000/docs` (Swagger) and `/redoc`.

## Analysis Pipeline

1. **File Handler** — validates file type/size, parses CSV/XLSX, auto-detects target column and problem type
2. **Data Profiler** — missing values, duplicates, correlations, outliers, feature importance
3. **Quality Scorer** — scores across 5 dimensions (Completeness, Consistency, Balance, Dimensionality, Separability)
4. **Leakage Detector** — column name patterns, statistical indicators, temporal leakage
5. **Consistency Checker** — format and type consistency, value range validation
6. **Model Recommender** — selects top 3 models based on dataset shape, problem type, and data characteristics
7. **Baseline Model Trainer** — trains simple baseline for comparison
8. **Enhanced LLM Client** — samples dataset intelligently, sends to GPT-4o, validates findings against real statistics

## Configuration

All settings are in `app/core/config.py` (Pydantic `Settings`). Override via `.env`:

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
CORS_ORIGINS=["http://localhost:3000"]
API_V1_PREFIX=/api/v1
```

Set `ENABLE_LLM=false` to run without an OpenAI key.

## Production Deployment

```bash
# With gunicorn
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Troubleshooting

**Port 8000 in use (Windows):**
```bash
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**OpenAI errors:** Check `OPENAI_API_KEY` in `.env` and your account balance.

**Dependency issues:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**CORS errors:** Add your frontend origin to `CORS_ORIGINS` in `.env`.
