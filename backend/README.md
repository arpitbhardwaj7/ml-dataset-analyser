# ML Dataset Analyser - Backend

The FastAPI backend server for ML Dataset Analyser. This component handles all data analysis, ML model training, and LLM integration for the application.

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- OpenAI API key (optional, for LLM features)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file** (optional but recommended):
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   DEBUG=False
   ENABLE_LLM=true
   ```

4. **Run the server**:
   ```bash
   python run.py
   ```

   The server will start at `http://localhost:8000`

## API Documentation

Interactive API documentation is automatically generated and available at:

- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer
- **ReDoc**: `http://localhost:8000/redoc` - Alternative documentation view
- **OpenAPI Schema**: `http://localhost:8000/openapi.json` - Raw OpenAPI specification

## API Endpoints

### Core Analysis Endpoint

**POST** `/api/v1/analyze`

Analyzes a dataset and returns comprehensive ML insights.

**Request:**
```
Content-Type: multipart/form-data

file: <binary file data> (required)
target_column: "column_name" (optional)
problem_type: "auto|classification|regression" (optional, default: "auto")
use_llm_insights: true|false (optional, default: true)
```

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
      "completeness": { "score": 90, "grade": "A", "description": "Excellent" },
      "consistency": { "score": 85, "grade": "A", "description": "Good" },
      "balance": { "score": 75, "grade": "B", "description": "Fair" },
      "dimensionality": { "score": 88, "grade": "A", "description": "Excellent" },
      "separability": { "score": 82, "grade": "A", "description": "Good" }
    }
  },
  "data_profile": {
    "missing_values_percentage": 2.5,
    "duplicate_rows": 10,
    "duplicate_percentage": 1.0,
    "column_types": { "integer": 10, "float": 5, "object": 5 },
    "memory_usage_mb": 0.45,
    "numeric_summary": { "total_features": 15, "high_correlation_pairs": 3, "low_variance_features": 1 },
    "categorical_summary": { "total_features": 5, "high_cardinality": 1, "binary_features": 2 }
  },
  "issues_detected": [
    { "issue": "Class Imbalance", "severity": "high", "description": "Class A: 70%, Class B: 30%" },
    { "issue": "Missing Values", "severity": "low", "description": "2.5% of data is missing" }
  ],
  "model_recommendations": [
    { "model": "RandomForest", "rank": 1, "reasoning": "Handles non-linear patterns well" },
    { "model": "LogisticRegression", "rank": 2, "reasoning": "Baseline classifier" },
    { "model": "XGBoost", "rank": 3, "reasoning": "Excellent for structured data" }
  ],
  "preprocessing_recommendations": [
    { "step": "Handle Missing Values", "priority": "high", "method": "forward_fill" },
    { "step": "Scale Features", "priority": "medium", "method": "standardization" }
  ],
  "llm_insights": "Your dataset shows strong potential for a classification model. Consider addressing the class imbalance through resampling..."
}
```

### Health Check Endpoints

**GET** `/ping`
```json
{ "message": "pong" }
```

**GET** `/`
```json
{
  "message": "Welcome to ML Dataset Analyser",
  "version": "1.0.0",
  "docs": "/docs",
  "status": "running"
}
```

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application setup
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py               # API route aggregation
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── analyze.py           # Main analysis endpoint
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                   # Configuration management
│   │   └── analyzer/                   # Core analysis modules
│   │       ├── __init__.py
│   │       ├── data_profiler.py        # Statistical profiling
│   │       ├── quality_scorer.py       # Quality assessment engine
│   │       ├── leakage_detector.py     # Data leakage detection
│   │       ├── consistency_checker.py  # Data consistency validation
│   │       ├── model_recommender.py    # ML model selection
│   │       ├── baseline_model_trainer.py # Baseline model training
│   │       ├── non_linear_scorer.py    # Non-linear feature analysis
│   │       ├── blended_scorer.py       # Composite scoring
│   │       ├── smart_row_sampler.py    # Intelligent sampling
│   │       └── llm_validator.py        # LLM-powered validation
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm_client.py               # OpenAI API wrapper
│   │   ├── enhanced_llm_client.py      # Enhanced LLM functionality
│   │   ├── prompt_templates.py         # LLM prompt templates
│   │   └── robust_json_parser.py       # LLM output parsing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py                 # Request schemas (Pydantic)
│   │   └── responses.py                # Response schemas (Pydantic)
│   ├── services/
│   │   ├── __init__.py
│   │   └── analysis_service.py         # Business logic orchestration
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py             # File upload/processing
├── run.py                               # Application entry point
└── requirements.txt                     # Python dependencies
```

## Core Analysis Modules

### Data Profiler (`core/analyzer/data_profiler.py`)
Performs comprehensive statistical analysis of datasets including:
- Basic info (rows, columns, data types)
- Missing value analysis
- Duplicate detection
- Statistical summaries
- Correlation analysis
- Outlier detection
- Feature importance calculation
- Data leakage detection
- Data consistency checks

### Quality Scorer (`core/analyzer/quality_scorer.py`)
Evaluates dataset quality across 5 dimensions:
1. **Completeness**: Missing values, data coverage
2. **Consistency**: Duplicates, data integrity, format consistency
3. **Balance**: Class distribution, feature representation
4. **Dimensionality**: Feature relevance, redundancy, multicollinearity
5. **Separability**: Class distinguishability, feature separation

Returns overall score (0-100) and grade (A-F).

### Model Recommender (`core/analyzer/model_recommender.py`)
Recommends suitable ML models based on:
- Dataset size and shape
- Problem type (classification/regression)
- Data characteristics (linear/non-linear patterns)
- Class balance
- Feature types

### Leakage Detector (`core/analyzer/leakage_detector.py`)
Identifies potential data leakage patterns:
- Column name-based patterns (ID, index columns)
- Statistical indicators of leakage
- Temporal leakage in time series
- Custom pattern detection

### Consistency Checker (`core/analyzer/consistency_checker.py`)
Validates data consistency:
- Format consistency across columns
- Type consistency
- Value range validation
- Duplicate detection

### LLM Validator (`core/analyzer/llm_validator.py`)
Uses GPT-4o to provide:
- Natural language insights about data quality
- Actionable recommendations
- Risk assessment
- Domain-specific observations

## Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# OpenAI / LLM Configuration
OPENAI_API_KEY=sk-your-api-key-here        # Your OpenAI API key
LLM_MODEL=gpt-4o                            # LLM model to use
LLM_MAX_TOKENS=2000                         # Max tokens per response
ENABLE_LLM=true                             # Enable/disable LLM features

# Application Configuration
DEBUG=false                                 # Debug mode
MAX_FILE_SIZE_MB=100                        # Maximum upload file size
ALLOWED_EXTENSIONS=csv,xlsx                 # Allowed file types

# Server Configuration
HOST=0.0.0.0                                # Server host
PORT=8000                                   # Server port
RELOAD=false                                # Auto-reload on code changes

# API Configuration
API_V1_PREFIX=/api/v1                       # API prefix
PROJECT_NAME=ML Dataset Analyser            # Project name
PROJECT_VERSION=1.0.0                       # Project version

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://127.0.0.1:3000"]
```

### File Configuration

Edit `app/core/config.py` to modify default settings:

```python
class Settings(BaseSettings):
    openai_api_key: str = ""                # OpenAI API key
    llm_model: str = "gpt-4o"               # LLM model
    max_file_size_mb: int = 100             # Max file size
    debug: bool = False                     # Debug mode
    host: str = "0.0.0.0"                   # Server host
    port: int = 8000                        # Server port
    # ... more settings
```

## Dependencies

Key dependencies (see `requirements.txt` for complete list):

- **fastapi** (0.104.1) - Web framework
- **uvicorn** (0.24.0) - ASGI server
- **pydantic** (≥2.7.4) - Data validation
- **pandas** (2.1.4) - Data manipulation
- **scikit-learn** (1.3.2) - ML algorithms
- **numpy** (1.25.2) - Numerical computing
- **ydata-profiling** (4.6.4) - Data profiling
- **openai** (≥1.58.1) - LLM integration
- **matplotlib** (3.8.2) - Visualization
- **plotly** (5.17.0) - Interactive plots
- **scipy** (1.11.4) - Scientific computing

## Testing

Run the test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run specific test file:

```bash
pytest tests/test_analyzer.py
```

## Deployment

### Development

```bash
python run.py
```

### Production

Use a production ASGI server:

```bash
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Or with uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
```

Build and run:

```bash
docker build -t ml-analyser-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ml-analyser-backend
```

## Security Best Practices

1. **API Keys**: Never commit `.env` files. Use environment variables or secrets management.
2. **CORS**: Configure CORS origins appropriately for production.
3. **File Uploads**: Validate file types and sizes. Current limits: CSV/XLSX, max 100MB.
4. **Input Validation**: All inputs are validated using Pydantic schemas.
5. **Error Handling**: Sensitive errors are hidden in production mode.

## Troubleshooting

### Port 8000 Already in Use
```bash
# Find and kill the process using port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port
export PORT=8001
python run.py
```

### OpenAI API Errors
- Verify your `OPENAI_API_KEY` in `.env`
- Check your OpenAI account balance and API limits
- Ensure `ENABLE_LLM=true` if you want LLM features

### Dependency Issues
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### CORS Errors
Add your frontend URL to `CORS_ORIGINS` in `.env`:
```env
CORS_ORIGINS=["http://localhost:3000", "http://your-domain.com"]
```

## Analysis Process Flow

1. **File Upload & Validation**
   - Check file type and size
   - Parse CSV/XLSX into DataFrame

2. **Initial Analysis**
   - Data Profiler generates comprehensive statistics
   - Target column auto-detection

3. **Quality Assessment**
   - Quality Scorer evaluates across 5 dimensions
   - Issue detection (imbalance, leakage, etc.)

4. **ML Insights**
   - Model Recommender suggests top 3 models
   - Baseline models trained for comparison
   - Preprocessing recommendations generated

5. **LLM Enrichment** (optional)
   - LLM Validator provides natural language insights
   - GPT-4o generates contextual recommendations

6. **Response Assembly**
   - All results compiled into comprehensive response
   - Returned to frontend for visualization

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

## Support

For general information, see the [main README](../README.md).
For frontend documentation, see [Frontend README](../frontend/README.md).
