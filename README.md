# ML Dataset Analyser

A comprehensive, AI-powered web application for analyzing machine learning datasets and providing intelligent insights about data quality, ML model recommendations, and preprocessing guidance.

## Overview

ML Dataset Analyser is an enterprise-grade tool that combines traditional statistical analysis with machine learning techniques and LLM-powered insights to help data scientists and ML engineers make informed decisions about their datasets. Upload your CSV or XLSX files and get actionable insights within seconds.

## Key Features

### Comprehensive Data Analysis
- **Data Profiling**: Automatic statistical analysis of datasets including missing values, duplicates, and data types
- **Quality Scoring**: Multi-dimensional quality assessment across 5 key dimensions:
  - Completeness (missing values and data coverage)
  - Consistency (duplicate detection and data integrity)
  - Balance (class distribution and feature representation)
  - Dimensionality (feature relevance and redundancy)
  - Separability (distinguishability between classes)

### Intelligent Recommendations
- **Auto-Detection**: Automatically identifies problem type (classification/regression) and target columns
- **Model Recommendations**: Get top 3 ML model recommendations based on dataset characteristics
- **Preprocessing Guidance**: Receive prioritized, actionable preprocessing steps
- **LLM Insights**: AI-powered analysis and recommendations using GPT-4o (optional)

### Advanced Issue Detection
- **Data Leakage Detection**: Identifies potential data leakage patterns
- **Outlier Detection**: Finds statistical outliers and anomalies
- **Class Imbalance Detection**: Warns about imbalanced datasets
- **Feature Analysis**: Correlation analysis, feature importance, and multicollinearity detection
- **Consistency Checks**: Validates data consistency and integrity

### Beautiful, Intuitive UI
- SAP Fiori design system for enterprise-grade aesthetics
- Interactive dashboards and visualizations using Recharts
- Real-time analysis progress tracking
- Responsive design for all devices

## Project Structure

```
ml-dataset-analyser/
├── backend/              # FastAPI backend server
│   ├── app/
│   │   ├── api/         # API endpoints and routing
│   │   ├── core/        # Core analysis modules
│   │   │   └── analyzer/ # ML analysis engines
│   │   ├── models/      # Request/response schemas
│   │   ├── services/    # Business logic layer
│   │   ├── llm/         # LLM integration
│   │   └── utils/       # Utility functions
│   ├── run.py          # Application entry point
│   └── requirements.txt # Python dependencies
├── frontend/            # React TypeScript frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/    # API client service
│   │   ├── styles/      # Styling
│   │   └── types/       # TypeScript types
│   └── package.json     # Node dependencies
└── README.md           # This file
```

## Quick Start

### Prerequisites
- Python 3.8+ (for backend)
- Node.js 16+ (for frontend)
- OpenAI API key (optional, for LLM insights)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file (optional, for LLM features):
   ```env
   OPENAI_API_KEY=your_api_key_here
   DEBUG=False
   ```

5. Run the server:
   ```bash
   python run.py
   ```
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The application will open at `http://localhost:3000`

## API Documentation

Once the backend is running, interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Main Analysis Endpoint

**POST** `/api/v1/analyze`

Upload a CSV or XLSX file for comprehensive analysis.

**Parameters:**
- `file` (multipart/form-data, required): The dataset file
- `target_column` (optional): Target column name (auto-detected if not provided)
- `problem_type` (optional): `'auto'`, `'classification'`, or `'regression'`
- `use_llm_insights` (optional): Enable LLM-powered insights (default: `true`)

**Response:**
```json
{
  "dataset_info": {
    "filename": "data.csv",
    "rows": 1000,
    "columns": 20,
    "size_mb": 0.5
  },
  "quality_score": {
    "overall": 85.5,
    "grade": "A",
    "breakdown": { ... }
  },
  "issues_detected": [ ... ],
  "model_recommendations": [ ... ],
  "preprocessing_recommendations": [ ... ],
  "data_profile": { ... }
}
```

## Architecture

### Backend Architecture
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialization
- **Scikit-learn**: ML algorithms and analysis
- **Pandas**: Data manipulation and analysis
- **YData Profiling**: Statistical profiling
- **OpenAI API**: LLM integration for insights

### Frontend Architecture
- **React 18**: UI library with hooks
- **TypeScript**: Type-safe development
- **Axios**: HTTP client for API communication
- **Recharts**: Interactive data visualizations
- **Lucide React**: Icon library
- **Fiori CSS**: Enterprise design system

## Core Analysis Modules

The backend includes several specialized analysis modules:

1. **Data Profiler** - Statistical analysis of datasets
2. **Quality Scorer** - Multi-dimensional quality assessment
3. **Leakage Detector** - Identifies potential data leakage
4. **Consistency Checker** - Validates data integrity
5. **Model Recommender** - ML model selection
6. **Baseline Model Trainer** - Trains baseline models for comparison
7. **Non-Linear Scorer** - Advanced feature analysis
8. **LLM Validator** - AI-powered validation and insights

## Security Considerations

- CORS is configured for development (adjust in `app/core/config.py` for production)
- File size limits are enforced (default: 100MB)
- Only CSV and XLSX files are accepted
- API keys and sensitive data should be stored in `.env` files (not in version control)

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# LLM Configuration
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
LLM_MAX_TOKENS=2000
ENABLE_LLM=true

# Application Configuration
DEBUG=false
MAX_FILE_SIZE_MB=100
ALLOWED_EXTENSIONS=csv,xlsx

# Server Configuration
HOST=0.0.0.0
PORT=8000

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]

# API Configuration
API_V1_PREFIX=/api/v1
```

## Testing

### Backend Testing
```bash
cd backend
pytest
```

### Frontend Testing
```bash
cd frontend
npm test
```

## Supported File Formats

- **CSV**: UTF-8 encoded CSV files
- **XLSX**: Microsoft Excel spreadsheets

Maximum file size: 100MB (configurable)

## Contributing

Guidelines for contributing:
1. Follow the existing code structure
2. Use type hints (Python) and TypeScript for type safety
3. Document complex logic with comments
4. Test your changes before submitting
5. Follow PEP 8 (Python) and Prettier (JavaScript) formatting

## License

This project is provided as-is for internal use.

## Troubleshooting

### Backend Issues
- **Port already in use**: Change `PORT` in config or kill the existing process
- **OpenAI API errors**: Verify your API key in `.env`
- **Missing dependencies**: Run `pip install -r requirements.txt`

### Frontend Issues
- **API connection errors**: Ensure backend is running on `http://localhost:8000`
- **Module not found**: Run `npm install`
- **Port 3000 in use**: Run `npm start -- --port 3001`

## Support

For detailed information about the backend and frontend, see:
- [Backend README](./backend/README.md)
- [Frontend README](./frontend/README.md)
