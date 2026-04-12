# ML Dataset Analyser — Frontend

React + TypeScript frontend for ML Dataset Analyser, styled with the SAP Fiori design system.

## Setup

```bash
npm install
npm start        # Dev server at http://localhost:3000
npm run build    # Production build to build/
npm test         # Run tests
```

Requires the backend running at `http://localhost:8000`. To use a different URL:

```bash
REACT_APP_API_URL=http://localhost:9000 npm start
```

Or set it permanently in a `.env` file in this directory:
```env
REACT_APP_API_URL=http://localhost:8000
```

## Application Flow

1. **Upload** — drag-and-drop or file picker (`FileUploadSection.tsx`)
2. **Configure** — select target column, problem type, LLM toggle (`AnalysisConfig.tsx`)
3. **Results** — quality dashboard with score, issues, model recommendations, data profile, and executive summary (`QualityDashboard.tsx` aggregates the result panels)

## Key Files

| File | Purpose |
|---|---|
| `src/services/api.ts` | Axios client — base URL, 300s timeout, multipart upload |
| `src/types/index.ts` | TypeScript types mirroring backend Pydantic response schemas |
| `src/styles/fiori-theme.css` | SAP Fiori design tokens — use these CSS variables for all styling |
| `src/App.tsx` | Top-level state and phase transitions (upload → configure → results) |

## Troubleshooting

| Problem | Fix |
|---|---|
| Cannot reach backend | Ensure `python run.py` is running on port 8000 |
| File upload fails | Check file is CSV or XLSX and under 100MB |
| Module not found | `npm install` or delete `node_modules/` and reinstall |
| Port 3000 in use | `npm start -- --port 3001` |
| TypeScript errors | `npx tsc --noEmit` to see full error list |
