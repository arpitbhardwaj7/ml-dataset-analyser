# ML Dataset Analyser - Frontend

A modern, responsive React TypeScript frontend for ML Dataset Analyser, featuring an enterprise-grade SAP Fiori design system and interactive data visualizations.

## Quick Start

### Prerequisites
- Node.js 16 or higher
- npm 8 or higher
- Backend API running at `http://localhost:8000`

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

   The application will automatically open at `http://localhost:3000`

3. (Optional) **Configure backend URL** in `src/services/api.ts` if using a different backend location.

## Available Scripts

### Development

```bash
npm start
```

Runs the app in development mode with hot reload. Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

### Production Build

```bash
npm run build
```

Builds the app for production to the `build/` folder.
- Minified and optimized for best performance
- Build is ready to be deployed

### Testing

```bash
npm test
```

Launches the test runner in interactive watch mode.

## Project Structure

```
frontend/
├── public/
│   └── index.html                    # Root HTML file
├── src/
│   ├── App.tsx                       # Main application component
│   ├── index.tsx                     # React entry point
│   ├── components/
│   │   ├── upload/
│   │   │   ├── FileUploadSection.tsx # File drag-drop component
│   │   │   └── AnalysisConfig.tsx    # Configuration form
│   │   ├── results/
│   │   │   ├── QualityDashboard.tsx  # Main results dashboard
│   │   │   ├── DataProfile.tsx       # Data statistics view
│   │   │   ├── IssuesDetected.tsx    # Issues and warnings
│   │   │   ├── QualityBreakdown.tsx  # Quality score breakdown
│   │   │   ├── ExecutiveSummary.tsx  # High-level insights
│   │   │   └── ModelRecommendations.tsx # ML model recommendations
│   │   └── shared/
│   │       ├── FioriButton.tsx       # Styled button component
│   │       ├── FioriCard.tsx         # Card container component
│   │       ├── MetricCard.tsx        # Metric display component
│   │       └── StatusTag.tsx         # Status badge component
│   ├── services/
│   │   └── api.ts                    # API client (axios)
│   ├── styles/
│   │   └── fiori-theme.css           # SAP Fiori design system
│   └── types/
│       └── index.ts                  # TypeScript type definitions
├── package.json                      # Dependencies and scripts
├── tsconfig.json                     # TypeScript configuration
└── README.md                         # This file
```

## Design System

The frontend uses the **SAP Fiori Design System**, providing:
- Enterprise-grade visual consistency
- Accessibility compliance
- Responsive design for all devices
- Professional color palette and typography
- Intuitive component library

### Key Design Tokens
- **Primary Color**: `#0CBCE6` (SAP Blue)
- **Success Color**: `#107E3E`
- **Warning Color**: `#FBC02D`
- **Error Color**: `#E81828`
- **Font Family**: 72 (SAP's system font)

See `src/styles/fiori-theme.css` for complete styling.

## Key Components

### FileUploadSection
Drag-and-drop file upload interface with progress tracking.

**Features:**
- Drag-and-drop support
- File validation
- Progress bar during upload
- Error handling with user-friendly messages

```tsx
<FileUploadSection 
  onFileSelect={handleFile}
  uploading={isUploading}
  progress={uploadProgress}
/>
```

### AnalysisConfig
Configuration form for analysis parameters.

**Options:**
- Target column selection
- Problem type selection (auto/classification/regression)
- LLM insights toggle
- Auto-detect functionality

### QualityDashboard
Main results dashboard displaying all analysis outputs.

**Sections:**
- Dataset information
- Quality score overview
- Issues detected
- Model recommendations
- Data profile
- Executive summary

### DataProfile
Detailed data statistics and characteristics.

**Shows:**
- Column types distribution
- Missing values percentage
- Duplicate rows count
- Statistical summaries
- Feature analysis

### IssuesDetected
List of detected problems with severity levels.

**Includes:**
- Class imbalance warnings
- Data leakage flags
- Missing value alerts
- Outlier detection
- Consistency issues

### QualityBreakdown
Visual breakdown of quality scores across 5 dimensions.

**Dimensions:**
1. Completeness - Data coverage and missing values
2. Consistency - Data integrity and duplicates
3. Balance - Class distribution
4. Dimensionality - Feature relevance
5. Separability - Class distinguishability

**Visualization:**
- Radar chart showing all dimensions
- Detailed breakdown cards
- Grade indicators (A-F)

### ModelRecommendations
Top 3 recommended ML models with explanations.

**For Each Model:**
- Rank and reasoning
- Applicability to dataset
- Key advantages
- Implementation links

### ExecutiveSummary
High-level insights and recommendations.

**Includes:**
- Dataset quality assessment
- Key findings
- Risk assessment
- Prioritized next steps
- LLM-powered insights (when enabled)

## API Integration

### API Client (`src/services/api.ts`)

The application communicates with the backend via REST API:

```typescript
import { apiClient } from './services/api';

// Analyze a dataset
const response = await apiClient.post('/api/v1/analyze', formData, {
  headers: { 'Content-Type': 'multipart/form-data' }
});
```

### Default Configuration
- **Base URL**: `http://localhost:8000`
- **Timeout**: 300 seconds (for large file processing)
- **Content Type**: `multipart/form-data` for file uploads

### Changing API Endpoint

Edit `src/services/api.ts`:

```typescript
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 300000
});
```

Or set environment variable:
```bash
REACT_APP_API_URL=http://api.example.com npm start
```

## Data Visualizations

The application uses **Recharts** for interactive visualizations:

- **Radar Chart**: Quality score breakdown across 5 dimensions
- **Bar Charts**: Feature distributions, missing values
- **Progress Indicators**: Upload progress, analysis status
- **Status Cards**: Key metrics and indicators

All charts are responsive and mobile-friendly.

## Application Flow

### 1. Upload Phase
- User drags and drops (or selects) a CSV/XLSX file
- File validation occurs client-side
- Upload progress is displayed

### 2. Configuration Phase
- User optionally selects target column
- User optionally specifies problem type
- User optionally toggles LLM insights
- Form can auto-detect these values

### 3. Analysis Phase
- File is sent to backend API
- Real-time status updates shown to user
- Analysis engine processes the dataset

### 4. Results Display
- Quality score prominently displayed
- Issues and warnings highlighted
- Model recommendations presented
- Data profile statistics shown
- Executive summary provided

### 5. Interaction
- Users can view detailed breakdowns
- Explore individual issue details
- Review preprocessing recommendations
- Access LLM-generated insights

## Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Build Configuration
REACT_APP_ENV=development
```

Available environment variables:
- `REACT_APP_API_URL` - Backend API base URL
- `REACT_APP_ENV` - Environment name (development/production)

### Build Configuration

Edit `tsconfig.json` for TypeScript settings:

```json
{
  "compilerOptions": {
    "target": "es2020",
    "lib": ["es2020", "dom", "dom.iterable"],
    "jsx": "react-jsx",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  }
}
```

## Dependencies

### Core Dependencies

- **react** (18.2.0) - UI library
- **react-dom** (18.2.0) - React DOM binding
- **typescript** (4.9.5) - Type safety
- **react-router-dom** (6.20.1) - Routing (if needed)

### UI & Visualization

- **recharts** (2.8.0) - Data visualizations
- **lucide-react** (0.298.0) - Icon library
- **axios** (1.6.2) - HTTP client

### Development

- **react-scripts** (5.0.1) - Create React App CLI
- **@types/react** (18.2.45) - React type definitions
- **@types/react-dom** (18.2.18) - React DOM types

See `package.json` for complete list.

## Styling

### Global Styles (`src/styles/fiori-theme.css`)

Contains:
- SAP Fiori color palette
- Typography system
- Spacing utilities
- Component styles
- Responsive breakpoints

### CSS Customization

All components use CSS classes from `fiori-theme.css`. To customize:

1. Edit `src/styles/fiori-theme.css`
2. Update color variables
3. Adjust spacing and sizing
4. Rebuild: `npm run build`

## Accessibility

The application follows WCAG 2.1 guidelines:

- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Color contrast compliance
- Focus indicators
- Screen reader support

## Responsive Design

Breakpoints:
- **Mobile**: < 576px
- **Tablet**: 576px - 992px
- **Desktop**: > 992px

All components are fully responsive and mobile-friendly.

## Deployment

### Build for Production

```bash
npm run build
```

Creates an optimized production build in the `build/` folder.

### Deployment Targets

#### Static Hosting (Netlify, Vercel, GitHub Pages)

```bash
npm run build
# Upload contents of 'build/' folder
```

#### Docker

Create a `Dockerfile`:

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:

```bash
docker build -t ml-analyser-frontend .
docker run -p 80:80 ml-analyser-frontend
```

#### Traditional Web Server

Copy the `build/` folder contents to your web server's root directory (Apache, Nginx, etc.).

## Security

### Best Practices

1. **CORS**: Backend must allow requests from your frontend URL
2. **API Keys**: Never expose API keys in frontend code
3. **HTTPS**: Always use HTTPS in production
4. **Input Validation**: All file uploads are validated
5. **Sensitive Data**: Never log sensitive information to console

### Environment Variables

Sensitive data should be in `.env` and NOT committed to version control:

```env
# .env (add to .gitignore)
REACT_APP_API_URL=https://api.example.com
```

## Troubleshooting

### Backend Connection Errors

**Error**: "Cannot reach backend at localhost:8000"

**Solutions:**
1. Ensure backend is running: `cd backend && python run.py`
2. Check backend is on port 8000
3. Update API URL in `src/services/api.ts` if using different port
4. Check CORS configuration in backend

### File Upload Not Working

**Error**: "Failed to upload file"

**Solutions:**
1. Verify file is CSV or XLSX
2. Check file size is under 100MB
3. Verify backend `/api/v1/analyze` endpoint exists
4. Check browser console for specific error

### Module Not Found

**Error**: `Cannot find module '...'`

**Solutions:**
1. Run `npm install` again
2. Delete `node_modules/` and `package-lock.json`, then run `npm install`
3. Ensure all dependencies in `package.json` are installed

### Port 3000 Already in Use

```bash
# Use a different port
npm start -- --port 3001
```

### TypeScript Errors

```bash
# Check TypeScript for errors
npx tsc --noEmit

# Fix formatting issues
npx prettier --write src/
```

## Code Examples

### Custom API Call

```typescript
import axios from 'axios';

const analyzeDataset = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('use_llm_insights', 'true');

  try {
    const response = await axios.post('/api/v1/analyze', formData, {
      baseURL: 'http://localhost:8000',
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  } catch (error) {
    console.error('Analysis failed:', error);
    throw error;
  }
};
```

### Custom Component

```typescript
import React from 'react';
import { FioriCard } from '../components/shared/FioriCard';

const MyComponent: React.FC = () => {
  return (
    <FioriCard title="My Analysis Results">
      <p>Quality Score: 85.5</p>
    </FioriCard>
  );
};

export default MyComponent;
```

## Related Documentation

- [Main Project README](../README.md)
- [Backend API Documentation](../backend/README.md)
- [SAP Fiori Design Guidelines](https://experience.sap.com/fiori-design-web/)
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)

## Support

For backend API issues, see [Backend README](../backend/README.md).
For general information, see [Main README](../README.md).

## License

This project is provided as-is for internal use.
