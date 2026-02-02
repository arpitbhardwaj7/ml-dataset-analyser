import React, { useState, useCallback } from 'react';
import { ChevronLeft, Database, Play, CheckCircle2 } from 'lucide-react';
import './styles/fiori-theme.css';

// Import components
import FileUploadSection from './components/upload/FileUploadSection';
import AnalysisConfig from './components/upload/AnalysisConfig';
import FioriButton from './components/shared/FioriButton';

// Import results components
import QualityDashboard from './components/results/QualityDashboard';
import DataProfile from './components/results/DataProfile';
import IssuesDetected from './components/results/IssuesDetected';
import QualityBreakdown from './components/results/QualityBreakdown';
import ExecutiveSummary from './components/results/ExecutiveSummary';
import ModelRecommendations from './components/results/ModelRecommendations';

// Import types and services
import { AnalysisRequest, UploadState, AnalysisState } from './types';
import { apiClient } from './services/api';

const App: React.FC = () => {
  // Upload state
  const [uploadState, setUploadState] = useState<UploadState>({
    file: null,
    isDragging: false,
    isUploading: false,
    uploadProgress: 0
  });

  // Analysis state
  const [analysisState, setAnalysisState] = useState<AnalysisState>({
    isAnalyzing: false,
    progress: 0,
    currentStep: '',
    results: null,
    error: null
  });

  // Analysis configuration
  const [analysisConfig, setAnalysisConfig] = useState<AnalysisRequest>({
    target_column: undefined,
    problem_type: 'auto',
    use_llm_insights: true
  });

  // Columns extracted from CSV file
  const [columns, setColumns] = useState<string[]>([]);

  // Function to extract column headers from CSV file
  const extractColumnsFromCSV = useCallback((file: File): Promise<string[]> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const text = e.target?.result as string;
          const lines = text.split('\n');
          if (lines.length > 0) {
            // Split by comma and clean up column names
            let columnHeaders = lines[0]
              .split(',')
              .map(col => col.trim().replace(/^["']|["']$/g, '')); // Remove quotes if any
            
            // Filter out empty column names
            columnHeaders = columnHeaders.filter(col => col.length > 0);
            
            console.log('Extracted columns:', columnHeaders); // Debug log
            resolve(columnHeaders);
          } else {
            reject(new Error('CSV file is empty'));
          }
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = () => reject(new Error('Failed to read file'));
      reader.readAsText(file);
    });
  }, []);

  const handleFileSelect = useCallback((file: File | null) => {
    setUploadState(prev => ({ ...prev, file }));
    
    // Extract columns from the selected file
    if (file) {
      extractColumnsFromCSV(file)
        .then(cols => {
          setColumns(cols);
        })
        .catch(error => {
          console.error('Failed to extract columns:', error);
          setColumns([]);
        });
    } else {
      setColumns([]);
    }
    
    // Clear previous results when a new file is selected
    if (file !== uploadState.file) {
      setAnalysisState(prev => ({
        ...prev,
        results: null,
        error: null
      }));
    }
  }, [extractColumnsFromCSV, uploadState.file]);

  const handleAnalysisConfigChange = useCallback((config: AnalysisRequest) => {
    setAnalysisConfig(config);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!uploadState.file) return;

    setAnalysisState(prev => ({
      ...prev,
      isAnalyzing: true,
      progress: 0,
      currentStep: 'Preparing dataset...',
      error: null
    }));

    try {
      // Simulate progress steps
      const steps = [
        'Preparing dataset...',
        'Connecting to analysis service...',
        'Running quality analysis...',
        'Generating insights...'
      ];

      let currentStepIndex = 0;
      const progressInterval = setInterval(() => {
        currentStepIndex = Math.min(currentStepIndex + 1, steps.length - 1);
        setAnalysisState(prev => ({
          ...prev,
          progress: Math.min((currentStepIndex / steps.length) * 90, 90),
          currentStep: steps[currentStepIndex]
        }));
      }, 1500);

      const response = await apiClient.analyzeDataset(
        uploadState.file,
        analysisConfig,
        (progress) => {
          setAnalysisState(prev => ({
            ...prev,
            progress: Math.max(prev.progress, progress)
          }));
        }
      );

      clearInterval(progressInterval);

      if (response.success && response.data) {
        setAnalysisState(prev => ({
          ...prev,
          isAnalyzing: false,
          progress: 100,
          currentStep: 'Analysis complete!',
          results: response.data!
        }));
      } else {
        throw new Error(response.error || 'Analysis failed');
      }
    } catch (error) {
      setAnalysisState(prev => ({
        ...prev,
        isAnalyzing: false,
        progress: 0,
        currentStep: '',
        error: error instanceof Error ? error.message : 'Analysis failed'
      }));
    }
  }, [uploadState.file, analysisConfig]);

  const handleStartOver = useCallback(() => {
    setUploadState({
      file: null,
      isDragging: false,
      isUploading: false,
      uploadProgress: 0
    });
    setAnalysisState({
      isAnalyzing: false,
      progress: 0,
      currentStep: '',
      results: null,
      error: null
    });
  }, []);

  const formatNumber = (num: number, decimals: number = 0): string => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(decimals) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(decimals) + 'K';
    } else {
      return num.toFixed(decimals);
    }
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds.toFixed(1)}s`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="object-page-header" style={{ padding: '1rem 2rem 0.75rem' }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '1rem',
          marginBottom: '1.5rem' 
        }}>
          {analysisState.results && (
            <button
              onClick={handleStartOver}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: 'var(--sap-primary)',
                padding: '0.5rem'
              }}
            >
              <ChevronLeft size={24} />
            </button>
          )}
          <div>
            <div className="breadcrumb">
              Dataset Analysis
            </div>
            <h1>
              ML Dataset Analyzer
            </h1>
          </div>
          <div style={{ flex: 1 }} />
          {analysisState.results && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <FioriButton variant="primary">Export Report</FioriButton>
              <FioriButton>Share Results</FioriButton>
            </div>
          )}
        </div>
      </div>

      <div className="page-container object-page-content">
        {!analysisState.results ? (
          // Upload and Configuration Phase
          <div className="grid grid-2" style={{ gap: '2rem' }}>
            <div>
              <FileUploadSection
                onFileSelect={handleFileSelect}
                selectedFile={uploadState.file}
                disabled={analysisState.isAnalyzing}
              />
            </div>
            <div>
              <AnalysisConfig
                onConfigChange={handleAnalysisConfigChange}
                selectedFile={uploadState.file}
                disabled={analysisState.isAnalyzing}
                columns={columns}
              />
            </div>
          </div>
        ) : (
          // Comprehensive Results Display Phase - Full Width Layout like Streamlit
          <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {/* Quality Assessment Dashboard */}
            <QualityDashboard results={analysisState.results} />

            {/* Data Profile - Full Width */}
            <DataProfile results={analysisState.results} />

            {/* Issues Detected - Full Width */}
            <IssuesDetected issues={analysisState.results.detected_issues} />

            {/* Quality Score Breakdown - Full Width */}
            <QualityBreakdown qualityScore={analysisState.results.quality_score} />

            {/* Executive Summary & Action Items - Full Width */}
            <ExecutiveSummary 
              llmInsights={analysisState.results.llm_insights}
              detectedIssues={analysisState.results.detected_issues}
            />

            {/* Model Recommendations - Full Width */}
            <ModelRecommendations recommendations={analysisState.results.model_recommendations} />
          </div>
        )}

        {/* Analysis Progress */}
        {analysisState.isAnalyzing && (
          <div style={{
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'var(--sap-white)',
            padding: '2rem',
            borderRadius: '8px',
            boxShadow: 'var(--sap-shadow)',
            border: '1px solid var(--sap-border)',
            minWidth: '300px',
            textAlign: 'center',
            zIndex: 1000
          }}>
            <div className="loading-spinner" style={{ margin: '0 auto 1rem', width: '32px', height: '32px' }}></div>
            <h3 style={{ margin: '0 0 0.5rem 0' }}>Analyzing Dataset</h3>
            <p style={{ margin: '0 0 1rem 0', color: 'var(--sap-text-secondary)' }}>
              {analysisState.currentStep}
            </p>
            <div className="progress-container" style={{ marginBottom: '0.5rem' }}>
              <div 
                className="progress-bar progress-success" 
                style={{ width: `${analysisState.progress}%` }}
              ></div>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--sap-text-muted)' }}>
              {analysisState.progress}% complete
            </div>
          </div>
        )}

        {/* Error Display */}
        {analysisState.error && (
          <div style={{
            padding: '1rem',
            background: 'var(--sap-danger-bg)',
            border: '1px solid var(--sap-danger)',
            borderRadius: '8px',
            marginTop: '1rem',
            color: 'var(--sap-danger)'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0' }}>Analysis Failed</h4>
            <p style={{ margin: 0 }}>{analysisState.error}</p>
          </div>
        )}

        {/* Analyze Button */}
        {uploadState.file && !analysisState.results && !analysisState.isAnalyzing && (
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <FioriButton
              variant="primary"
              size="large"
              icon={<Play size={18} />}
              onClick={handleAnalyze}
              loading={analysisState.isAnalyzing}
            >
              Analyze Dataset
            </FioriButton>
          </div>
        )}

        {analysisState.results && (
          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <FioriButton
              variant="secondary"
              icon={<CheckCircle2 size={18} />}
              onClick={handleStartOver}
            >
              Analyze Another Dataset
            </FioriButton>
          </div>
        )}
      </div>

      {/* Overlay for analysis progress */}
      {analysisState.isAnalyzing && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.3)',
          zIndex: 999
        }} />
      )}
    </div>
  );
};

export default App;