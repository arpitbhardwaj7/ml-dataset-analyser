import React, { useState, useEffect } from 'react';
import { Settings, Brain, Target } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import FioriButton from '../shared/FioriButton';
import { AnalysisRequest } from '../../types';

export interface AnalysisConfigProps {
  onConfigChange: (config: AnalysisRequest) => void;
  selectedFile: File | null;
  disabled?: boolean;
  columns?: string[];
}

const AnalysisConfig: React.FC<AnalysisConfigProps> = ({
  onConfigChange,
  selectedFile,
  disabled = false,
  columns = []
}) => {
  const [config, setConfig] = useState<AnalysisRequest>({
    target_column: undefined,
    problem_type: 'auto',
    use_llm_insights: true
  });

  useEffect(() => {
    onConfigChange(config);
  }, [config, onConfigChange]);

  const handleTargetColumnChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value === '' ? undefined : e.target.value;
    setConfig(prev => ({ ...prev, target_column: value }));
  };

  const handleProblemTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setConfig(prev => ({ 
      ...prev, 
      problem_type: e.target.value as 'auto' | 'classification' | 'regression' 
    }));
  };

  const handleLLMToggle = (e: React.ChangeEvent<HTMLInputElement>) => {
    setConfig(prev => ({ ...prev, use_llm_insights: e.target.checked }));
  };

  const formatFileSize = (bytes: number): string => {
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <FioriCard 
      title="Analysis Configuration" 
      headerAction={
        <Settings size={18} color="var(--sap-text-secondary)" />
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Target Column Selection */}
        <div className="form-group">
          <label htmlFor="target-column" className="form-label">
            <Target size={16} style={{ marginRight: '0.5rem' }} />
            Target Column (Optional)
          </label>
          <select
            id="target-column"
            className="form-select"
            value={config.target_column || ''}
            onChange={handleTargetColumnChange}
            disabled={disabled}
          >
            <option value="">Auto-detect</option>
            {columns.map(column => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--sap-text-muted)', 
            marginTop: '0.25rem' 
          }}>
            Select the target variable for supervised learning analysis
          </div>
        </div>

        {/* Problem Type */}
        <div className="form-group">
          <label htmlFor="problem-type" className="form-label">
            Problem Type
          </label>
          <select
            id="problem-type"
            className="form-select"
            value={config.problem_type}
            onChange={handleProblemTypeChange}
            disabled={disabled}
          >
            <option value="auto">Auto-detect</option>
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--sap-text-muted)', 
            marginTop: '0.25rem' 
          }}>
            Specify the type of ML problem for targeted recommendations
          </div>
        </div>

        {/* LLM Insights Toggle */}
        <div className="form-group">
          <label style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.75rem',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 500
          }}>
            <input
              type="checkbox"
              checked={config.use_llm_insights}
              onChange={handleLLMToggle}
              disabled={disabled}
              style={{ width: '18px', height: '18px' }}
            />
            <Brain size={16} color="var(--sap-primary)" />
            Enhanced AI Insights
          </label>
          <div style={{ 
            fontSize: '0.75rem', 
            color: 'var(--sap-text-muted)', 
            marginTop: '0.5rem',
            marginLeft: '2.25rem'
          }}>
            Use advanced AI for deeper analysis, executive summaries, and actionable recommendations
          </div>
        </div>

        {/* File Preview */}
        {selectedFile && (
          <div style={{
            padding: '1rem',
            background: 'var(--sap-light)',
            borderRadius: '8px',
            border: '1px solid var(--sap-border)'
          }}>
            <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '0.5rem' }}>
              Dataset Preview
            </div>
            
            <div style={{ fontSize: '0.75rem', color: 'var(--sap-text-secondary)' }}>
              <strong>{selectedFile.name}</strong> • {formatFileSize(selectedFile.size)}
              {columns.length > 0 && (
                <span> • {columns.length} columns detected</span>
              )}
            </div>
          </div>
        )}
      </div>
    </FioriCard>
  );
};

export default AnalysisConfig;