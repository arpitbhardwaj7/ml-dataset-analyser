import React from 'react';
import { Target, Database, AlertTriangle, Clock } from 'lucide-react';
import MetricCard from '../shared/MetricCard';
import { AnalysisResults } from '../../types';

interface QualityDashboardProps {
  results: AnalysisResults;
}

const QualityDashboard: React.FC<QualityDashboardProps> = ({ results }) => {
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

  const getGradeColor = (grade: string): string => {
    switch (grade.toUpperCase()) {
      case 'A': return 'var(--sap-success)';
      case 'B': return 'var(--sap-info)';
      case 'C': return 'var(--sap-warning)';
      case 'D': return 'var(--sap-warning)';
      case 'F': return 'var(--sap-danger)';
      default: return 'var(--sap-primary)';
    }
  };

  const getIssuesSeverity = (issues: any[]): { color: string; icon: string } => {
    const highIssues = issues.filter(i => i.severity === 'high').length;
    const mediumIssues = issues.filter(i => i.severity === 'medium').length;
    
    if (highIssues > 0) {
      return { color: 'var(--sap-danger)', icon: '' };
    } else if (mediumIssues > 0) {
      return { color: 'var(--sap-warning)', icon: '' };
    } else if (issues.length > 0) {
      return { color: 'var(--sap-success)', icon: '' };
    } else {
      return { color: 'var(--sap-success)', icon: '' };
    }
  };

  const issuesSeverity = getIssuesSeverity(results.detected_issues);

  return (
    <div>
      <h2 style={{ 
        marginBottom: '1rem', 
        color: 'var(--sap-text-primary)',
        fontSize: '1.25rem',
        fontWeight: 600 
      }}>
        Quality Assessment Dashboard
      </h2>
      
      <div className="grid grid-4" style={{ marginBottom: '2rem' }}>
        <MetricCard
          title="Quality Score"
          value={results.quality_score.overall}
          subtitle={`Grade ${results.quality_score.grade}`}
          color={getGradeColor(results.quality_score.grade)}
        />
        
        <MetricCard
          title="Dataset Size"
          value={formatNumber(results.dataset_info.rows)}
          subtitle={`${results.dataset_info.columns} columns`}
          color="var(--sap-info)"
        />
        
        <MetricCard
          title="Issues Detected"
          value={results.detected_issues.length}
          subtitle="Total issues"
          color={issuesSeverity.color}
        />
        
        <MetricCard
          title="Analysis Time"
          value={formatDuration(results.metadata.processing_time_seconds)}
          subtitle="Processing time"
          color="var(--sap-success)"
        />
      </div>
    </div>
  );
};

export default QualityDashboard;