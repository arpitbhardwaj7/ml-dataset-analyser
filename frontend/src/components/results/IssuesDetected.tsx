import React, { useState } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, Info } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import StatusTag from '../shared/StatusTag';
import { DetectedIssue } from '../../types';

interface IssuesDetectedProps {
  issues: DetectedIssue[];
}

interface IssueItemProps {
  issue: DetectedIssue;
  index: number;
}

const IssueItem: React.FC<IssueItemProps> = ({ issue, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high': return '';
      case 'medium': return '';
      case 'low': return '';
      default: return '';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'var(--sap-danger)';
      case 'medium': return 'var(--sap-warning)';
      case 'low': return 'var(--sap-success)';
      default: return 'var(--sap-info)';
    }
  };

  return (
    <div style={{
      border: '1px solid var(--sap-border)',
      borderLeft: `4px solid ${getSeverityColor(issue.severity)}`,
      borderRadius: '8px',
      marginBottom: '0.75rem',
      background: 'var(--sap-white)'
    }}>
      <div 
        style={{
          padding: '1rem',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          background: getSeverityColor(issue.severity),
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '12px',
          animation: issue.severity === 'high' ? 'pulse 2s infinite' : 'none'
        }}>
          {getSeverityIcon(issue.severity)}
        </div>
        
        <div style={{ flex: 1 }}>
          <div style={{
            fontWeight: 600,
            color: 'var(--sap-text-primary)',
            marginBottom: '0.25rem'
          }}>
            {issue.title}
          </div>
          <div style={{
            fontSize: '0.875rem',
            color: 'var(--sap-text-secondary)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <StatusTag status={issue.severity} />
            <span>•</span>
            <span>{issue.category}</span>
          </div>
        </div>
        
        {isExpanded ? (
          <ChevronUp size={18} color="var(--sap-text-secondary)" />
        ) : (
          <ChevronDown size={18} color="var(--sap-text-secondary)" />
        )}
      </div>

      {isExpanded && (
        <div style={{ padding: '0 1rem 1rem 4rem' }}>
          <div style={{
            padding: '0.75rem',
            background: 'var(--sap-light)',
            borderRadius: '4px',
            marginBottom: '0.75rem',
            fontSize: '0.875rem',
            color: 'var(--sap-text-primary)',
            lineHeight: 1.5
          }}>
            <strong>Description:</strong> {issue.description}
          </div>
          
          {issue.recommendation && (
            <div style={{
              padding: '0.75rem',
              background: 'var(--sap-info-bg)',
              border: '1px solid var(--sap-info)',
              borderRadius: '4px',
              marginBottom: '0.75rem'
            }}>
              <div style={{
                fontSize: '0.75rem',
                fontWeight: 700,
                color: 'var(--sap-info)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem'
              }}>
                <Info size={12} /> Recommended Action
              </div>
              <div style={{
                fontSize: '0.875rem',
                color: 'var(--sap-text-primary)',
                lineHeight: 1.4
              }}>
                {issue.recommendation}
              </div>
            </div>
          )}
          
          {issue.affected_columns && issue.affected_columns.length > 0 && (
            <div>
              <div style={{
                fontSize: '0.75rem',
                fontWeight: 600,
                color: 'var(--sap-text-primary)',
                marginBottom: '0.5rem'
              }}>
                Affected Columns:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                {issue.affected_columns.map((column, colIndex) => (
                  <span
                    key={colIndex}
                    style={{
                      padding: '0.25rem 0.5rem',
                      background: 'var(--sap-warning-bg)',
                      color: 'var(--sap-warning)',
                      fontSize: '0.7rem',
                      borderRadius: '12px',
                      border: '1px solid var(--sap-warning)'
                    }}
                  >
                    {column}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const IssuesDetected: React.FC<IssuesDetectedProps> = ({ issues }) => {
  const getIssuesSummary = () => {
    const high = issues.filter(i => i.severity === 'high').length;
    const medium = issues.filter(i => i.severity === 'medium').length;
    const low = issues.filter(i => i.severity === 'low').length;
    
    return { high, medium, low, total: issues.length };
  };

  const summary = getIssuesSummary();

  return (
    <FioriCard 
      title="Issues Detected"
      headerAction={
        <div style={{ 
          fontSize: '0.75rem', 
          color: 'var(--sap-text-secondary)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          {summary.high > 0 && <span style={{ color: 'var(--sap-danger)' }}>{summary.high} High</span>}
          {summary.medium > 0 && <span style={{ color: 'var(--sap-warning)' }}>{summary.medium} Medium</span>}
          {summary.low > 0 && <span style={{ color: 'var(--sap-success)' }}>{summary.low} Low</span>}
        </div>
      }
    >
      {issues.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '2rem',
          color: 'var(--sap-text-secondary)'
        }}>
          <div style={{
            fontSize: '48px',
            marginBottom: '1rem',
            color: 'var(--sap-success)'
          }}>✓</div>
          <h3 style={{ 
            margin: '0 0 0.5rem 0', 
            color: 'var(--sap-success)',
            fontSize: '1.125rem' 
          }}>
            No Issues Detected
          </h3>
          <p style={{ 
            margin: 0, 
            fontSize: '0.875rem' 
          }}>
            Your dataset appears to be in excellent condition!
          </p>
        </div>
      ) : (
        <div>
          {issues.map((issue, index) => (
            <IssueItem key={issue.id || index} issue={issue} index={index} />
          ))}
        </div>
      )}
    </FioriCard>
  );
};

export default IssuesDetected;