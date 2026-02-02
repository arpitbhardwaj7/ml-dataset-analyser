import React from 'react';
import { FileText, Brain, AlertCircle, TrendingUp } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import StatusTag from '../shared/StatusTag';
import { LLMInsights } from '../../types';

interface ExecutiveSummaryProps {
  llmInsights?: LLMInsights;
  detectedIssues: any[];
}

const ExecutiveSummary: React.FC<ExecutiveSummaryProps> = ({ llmInsights, detectedIssues }) => {
  const createFallbackSummary = (issues: any[]) => {
    const highIssues = issues.filter(i => i.severity === 'high').length;
    const mediumIssues = issues.filter(i => i.severity === 'medium').length;
    const lowIssues = issues.filter(i => i.severity === 'low').length;

    if (issues.length === 0) {
      return "Your dataset demonstrates excellent quality with no significant issues detected. The data appears well-structured and ready for machine learning applications.";
    }

    let summary = `Analysis has identified ${issues.length} potential issues in your dataset. `;
    
    if (highIssues > 0) {
      summary += `${highIssues} high-priority issues require immediate attention. `;
    }
    if (mediumIssues > 0) {
      summary += `${mediumIssues} medium-priority issues should be addressed for optimal results. `;
    }
    if (lowIssues > 0) {
      summary += `${lowIssues} low-priority issues may benefit from minor improvements. `;
    }

    return summary + "Review the detailed recommendations below to enhance your dataset quality.";
  };

  const createActionItems = (issues: any[]) => {
    return issues
      .slice(0, 5)
      .map((issue, index) => ({
        action: issue.recommendation || `Address ${issue.title}`,
        impact: issue.severity.toUpperCase() as 'HIGH' | 'MEDIUM' | 'LOW',
        effort: 'Unknown effort'
      }));
  };

  const getRiskColor = (risk: string): string => {
    switch (risk.toLowerCase()) {
      case 'high': return 'var(--sap-danger)';
      case 'medium': return 'var(--sap-warning)';
      case 'low': return 'var(--sap-success)';
      default: return 'var(--sap-info)';
    }
  };

  const executiveSummary = llmInsights?.executive_summary || createFallbackSummary(detectedIssues);
  const actionItems = llmInsights?.top_action_items || createActionItems(detectedIssues);
  const riskAssessment = llmInsights?.risk_assessment;

  return (
    <div>
      <FioriCard 
        title="Executive Summary & Action Items"
        headerAction={<Brain size={18} color="var(--sap-primary)" />}
        borderColor="var(--sap-primary)"
      >
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 1.2fr',
          gap: '2rem'
        }}>
          {/* AI Analysis Summary */}
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1rem'
            }}>
              <FileText size={18} color="var(--sap-primary)" />
              <h4 style={{ 
                margin: 0, 
                fontSize: '1rem',
                color: 'var(--sap-text-primary)'
              }}>
                AI Analysis Summary
              </h4>
            </div>
            
            <div style={{
              padding: '1.25rem',
              background: 'var(--sap-info-bg)',
              border: '1px solid var(--sap-info)',
              borderRadius: '8px',
              marginBottom: '1.5rem'
            }}>
              <p style={{
                margin: 0,
                fontSize: '0.875rem',
                lineHeight: 1.6,
                color: 'var(--sap-text-primary)'
              }}>
                {executiveSummary}
              </p>
            </div>

            {/* Risk Assessment */}
            {riskAssessment && (
              <div>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  marginBottom: '1rem'
                }}>
                  <AlertCircle size={18} color="var(--sap-warning)" />
                  <h4 style={{ 
                    margin: 0, 
                    fontSize: '1rem',
                    color: 'var(--sap-text-primary)'
                  }}>
                    Risk Assessment
                  </h4>
                </div>
                
                <div className="fiori-table-container">
                  <table className="fiori-table">
                    <thead>
                      <tr>
                        <th>Risk Type</th>
                        <th style={{ width: '120px', textAlign: 'center' }}>Level</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>Overfitting Risk</td>
                        <td style={{ textAlign: 'center' }}>
                          <StatusTag 
                            status={riskAssessment.overfitting_risk} 
                            severity={riskAssessment.overfitting_risk.toLowerCase() === 'high' ? 'error' : 
                                     riskAssessment.overfitting_risk.toLowerCase() === 'medium' ? 'warning' : 'success'} 
                          />
                        </td>
                      </tr>
                      <tr>
                        <td>Underfitting Risk</td>
                        <td style={{ textAlign: 'center' }}>
                          <StatusTag 
                            status={riskAssessment.underfitting_risk}
                            severity={riskAssessment.underfitting_risk.toLowerCase() === 'high' ? 'error' : 
                                     riskAssessment.underfitting_risk.toLowerCase() === 'medium' ? 'warning' : 'success'}
                          />
                        </td>
                      </tr>
                      <tr>
                        <td>Data Leakage Risk</td>
                        <td style={{ textAlign: 'center' }}>
                          <StatusTag 
                            status={riskAssessment.data_leakage_risk}
                            severity={riskAssessment.data_leakage_risk.toLowerCase() === 'high' ? 'error' : 
                                     riskAssessment.data_leakage_risk.toLowerCase() === 'medium' ? 'warning' : 'success'}
                          />
                        </td>
                      </tr>
                      <tr>
                        <td>Curse of Dimensionality</td>
                        <td style={{ textAlign: 'center' }}>
                          <StatusTag 
                            status={riskAssessment.curse_of_dimensionality}
                            severity={riskAssessment.curse_of_dimensionality.toLowerCase() === 'high' ? 'error' : 
                                     riskAssessment.curse_of_dimensionality.toLowerCase() === 'medium' ? 'warning' : 'success'}
                          />
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Top Action Items */}
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1rem'
            }}>
              <TrendingUp size={18} color="var(--sap-primary)" />
              <h4 style={{ 
                margin: 0, 
                fontSize: '1rem',
                color: 'var(--sap-text-primary)'
              }}>
                Top Action Items
              </h4>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {actionItems.length > 0 ? (
                actionItems.slice(0, 5).map((item, index) => {
                  const priorityClass = item.impact === 'HIGH' ? 'high-priority' : 
                                       item.impact === 'MEDIUM' ? 'medium-priority' : 'low-priority';
                  const priorityColor = item.impact === 'HIGH' ? 'var(--sap-danger)' :
                                       item.impact === 'MEDIUM' ? 'var(--sap-warning)' : 'var(--sap-success)';

                  return (
                    <div
                      key={index}
                      style={{
                        padding: '1rem',
                        background: 'var(--sap-white)',
                        border: '1px solid var(--sap-border)',
                        borderLeft: `4px solid ${priorityColor}`,
                        borderRadius: '8px'
                      }}
                    >
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        marginBottom: '0.5rem'
                      }}>
                        <span style={{
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          color: priorityColor
                        }}>
                          #{index + 1}
                        </span>
                        <StatusTag 
                          status={`${item.impact} Impact`} 
                          severity={item.impact === 'HIGH' ? 'error' : 
                                   item.impact === 'MEDIUM' ? 'warning' : 'success'} 
                        />
                      </div>
                      <div style={{
                        fontSize: '0.875rem',
                        color: 'var(--sap-text-primary)',
                        lineHeight: 1.4,
                        fontWeight: 500
                      }}>
                        {item.action}
                      </div>
                      {item.effort && (
                        <div style={{
                          fontSize: '0.75rem',
                          color: 'var(--sap-text-muted)',
                          marginTop: '0.25rem'
                        }}>
                          Effort: {item.effort}
                        </div>
                      )}
                    </div>
                  );
                })
              ) : (
                <div style={{
                  textAlign: 'center',
                  padding: '2rem',
                  color: 'var(--sap-text-secondary)'
                }}>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>✨</div>
                  <div style={{ fontSize: '0.875rem' }}>
                    No specific action items identified. Your dataset appears to be in good condition!
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </FioriCard>
    </div>
  );
};

export default ExecutiveSummary;