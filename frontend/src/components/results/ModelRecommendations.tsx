import React from 'react';
import { Target, Star, TrendingUp } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import StatusTag from '../shared/StatusTag';
import { ModelRecommendation } from '../../types';

interface ModelRecommendationsProps {
  recommendations: ModelRecommendation[];
}

const ModelRecommendations: React.FC<ModelRecommendationsProps> = ({ recommendations }) => {
  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 80) return 'var(--sap-success)';
    if (confidence >= 60) return 'var(--sap-warning)';
    return 'var(--sap-danger)';
  };

  const getConfidenceSeverity = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 80) return 'success';
    if (confidence >= 60) return 'warning';
    return 'error';
  };

  return (
    <FioriCard 
      title="Model Recommendations"
      headerAction={
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Target size={18} color="var(--sap-text-secondary)" />
          <span style={{ fontSize: '0.75rem', color: 'var(--sap-text-secondary)' }}>
            Top {Math.min(recommendations.length, 5)} Recommendations
          </span>
        </div>
      }
    >
      {recommendations.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '3rem 2rem',
          color: 'var(--sap-text-secondary)'
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem', color: 'var(--sap-info)' }}>AI</div>
          <h3 style={{ margin: '0 0 0.5rem 0', color: 'var(--sap-text-primary)' }}>
            No Model Recommendations Available
          </h3>
          <p style={{ margin: 0, fontSize: '0.875rem' }}>
            Upload a dataset and run analysis to receive AI-powered model recommendations.
          </p>
        </div>
      ) : (
        <div>
          <div style={{
            padding: '1rem',
            background: 'var(--sap-light)',
            borderRadius: '8px',
            marginBottom: '1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <div style={{ 
                fontWeight: 600, 
                color: 'var(--sap-text-primary)',
                fontSize: '1rem'
              }}>
                {recommendations.length} Model{recommendations.length !== 1 ? 's' : ''} Recommended
              </div>
              <div style={{ 
                fontSize: '0.875rem', 
                color: 'var(--sap-text-secondary)' 
              }}>
                Based on your dataset characteristics and problem type
              </div>
            </div>
            <TrendingUp size={24} color="var(--sap-primary)" />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {recommendations.slice(0, 5).map((model, index) => (
              <div
                key={index}
                style={{
                  border: '1px solid var(--sap-border)',
                  borderLeft: `4px solid ${index === 0 ? 'var(--sap-primary)' : 'var(--sap-info)'}`,
                  borderRadius: '8px',
                  padding: '1.25rem',
                  background: 'var(--sap-white)',
                  position: 'relative'
                }}
              >
                {/* Rank Badge */}
                <div style={{
                  position: 'absolute',
                  top: '-8px',
                  left: '16px',
                  background: index === 0 ? 'var(--sap-primary)' : 'var(--sap-info)',
                  color: 'white',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '12px',
                  fontSize: '0.75rem',
                  fontWeight: 600
                }}>
                  #{index + 1}
                </div>

                <div style={{ marginTop: '0.5rem' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    marginBottom: '0.75rem'
                  }}>
                    <div>
                      <h4 style={{
                        margin: '0 0 0.5rem 0',
                        fontSize: '1.125rem',
                        fontWeight: 600,
                        color: 'var(--sap-text-primary)'
                      }}>
                        {model.model_name}
                      </h4>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.75rem'
                      }}>
                        <StatusTag 
                          status={`${model.confidence_score}% Confidence`} 
                          severity={getConfidenceSeverity(model.confidence_score)}
                        />
                      </div>
                    </div>

                    <div style={{
                      textAlign: 'right',
                      minWidth: '80px'
                    }}>
                      <div style={{
                        fontSize: '1.5rem',
                        fontWeight: 700,
                        color: getConfidenceColor(model.confidence_score)
                      }}>
                        {model.confidence_score}
                      </div>
                      <div style={{
                        fontSize: '0.75rem',
                        color: 'var(--sap-text-muted)'
                      }}>
                        Confidence
                      </div>
                    </div>
                  </div>


                  {/* Model Characteristics */}
                  {(model.pros || model.cons) && (
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '1rem'
                    }}>
                      {model.pros && (
                        <div>
                          <div style={{
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            color: 'var(--sap-success)',
                            marginBottom: '0.5rem',
                            textTransform: 'uppercase'
                          }}>
                            ✓ Pros
                          </div>
                          <ul style={{
                            margin: 0,
                            paddingLeft: '1rem',
                            fontSize: '0.75rem',
                            color: 'var(--sap-text-secondary)',
                            lineHeight: 1.4
                          }}>
                            {model.pros.map((pro, proIndex) => (
                              <li key={proIndex} style={{ marginBottom: '0.25rem' }}>
                                {pro}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {model.cons && (
                        <div>
                          <div style={{
                            fontSize: '0.75rem',
                            fontWeight: 600,
                            color: 'var(--sap-warning)',
                            marginBottom: '0.5rem',
                            textTransform: 'uppercase'
                          }}>
                            ! Cons
                          </div>
                          <ul style={{
                            margin: 0,
                            paddingLeft: '1rem',
                            fontSize: '0.75rem',
                            color: 'var(--sap-text-secondary)',
                            lineHeight: 1.4
                          }}>
                            {model.cons.map((con, conIndex) => (
                              <li key={conIndex} style={{ marginBottom: '0.25rem' }}>
                                {con}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {recommendations.length > 5 && (
            <div style={{
              textAlign: 'center',
              marginTop: '1rem',
              padding: '0.75rem',
              background: 'var(--sap-light)',
              borderRadius: '6px',
              fontSize: '0.875rem',
              color: 'var(--sap-text-secondary)'
            }}>
              Showing top 5 of {recommendations.length} recommendations
            </div>
          )}
        </div>
      )}
    </FioriCard>
  );
};

export default ModelRecommendations;