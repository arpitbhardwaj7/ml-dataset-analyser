import React, { useState } from 'react';
import { Database, BarChart3, ChevronDown, ChevronUp, FileText, Target, Brain } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import FioriButton from '../shared/FioriButton';
import StatusTag from '../shared/StatusTag';
import { AnalysisResults } from '../../types';

interface DataProfileProps {
  results: AnalysisResults;
}

const DataProfile: React.FC<DataProfileProps> = ({ results }) => {
  const [showDetails, setShowDetails] = useState(false);
  const createProfileData = () => {
    const dataProfile = results.data_profile;
    const columnTypes = dataProfile.column_types;
    
    return [
      {
        metric: 'Missing Values %',
        value: `${dataProfile.missing_values_percentage.toFixed(1)}%`
      },
      {
        metric: 'Duplicate Rows',
        value: dataProfile.duplicate_rows.toString()
      },
      {
        metric: 'Numerical Features',
        value: columnTypes.numerical.toString()
      },
      {
        metric: 'Categorical Features',
        value: columnTypes.categorical.toString()
      },
      {
        metric: 'DateTime Features',
        value: columnTypes.datetime.toString()
      },
      {
        metric: 'Boolean Features',
        value: columnTypes.boolean.toString()
      },
      {
        metric: 'Memory Usage (MB)',
        value: `${dataProfile.memory_usage_mb.toFixed(1)} MB`
      }
    ];
  };

  const profileData = createProfileData();

  const getProblemTypeColor = (problemType: string) => {
    switch (problemType.toLowerCase()) {
      case 'classification':
        return 'var(--sap-status-positive)';
      case 'regression':
        return 'var(--sap-status-information)';
      default:
        return 'var(--sap-text-secondary)';
    }
  };

  const createDetailedProfileData = () => {
    const dataProfile = results.data_profile;
    const datasetInfo = results.dataset_info;
    return [
      {
        category: 'Dataset Overview',
        metrics: [
          { metric: 'Total Rows', value: datasetInfo.rows.toLocaleString() },
          { metric: 'Total Columns', value: datasetInfo.columns.toString() },
          { metric: 'Dataset Size', value: `${(dataProfile.memory_usage_mb / 1024).toFixed(2)} GB` }
        ]
      },
      {
        category: 'Data Quality',
        metrics: [
          { metric: 'Missing Values %', value: `${dataProfile.missing_values_percentage.toFixed(1)}%` },
          { metric: 'Duplicate Rows', value: dataProfile.duplicate_rows.toLocaleString() },
          { metric: 'Complete Rows', value: (datasetInfo.rows - dataProfile.duplicate_rows).toLocaleString() }
        ]
      },
      {
        category: 'Column Types Distribution',
        metrics: [
          { metric: 'Numerical Features', value: dataProfile.column_types.numerical.toString() },
          { metric: 'Categorical Features', value: dataProfile.column_types.categorical.toString() },
          { metric: 'DateTime Features', value: dataProfile.column_types.datetime.toString() },
          { metric: 'Boolean Features', value: dataProfile.column_types.boolean.toString() }
        ]
      }
    ];
  };

  return (
    <FioriCard 
      title="Data Profile" 
      headerAction={
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <FioriButton
            variant="ghost"
            size="small"
            icon={showDetails ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? 'Hide' : 'Show'} Details
          </FioriButton>
          <Database size={18} color="var(--sap-text-secondary)" />
        </div>
      }
    >
      {/* Dataset Info Header */}
      <div style={{ 
        marginBottom: '1.5rem',
        padding: '1rem',
        backgroundColor: 'var(--sap-background-subtle)',
        borderRadius: '8px',
        border: '1px solid var(--sap-border-light)'
      }}>
        <div style={{ 
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          {/* File Name */}
          <div style={{ 
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            flex: '1 1 auto',
            minWidth: '200px'
          }}>
            <div style={{
              padding: '0.5rem',
              backgroundColor: 'var(--sap-status-information)',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <FileText size={16} color="white" />
            </div>
            <div>
              <div style={{ 
                fontSize: '0.75rem',
                color: 'var(--sap-text-secondary)',
                fontWeight: 500,
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                Dataset File
              </div>
              <div style={{ 
                fontSize: '0.95rem',
                color: 'var(--sap-text-primary)',
                fontWeight: 600,
                fontFamily: 'var(--sap-font-mono)'
              }}>
                {results.dataset_info.filename}
              </div>
            </div>
          </div>

          {/* Target Column */}
          <div style={{ 
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            flex: '1 1 auto',
            minWidth: '180px'
          }}>
            <div style={{
              padding: '0.5rem',
              backgroundColor: 'var(--sap-status-positive)',
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Target size={16} color="white" />
            </div>
            <div>
              <div style={{ 
                fontSize: '0.75rem',
                color: 'var(--sap-text-secondary)',
                fontWeight: 500,
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                Target Column
              </div>
              <div style={{ 
                fontSize: '0.95rem',
                color: 'var(--sap-text-primary)',
                fontWeight: 600,
                fontFamily: 'var(--sap-font-mono)'
              }}>
                {results.dataset_info.detected_target_column || 'Auto-detected'}
              </div>
            </div>
          </div>

          {/* Problem Type */}
          <div style={{ 
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            flex: '0 1 auto',
            minWidth: '160px'
          }}>
            <div style={{
              padding: '0.5rem',
              backgroundColor: getProblemTypeColor(results.dataset_info.detected_problem_type),
              borderRadius: '6px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <Brain size={16} color="white" />
            </div>
            <div>
              <div style={{ 
                fontSize: '0.75rem',
                color: 'var(--sap-text-secondary)',
                fontWeight: 500,
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                Problem Type
              </div>
              <div style={{ 
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <span style={{ 
                  fontSize: '0.95rem',
                  color: 'var(--sap-text-primary)',
                  fontWeight: 600,
                  textTransform: 'capitalize'
                }}>
                  {results.dataset_info.detected_problem_type}
                </span>
                <div style={{
                  padding: '0.25rem 0.5rem',
                  backgroundColor: getProblemTypeColor(results.dataset_info.detected_problem_type),
                  color: 'white',
                  borderRadius: '12px',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  {results.dataset_info.detected_problem_type === 'classification' ? 'CLS' : 'REG'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="fiori-table-container">
        <table className="fiori-table">
          <thead>
            <tr>
              <th style={{ width: '60%' }}>Metric</th>
              <th style={{ width: '40%', textAlign: 'right' }}>Value</th>
            </tr>
          </thead>
          <tbody>
            {profileData.map((item, index) => (
              <tr key={index}>
                <td style={{ fontWeight: 500, color: 'var(--sap-text-primary)' }}>
                  {item.metric}
                </td>
                <td style={{ 
                  textAlign: 'right',
                  fontFamily: 'var(--sap-font-mono)',
                  color: 'var(--sap-text-secondary)',
                  fontWeight: 600
                }}>
                  {item.value}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {showDetails && (
        <div style={{ marginTop: '1.5rem' }}>
          {createDetailedProfileData().map((category, categoryIndex) => (
            <div key={categoryIndex} style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ 
                margin: '0 0 1rem 0',
                fontSize: '1rem',
                color: 'var(--sap-text-primary)',
                fontWeight: 600,
                paddingBottom: '0.5rem',
                borderBottom: '1px solid var(--sap-border-light)'
              }}>
                {category.category}
              </h4>
              
              <div className="fiori-table-container">
                <table className="fiori-table">
                  <thead>
                    <tr>
                      <th style={{ width: '60%' }}>Metric</th>
                      <th style={{ width: '40%', textAlign: 'right' }}>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {category.metrics.map((item, index) => (
                      <tr key={index}>
                        <td style={{ fontWeight: 500, color: 'var(--sap-text-primary)' }}>
                          {item.metric}
                        </td>
                        <td style={{ 
                          textAlign: 'right',
                          fontFamily: 'var(--sap-font-mono)',
                          color: 'var(--sap-text-secondary)',
                          fontWeight: 600
                        }}>
                          {item.value}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </FioriCard>
  );
};

export default DataProfile;