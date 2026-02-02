import React from 'react';
import { Target, TrendingUp } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import { QualityScore } from '../../types';

interface QualityBreakdownProps {
  qualityScore: QualityScore;
}

const QualityBreakdown: React.FC<QualityBreakdownProps> = ({ qualityScore }) => {
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

  const createQualityGauge = (score: number, grade: string) => {
    const radius = 120;
    const strokeWidth = 16;
    const normalizedRadius = radius - strokeWidth * 2;
    const circumference = normalizedRadius * 2 * Math.PI;
    const strokeDasharray = `${circumference} ${circumference}`;
    const strokeDashoffset = circumference - (score / 100) * circumference;
    const color = getGradeColor(grade);

    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        position: 'relative',
        width: '240px',
        height: '240px',
        margin: '0 auto'
      }}>
        <svg
          height={radius * 2}
          width={radius * 2}
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Background circle */}
          <circle
            stroke="var(--sap-border)"
            fill="transparent"
            strokeWidth={strokeWidth}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
          {/* Progress circle */}
          <circle
            stroke={color}
            fill="transparent"
            strokeWidth={strokeWidth}
            strokeDasharray={strokeDasharray}
            style={{ 
              strokeDashoffset,
              transition: 'stroke-dashoffset 1s ease-in-out',
              strokeLinecap: 'round'
            }}
            r={normalizedRadius}
            cx={radius}
            cy={radius}
          />
        </svg>
        
        {/* Center content */}
        <div style={{
          position: 'absolute',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{
            fontSize: '2rem',
            fontWeight: 700,
            color: color
          }}>
            {score}
          </div>
          <div style={{
            fontSize: '1.25rem',
            fontWeight: 600,
            color: color
          }}>
            {grade}
          </div>
        </div>
      </div>
    );
  };

  const createDimensionBars = (breakdown: any) => {
    return Object.entries(breakdown).map(([dimension, data]: [string, any]) => {
      const score = data.score || 0;
      const grade = data.grade || 'F';
      const description = data.description || 'No description available';
      const color = getGradeColor(grade);

      return (
        <div key={dimension} style={{ marginBottom: '1rem' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '0.5rem'
          }}>
            <span style={{
              fontSize: '0.875rem',
              fontWeight: 500,
              color: 'var(--sap-text-primary)'
            }}>
              {dimension.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{
                fontSize: '0.875rem',
                fontWeight: 600,
                color: color
              }}>
                {score}
              </span>
              <span style={{
                fontSize: '1rem',
                fontWeight: 600,
                color: color,
                minWidth: '20px',
                textAlign: 'center'
              }}>
                {grade}
              </span>
            </div>
          </div>
          
          <div className="progress-container">
            <div 
              className="progress-bar"
              style={{ 
                width: `${score}%`,
                backgroundColor: color,
                transition: 'width 1s ease-in-out'
              }}
            />
          </div>
        </div>
      );
    });
  };

  const createDimensionsTable = (breakdown: any) => {
    const dimensionsData = Object.entries(breakdown).map(([dimension, data]: [string, any]) => ({
      dimension: dimension.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      score: data.score || 0,
      grade: data.grade || 'F',
      description: data.description || 'No description available'
    }));

    return (
      <div className="fiori-table-container">
        <table className="fiori-table">
          <thead>
            <tr>
              <th>Dimension</th>
              <th style={{ width: '80px', textAlign: 'center' }}>Score</th>
              <th style={{ width: '60px', textAlign: 'center' }}>Grade</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {dimensionsData.map((item, index) => (
              <tr key={index}>
                <td style={{ fontWeight: 500 }}>{item.dimension}</td>
                <td style={{ 
                  textAlign: 'center',
                  fontFamily: 'var(--sap-font-mono)',
                  fontWeight: 600,
                  color: getGradeColor(item.grade)
                }}>
                  {item.score}
                </td>
                <td style={{ 
                  textAlign: 'center',
                  fontWeight: 600,
                  color: getGradeColor(item.grade),
                  fontSize: '1rem'
                }}>
                  {item.grade}
                </td>
                <td style={{ 
                  fontSize: '0.875rem',
                  color: 'var(--sap-text-secondary)',
                  lineHeight: 1.4
                }}>
                  {item.description}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div>
      <FioriCard 
        title="Quality Score Breakdown"
        headerAction={<TrendingUp size={18} color="var(--sap-text-secondary)" />}
      >
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: '1fr 2fr',
          gap: '2rem',
          marginBottom: '2rem'
        }}>
          {/* Gauge Chart */}
          <div>
            {createQualityGauge(qualityScore.overall, qualityScore.grade)}
          </div>
          
          {/* Dimension Bars */}
          <div>
            <h4 style={{ 
              marginBottom: '1rem',
              fontSize: '1rem',
              color: 'var(--sap-text-primary)'
            }}>
              Quality Dimensions
            </h4>
            {qualityScore.breakdown && createDimensionBars(qualityScore.breakdown)}
          </div>
        </div>
      </FioriCard>

      {/* Detailed Dimensions Table */}
      {qualityScore.breakdown && (
        <FioriCard 
          title="Quality Dimensions Details"
          headerAction={<Target size={18} color="var(--sap-text-secondary)" />}
        >
          {createDimensionsTable(qualityScore.breakdown)}
        </FioriCard>
      )}
    </div>
  );
};

export default QualityBreakdown;