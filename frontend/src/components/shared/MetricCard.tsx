import React from 'react';
import { MetricCardData } from '../../types';

export interface MetricCardProps extends MetricCardData {
  onClick?: () => void;
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  color,
  icon,
  onClick,
  loading = false
}) => {
  const cardStyle = color ? { 
    borderTop: `4px solid ${color}` 
  } : undefined;

  const valueStyle = color ? { color } : undefined;

  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  return (
    <div 
      className={`metric-card ${onClick ? 'cursor-pointer' : ''}`}
      style={cardStyle}
      onClick={handleClick}
    >
      {loading ? (
        <div className="loading-spinner"></div>
      ) : (
        <>
          <div className="metric-title">
            {icon && <span style={{ marginRight: '0.5rem' }}>{icon}</span>}
            {title}
          </div>
          <div className="metric-value" style={valueStyle}>
            {value}
          </div>
          {subtitle && (
            <div className="metric-subtitle">
              {subtitle}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default MetricCard;