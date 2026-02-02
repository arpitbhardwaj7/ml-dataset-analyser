import React from 'react';

export interface FioriCardProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  className?: string;
  borderColor?: string;
  headerAction?: React.ReactNode;
}

const FioriCard: React.FC<FioriCardProps> = ({
  children,
  title,
  subtitle,
  className = '',
  borderColor,
  headerAction
}) => {
  const cardStyle = borderColor ? { borderLeftColor: borderColor, borderLeftWidth: '4px' } : undefined;

  return (
    <div className={`fiori-card ${className}`} style={cardStyle}>
      {(title || headerAction) && (
        <div className="fiori-card-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              {title && <h3>{title}</h3>}
              {subtitle && (
                <div style={{ 
                  fontSize: '0.875rem', 
                  color: 'var(--sap-text-secondary)', 
                  marginTop: '0.25rem' 
                }}>
                  {subtitle}
                </div>
              )}
            </div>
            {headerAction && (
              <div>{headerAction}</div>
            )}
          </div>
        </div>
      )}
      <div className="fiori-card-content">
        {children}
      </div>
    </div>
  );
};

export default FioriCard;