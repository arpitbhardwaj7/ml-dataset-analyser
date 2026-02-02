import React from 'react';

export interface FioriButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'small' | 'medium' | 'large';
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  className?: string;
  type?: 'button' | 'submit' | 'reset';
}

const FioriButton: React.FC<FioriButtonProps> = ({
  children,
  variant = 'secondary',
  size = 'medium',
  icon,
  iconPosition = 'left',
  disabled = false,
  loading = false,
  onClick,
  className = '',
  type = 'button'
}) => {
  const baseClass = 'fiori-btn';
  const variantClass = variant === 'primary' ? 'fiori-btn-primary' : '';
  const sizeClass = size !== 'medium' ? `fiori-btn-${size}` : '';
  const buttonClass = `${baseClass} ${variantClass} ${sizeClass} ${className}`.trim();

  const handleClick = () => {
    if (!disabled && !loading && onClick) {
      onClick();
    }
  };

  return (
    <button
      type={type}
      className={buttonClass}
      onClick={handleClick}
      disabled={disabled || loading}
    >
      {loading ? (
        <>
          <div className="loading-spinner" style={{ width: '16px', height: '16px' }}></div>
          <span>Loading...</span>
        </>
      ) : (
        <>
          {icon && iconPosition === 'left' && (
            <span className="btn-icon">{icon}</span>
          )}
          <span>{children}</span>
          {icon && iconPosition === 'right' && (
            <span className="btn-icon">{icon}</span>
          )}
        </>
      )}
    </button>
  );
};

export default FioriButton;