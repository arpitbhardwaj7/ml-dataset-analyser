import React from 'react';

export interface StatusTagProps {
  status: string;
  severity?: 'success' | 'warning' | 'error' | 'info';
  className?: string;
}

const StatusTag: React.FC<StatusTagProps> = ({ 
  status, 
  severity, 
  className = '' 
}) => {
  // Auto-determine severity based on common status values if not provided
  const getSeverity = (status: string): string => {
    if (severity) return severity;
    
    const lowerStatus = status.toLowerCase();
    
    if (lowerStatus.includes('high') || lowerStatus.includes('critical') || lowerStatus.includes('error') || lowerStatus === 'f') {
      return 'error';
    } else if (lowerStatus.includes('medium') || lowerStatus.includes('warning') || lowerStatus === 'd' || lowerStatus === 'c') {
      return 'warning';
    } else if (lowerStatus.includes('low') || lowerStatus.includes('success') || lowerStatus.includes('good') || lowerStatus === 'a' || lowerStatus === 'b') {
      return 'success';
    } else {
      return 'info';
    }
  };

  const statusSeverity = getSeverity(status);
  const statusClass = `status-tag status-${statusSeverity} ${className}`;

  return (
    <span className={statusClass}>
      {status}
    </span>
  );
};

export default StatusTag;