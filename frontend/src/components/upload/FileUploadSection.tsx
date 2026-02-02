import React, { useCallback, useState } from 'react';
import { Upload, FileText, X } from 'lucide-react';
import FioriCard from '../shared/FioriCard';
import FioriButton from '../shared/FioriButton';

export interface FileUploadSectionProps {
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  disabled?: boolean;
  acceptedTypes?: string[];
  maxSize?: number; // in bytes
}

const FileUploadSection: React.FC<FileUploadSectionProps> = ({
  onFileSelect,
  selectedFile,
  disabled = false,
  acceptedTypes = ['.csv'],
  maxSize = 100 * 1024 * 1024 // 100MB default
}) => {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateFile = useCallback((file: File): string | null => {
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!acceptedTypes.includes(fileExtension)) {
      return `File type not supported. Please upload: ${acceptedTypes.join(', ')}`;
    }

    // Check file size
    if (file.size > maxSize) {
      return `File too large. Maximum size: ${Math.round(maxSize / 1024 / 1024)}MB`;
    }

    return null;
  }, [acceptedTypes, maxSize]);

  const handleFileSelect = useCallback((file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError(null);
    onFileSelect(file);
  }, [validateFile, onFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) {
      setDragOver(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    if (disabled) return;

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [disabled, handleFileSelect]);

  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  }, [handleFileSelect]);

  const handleRemoveFile = useCallback(() => {
    setError(null);
    onFileSelect(null);
  }, [onFileSelect]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <FioriCard title="Select Dataset">
      {!selectedFile ? (
        <div
          className={`file-upload-area ${dragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <Upload size={48} style={{ color: 'var(--sap-text-secondary)', marginBottom: '1rem' }} />
          <h3 style={{ margin: '0 0 0.5rem 0', color: 'var(--sap-text-primary)' }}>
            Upload your CSV dataset
          </h3>
          <p style={{ margin: '0 0 1rem 0', color: 'var(--sap-text-secondary)' }}>
            Drag and drop your file here, or click to browse
          </p>
          <FioriButton
            variant="primary"
            disabled={disabled}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            Browse Files
          </FioriButton>
          <input
            id="file-input"
            type="file"
            accept={acceptedTypes.join(',')}
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
            disabled={disabled}
          />
          <div style={{ 
            marginTop: '1rem', 
            fontSize: '0.75rem', 
            color: 'var(--sap-text-muted)',
            textAlign: 'center'
          }}>
            Supported formats: {acceptedTypes.join(', ')} • Max size: {Math.round(maxSize / 1024 / 1024)}MB
          </div>
        </div>
      ) : (
        <div style={{ padding: '1rem' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '1rem',
            padding: '1rem',
            background: 'var(--sap-success-bg)',
            border: '1px solid var(--sap-success)',
            borderRadius: '8px'
          }}>
            <FileText size={24} color="var(--sap-success)" />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, color: 'var(--sap-text-primary)' }}>
                {selectedFile.name}
              </div>
              <div style={{ fontSize: '0.875rem', color: 'var(--sap-text-secondary)' }}>
                {formatFileSize(selectedFile.size)} • {selectedFile.type || 'CSV File'}
              </div>
            </div>
            <FioriButton
              variant="ghost"
              size="small"
              icon={<X size={16} />}
              onClick={handleRemoveFile}
              disabled={disabled}
            >
              Remove
            </FioriButton>
          </div>
        </div>
      )}

      {error && (
        <div style={{ 
          marginTop: '1rem',
          padding: '0.75rem 1rem',
          background: 'var(--sap-danger-bg)',
          border: '1px solid var(--sap-danger)',
          borderRadius: '4px',
          color: 'var(--sap-danger)',
          fontSize: '0.875rem'
        }}>
          {error}
        </div>
      )}
    </FioriCard>
  );
};

export default FileUploadSection;