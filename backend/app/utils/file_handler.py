import pandas as pd
import io
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from app.core.config import settings

class FileHandler:
    @staticmethod
    async def validate_file(file: UploadFile) -> None:
        """Validate uploaded file format and size"""
        
        # Check file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type '{file_extension}' not supported. Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File size {file_size_mb:.2f}MB exceeds maximum allowed size of {settings.max_file_size_mb}MB"
            )
        
        # Reset file pointer
        await file.seek(0)
    
    @staticmethod
    async def read_file(file: UploadFile) -> Tuple[pd.DataFrame, dict]:
        """Read uploaded file and return DataFrame with metadata"""
        
        await FileHandler.validate_file(file)
        
        try:
            file_content = await file.read()
            file_extension = file.filename.split('.')[-1].lower()
            
            # Read based on file type
            if file_extension == 'csv':
                df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
            elif file_extension == 'xlsx':
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
            
            # Validate DataFrame
            if df.empty:
                raise HTTPException(status_code=400, detail="File is empty")
            
            if len(df.columns) == 0:
                raise HTTPException(status_code=400, detail="No columns found in file")
            
            # Generate metadata
            metadata = {
                "filename": file.filename,
                "rows": len(df),
                "columns": len(df.columns),
                "size_mb": len(file_content) / (1024 * 1024),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict()
            }
            
            return df, metadata
            
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported. Please use UTF-8 encoding.")
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="File is empty or contains no data")
        except pd.errors.ParserError as e:
            raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame, suggested_target: Optional[str] = None) -> Optional[str]:
        """Auto-detect the most likely target column"""
        
        # If user provided target column, validate it exists
        if suggested_target:
            if suggested_target in df.columns:
                return suggested_target
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Target column '{suggested_target}' not found in dataset. Available columns: {', '.join(df.columns)}"
                )
        
        # Auto-detection heuristics
        target_candidates = []
        
        # Check for common target column names
        common_targets = ['target', 'label', 'y', 'class', 'output', 'outcome', 'result', 'prediction', 'churn', 'fraud', 'default']
        for col in df.columns:
            if col.lower() in common_targets:
                target_candidates.append((col, 10))  # High priority
        
        # Check last column (common ML convention)
        last_column = df.columns[-1]
        target_candidates.append((last_column, 5))  # Medium priority
        
        # Check for binary columns (likely classification targets)
        for col in df.columns:
            if df[col].dtype in ['bool']:
                target_candidates.append((col, 8))
            elif set(df[col].dropna().unique()) <= {0, 1} or set(df[col].dropna().unique()) <= {'yes', 'no'} or set(df[col].dropna().unique()) <= {True, False}:
                target_candidates.append((col, 7))
        
        # Return highest priority candidate
        if target_candidates:
            target_candidates.sort(key=lambda x: x[1], reverse=True)
            return target_candidates[0][0]
        
        return None
    
    @staticmethod
    def detect_problem_type(target_series: pd.Series) -> str:
        """Auto-detect if problem is classification or regression"""
        
        # Check data type
        if target_series.dtype in ['object', 'category', 'bool']:
            return "classification"
        
        # Check unique value ratio
        unique_ratio = len(target_series.unique()) / len(target_series)
        
        if unique_ratio < 0.05:  # Less than 5% unique values
            return "classification"
        
        # Check for integer values with small range
        if target_series.dtype in ['int64', 'int32']:
            unique_values = target_series.nunique()
            if unique_values <= 20:
                return "classification"
        
        # Check for continuous float values
        if target_series.dtype in ['float64', 'float32']:
            if unique_ratio > 0.1:
                return "regression"
        
        # Default to regression for continuous numerical data
        return "regression"