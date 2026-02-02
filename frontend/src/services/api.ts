import axios, { AxiosInstance, AxiosProgressEvent } from 'axios';
import { AnalysisResults, AnalysisRequest, APIResponse } from '../types';

class APIClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor() {
    this.baseURL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
    
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: 300000, // 5 minutes
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<APIResponse<{ status: string }>> {
    try {
      const response = await this.client.get('/health');
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Health check failed' 
      };
    }
  }

  /**
   * Analyze dataset endpoint
   */
  async analyzeDataset(
    file: File,
    config: AnalysisRequest,
    onProgress?: (progress: number) => void
  ): Promise<APIResponse<AnalysisResults>> {
    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);
      
      // Only append target_column if it's defined and not empty
      if (config.target_column && config.target_column.trim()) {
        formData.append('target_column', config.target_column.trim());
        console.log('Sending target_column:', config.target_column.trim()); // Debug log
      } else {
        console.log('No target_column provided - using auto-detection'); // Debug log
      }
      
      formData.append('problem_type', config.problem_type);
      formData.append('use_llm_insights', String(config.use_llm_insights));

      // Make the request with progress tracking
      const response = await this.client.post('/api/v1/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent: AxiosProgressEvent) => {
          if (onProgress && progressEvent.total) {
            const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            onProgress(progress);
          }
        },
      });

      return { success: true, data: response.data };
    } catch (error) {
      let errorMessage = 'Analysis failed';
      
      if (axios.isAxiosError(error)) {
        if (error.response?.data?.error) {
          errorMessage = error.response.data.error;
        } else if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail;
        } else if (error.response?.status === 413) {
          errorMessage = 'File too large. Please try with a smaller dataset.';
        } else if (error.response?.status && error.response.status >= 500) {
          errorMessage = 'Server error. Please try again later.';
        } else if (error.code === 'ECONNABORTED') {
          errorMessage = 'Request timeout. Please try with a smaller dataset.';
        } else {
          errorMessage = error.message;
        }
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }

      console.error('API Error Details:', error); // Debug log
      return { success: false, error: errorMessage };
    }
  }

  /**
   * Get analysis status (if implementing async processing)
   */
  async getAnalysisStatus(taskId: string): Promise<APIResponse<{ status: string; progress: number }>> {
    try {
      const response = await this.client.get(`/api/v1/analyze/status/${taskId}`);
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to get analysis status' 
      };
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export for testing
export { APIClient };