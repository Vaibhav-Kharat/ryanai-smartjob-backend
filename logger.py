import logging
import json
from datetime import datetime
from typing import Dict, Any

class GeminiLogger:
    def __init__(self, log_file: str = "gemini_api_logs.txt"):
        self.log_file = log_file
        self.logger = logging.getLogger("GeminiAPILogger")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_api_call(self, endpoint: str, request_data: Dict[str, Any], response_data: Dict[str, Any]):
        """Log API call details including token usage"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "request": request_data,
            "response": {
                "tokens_used": response_data.get("usage_metadata", {}).get("total_token_count", 0),
                "prompt_tokens": response_data.get("usage_metadata", {}).get("prompt_token_count", 0),
                "candidates_tokens": response_data.get("usage_metadata", {}).get("candidates_token_count", 0),
                "model": response_data.get("model", "unknown"),
                "timestamp": response_data.get("timestamp", "")
            }
        }
        
        self.logger.info(json.dumps(log_entry, indent=2))
    
    @staticmethod
    def extract_usage_metadata(response):
        """Extract usage metadata from the Gemini API response"""
        try:
            # Check if response has the expected structure
            if hasattr(response, 'result') and hasattr(response.result, 'usage_metadata'):
                metadata = response.result.usage_metadata
                return {
                    "prompt_token_count": metadata.prompt_token_count,
                    "candidates_token_count": metadata.candidates_token_count,
                    "total_token_count": metadata.total_token_count,
                }
            # Fallback for different response structure
            elif hasattr(response, 'usage_metadata'):
                metadata = response.usage_metadata
                return {
                    "prompt_token_count": metadata.prompt_token_count,
                    "candidates_token_count": metadata.candidates_token_count,
                    "total_token_count": metadata.total_token_count,
                }
            # If no usage metadata found
            return {}
        except Exception as e:
            print(f"Error extracting usage metadata: {e}")
            return {}

# Global logger instance
gemini_logger = GeminiLogger()