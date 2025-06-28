"""
API data extractor for fetching data from REST APIs.
"""

import requests
from typing import Any, Dict, List, Optional
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseExtractor
from ..logger import get_logger


class APIExtractor(BaseExtractor):
    """
    Extractor for fetching data from REST APIs.
    
    Supports pagination, authentication, and retry logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize API extractor.
        
        Args:
            config: Configuration containing API settings
        """
        super().__init__(config)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {config.get("api_key")}',
            'Content-Type': 'application/json'
        })
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            requests.Response: API response
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.get("timeout", 30)
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error("API request failed", 
                            url=url, 
                            error=str(e),
                            attempt=self._make_request.retry.statistics.get('attempt_number', 0))
            raise
    
    def extract(self, endpoint: str = "/data", params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Extract data from API endpoint.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters for the API call
            
        Returns:
            pd.DataFrame: Extracted data
            
        Raises:
            Exception: If extraction fails
        """
        self.logger.info("Starting API data extraction", 
                        endpoint=endpoint,
                        base_url=self.config.get("base_url"))
        
        try:
            url = f"{self.config['base_url']}{endpoint}"
            all_data = []
            
            # Handle pagination if present
            page = 1
            while True:
                request_params = params or {}
                request_params['page'] = page
                
                response = self._make_request(url, request_params)
                data = response.json()
                
                # Check if data is in expected format
                if isinstance(data, dict):
                    if 'data' in data:
                        page_data = data['data']
                    elif 'results' in data:
                        page_data = data['results']
                    else:
                        page_data = data
                else:
                    page_data = data
                
                if not page_data:
                    break
                
                all_data.extend(page_data)
                
                # Check for pagination indicators
                if isinstance(data, dict):
                    if data.get('next') is None or not data.get('has_next', True):
                        break
                
                page += 1
                
                # Safety check to prevent infinite loops
                if page > 1000:
                    self.logger.warning("Pagination limit reached, stopping extraction")
                    break
            
            if not all_data:
                self.logger.warning("No data extracted from API")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_data)
            self.logger.info("API extraction completed", 
                           rows=len(df),
                           columns=list(df.columns))
            
            return df
            
        except Exception as e:
            self.logger.error("API extraction failed", 
                            endpoint=endpoint,
                            error=str(e))
            raise
    
    def close(self):
        """Close the session."""
        self.session.close() 