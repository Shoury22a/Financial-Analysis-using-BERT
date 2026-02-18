"""
Centralized API Clients for FINSIGHT AI
Manages connections to external services (Finnhub, YFinance, etc.)
"""
import os
import finnhub
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class FinnhubClient:
    """Wrapper for Finnhub API client"""
    
    _instance: Optional[finnhub.Client] = None
    
    @classmethod
    def get_client(cls) -> finnhub.Client:
        """
        Get Finnhub client instance (singleton pattern)
        API key is loaded from environment variable
        """
        if cls._instance is None:
            api_key = os.getenv('FINNHUB_API_KEY')
            
            if not api_key:
                raise ValueError(
                    "FINNHUB_API_KEY not found in environment variables. "
                    "Please create a .env file with your API key. "
                    "See .env.example for template."
                )
            
            cls._instance = finnhub.Client(api_key=api_key)
        
        return cls._instance


class YFinanceSession:
    """Wrapper for yfinance with custom session"""
    
    _session: Optional[requests.Session] = None
    
    @classmethod
    def get_session(cls) -> requests.Session:
        """
        Get requests session with proper headers for yfinance
        Helps avoid blocking issues on cloud platforms
        """
        if cls._session is None:
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
            })
            cls._session = session
        
        return cls._session


# Convenience functions
def get_finnhub_client() -> finnhub.Client:
    """Get Finnhub API client"""
    return FinnhubClient.get_client()


def get_yfinance_session() -> requests.Session:
    """Get YFinance session"""
    return YFinanceSession.get_session()
