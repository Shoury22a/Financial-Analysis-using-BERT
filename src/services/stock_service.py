"""
Stock Service for FINSIGHT AI
Dynamic stock lookup and data fetching - NO HARDCODED STOCK DATABASE
"""
import time
import pandas as pd
import yfinance as yf
from yahooquery import Ticker
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

from src.clients import get_finnhub_client, get_yfinance_session

logger = logging.getLogger(__name__)


class StockService:
    """
    Stock data service using dynamic API lookups
    NO hardcoded stock databases!
    """
    
    def __init__(self):
        self.finnhub_client = get_finnhub_client()
        self.yf_session = get_yfinance_session()
        self._cache = {}  # Simple in-memory cache
    
    def search_symbol(self, query: str) -> List[Dict[str, str]]:
        """
        Search for stock symbols using Finnhub API
        
        Args:
            query: Company name or ticker symbol
        
        Returns:
            List of matching stocks with symbol, name, type, etc.
        """
        if not query or len(query) < 2:
            return []
        
        # Check cache
        cache_key = f"search_{query.lower()}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for search: {query}")
            return self._cache[cache_key]
        
        try:
            # Use Finnhub symbol lookup API
            results = self.finnhub_client.symbol_lookup(query)
            
            matches = []
            for item in results.get('result', [])[:15]:  # Limit to 15 results
                matches.append({
                    'symbol': item.get('symbol', ''),
                    'description': item.get('description', ''),
                    'type': item.get('type', ''),
                    'displaySymbol': item.get('displaySymbol', '')
                })
            
            # Cache results
            self._cache[cache_key] = matches
            logger.info(f"Found {len(matches)} results for '{query}'")
            
            return matches
        
        except Exception as e:
            logger.error(f"Symbol search failed for '{query}': {e}")
            return []
    
    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information using Finnhub API
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dict with company info (name, sector, industry, etc.)
        """
        cache_key = f"info_{symbol}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            
            if profile:
                info = {
                    'symbol': symbol,
                    'name': profile.get('name', symbol),
                    'country': profile.get('country', 'Unknown'),
                    'sector': profile.get('finnhubIndustry', 'Unknown'),
                    'currency': profile.get('currency', 'USD'),
                    'marketCap': profile.get('marketCapitalization'),
                    'ipo': profile.get('ipo'),
                    'logo': profile.get('logo'),
                    'weburl': profile.get('weburl')
                }
                
                self._cache[cache_key] = info
                return info
            
        except Exception as e:
            logger.warning(f"Failed to get company info for {symbol}: {e}")
        
        return {'symbol': symbol, 'name': symbol, 'sector': 'Unknown'}
    
    def get_stock_data(self, symbol: str, period: str = "1mo") -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Fetch stock OHLCV data using multiple strategies
        
        Strategy 1: Finnhub API (most reliable)
        Strategy 2: YahooQuery
        Strategy 3: yfinance
        
        Args:
            symbol: Stock ticker
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y)
        
        Returns:
            (DataFrame with OHLCV data, company info dict)
        """
        logger.info(f"Fetching data for {symbol} (period: {period})")
        
        # Strategy 1: Finnhub API
        hist, info = self._fetch_from_finnhub(symbol, period)
        if hist is not None and not hist.empty:
            return hist, info
        
        # Strategy 2: YahooQuery
        hist, info = self._fetch_from_yahooquery(symbol, period)
        if hist is not None and not hist.empty:
            return hist, info
        
        # Strategy 3: yfinance
        hist, info = self._fetch_from_yfinance(symbol, period)
        if hist is not None and not hist.empty:
            return hist, info
        
        logger.error(f"All data fetch strategies failed for {symbol}")
        return None, {}
    
    def _fetch_from_finnhub(self, symbol: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Fetch from Finnhub API"""
        try:
            # Map period to days
            period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "5y": 1825}
            days = period_map.get(period, 30)
            
            end_ts = int(time.time())
            start_ts = end_ts - (days * 24 * 60 * 60)
            
            # Fetch candle data
            res = self.finnhub_client.stock_candles(symbol, 'D', start_ts, end_ts)
            
            if res.get('s') == 'ok':
                df = pd.DataFrame({
                    'Open': res['o'],
                    'High': res['h'],
                    'Low': res['l'],
                    'Close': res['c'],
                    'Volume': res['v'],
                }, index=pd.to_datetime(res['t'], unit='s'))
                
                info = self.get_company_info(symbol)
                logger.info(f"Successfully fetched {len(df)} candles from Finnhub for {symbol}")
                return df, info
        
        except Exception as e:
            logger.debug(f"Finnhub fetch failed for {symbol}: {e}")
        
        return None, {}
    
    def _fetch_from_yahooquery(self, symbol: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Fetch from YahooQuery"""
        try:
            ticker = Ticker(symbol, session=self.yf_session, retry=2)
            hist = ticker.history(period=period)
            
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                info = {}
                try:
                    quotes = ticker.quotes.get(symbol, {})
                    info['name'] = quotes.get('longName', symbol)
                    info['sector'] = quotes.get('sector', 'Unknown')
                except:
                    info = {'name': symbol}
                
                logger.info(f"Successfully fetched {len(hist)} bars from YahooQuery for {symbol}")
                return hist, info
        
        except Exception as e:
            logger.debug(f"YahooQuery fetch failed for {symbol}: {e}")
        
        return None, {}
    
    def _fetch_from_yfinance(self, symbol: str, period: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """Fetch from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if not hist.empty:
                info = {}
                try:
                    stock_info = stock.info
                    info['name'] = stock_info.get('longName', symbol)
                    info['sector'] = stock_info.get('sector', 'Unknown')
                except:
                    info = {'name': symbol}
                
                logger.info(f"Successfully fetched {len(hist)} bars from yfinance for {symbol}")
                return hist, info
        
        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol}: {e}")
        
        return None, {}
