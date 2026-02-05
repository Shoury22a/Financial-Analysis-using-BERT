import pandas as pd
import os
import json
import requests
from typing import Dict, Tuple

class StockManager:
    """
    Manages fetching and caching of global stock lists from public sources (Wikipedia).
    """
    CACHE_FILE = "stocks_cache.json"
    
    @staticmethod
    def scrape_sp500() -> Dict[str, Tuple[str, str, str]]:
        """Scrape S&P 500 companies from Wikipedia"""
        print("Scraping S&P 500 from Wikipedia...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            df = tables[0]
            
            stocks = {}
            for _, row in df.iterrows():
                symbol = str(row['Symbol']).replace('.', '-')
                name = str(row['Security'])
                sector = str(row['GICS Sector'])
                stocks[symbol] = (name, 'US', sector)
            return stocks
        except Exception as e:
            print(f"Error scraping S&P 500: {e}")
            return {}

    @staticmethod
    def scrape_nifty50() -> Dict[str, Tuple[str, str, str]]:
        """Scrape Nifty 50 companies from Wikipedia"""
        print("Scraping Nifty 50 from Wikipedia...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            url = "https://en.wikipedia.org/wiki/NIFTY_50"
            response = requests.get(url, headers=headers)
            tables = pd.read_html(response.text)
            # Wikipedia structure can change, usually the first or second table
            df = None
            for table in tables:
                if 'Symbol' in table.columns and 'Company Name' in table.columns:
                    df = table
                    break
            
            if df is None:
                return {}
                
            stocks = {}
            for _, row in df.iterrows():
                symbol = str(row['Symbol']) + ".NS" # Yahoo Finance format
                name = str(row['Company Name'])
                sector = str(row['Sector'])
                stocks[symbol] = (name, 'India', sector)
            return stocks
        except Exception as e:
            print(f"Error scraping Nifty 50: {e}")
            return {}

    @classmethod
    def get_global_stocks(cls, force_refresh=False) -> Dict[str, Tuple[str, str, str]]:
        """
        Returns a dictionary of stocks. Uses cache if available unless force_refresh is True.
        Format: { Ticker: (Name, Region, Sector) }
        """
        if not force_refresh and os.path.exists(cls.CACHE_FILE):
            try:
                with open(cls.CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                # Convert back from list to tuple
                return {k: tuple(v) for k, v in cached_data.items()}
            except Exception:
                pass
        
        # Scrape data
        sp500 = cls.scrape_sp500()
        nifty50 = cls.scrape_nifty50()
        
        combined_stocks = {**sp500, **nifty50}
        
        # Save to cache
        if combined_stocks:
            try:
                with open(cls.CACHE_FILE, 'w') as f:
                    json.dump(combined_stocks, f)
            except Exception as e:
                print(f"Error caching stocks: {e}")
                
        return combined_stocks

if __name__ == "__main__":
    # Test the scraper
    manager = StockManager()
    stocks = manager.get_global_stocks(force_refresh=True)
    print(f"Successfully scraped {len(stocks)} stocks.")
    for i, (ticker, info) in enumerate(list(stocks.items())[:5]):
        print(f"{ticker}: {info}")
