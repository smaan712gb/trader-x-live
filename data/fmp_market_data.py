"""
Financial Modeling Prep Market Data Provider
Provides real-time and historical stock data using FMP API
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

class FMPMarketDataProvider:
    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY', 'demo')  # Use demo key if not provided
        self.base_url = 'https://financialmodelingprep.com/api/v3'
        
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote data"""
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                quote = data[0]
                return {
                    'symbol': quote.get('symbol'),
                    'price': quote.get('price'),
                    'change': quote.get('change'),
                    'changesPercentage': quote.get('changesPercentage'),
                    'dayLow': quote.get('dayLow'),
                    'dayHigh': quote.get('dayHigh'),
                    'yearHigh': quote.get('yearHigh'),
                    'yearLow': quote.get('yearLow'),
                    'marketCap': quote.get('marketCap'),
                    'priceAvg50': quote.get('priceAvg50'),
                    'priceAvg200': quote.get('priceAvg200'),
                    'volume': quote.get('volume'),
                    'avgVolume': quote.get('avgVolume'),
                    'timestamp': quote.get('timestamp')
                }
            else:
                return {'error': 'No data returned from API'}
                
        except Exception as e:
            return {'error': f'Failed to get real-time quote: {str(e)}'}
    
    def get_historical_data(self, symbol: str, period: str = '1year') -> pd.DataFrame:
        """Get historical price data"""
        try:
            # Map period to FMP format
            period_map = {
                '1d': '1day',
                '5d': '5day', 
                '1mo': '1month',
                '3mo': '3month',
                '6mo': '6month',
                '1y': '1year',
                '2y': '2year',
                '5y': '5year'
            }
            
            fmp_period = period_map.get(period, '1year')
            
            url = f"{self.base_url}/historical-price-full/{symbol}"
            params = {
                'apikey': self.api_key,
                'timeseries': 252 if fmp_period == '1year' else 100  # Adjust based on period
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'historical' in data and data['historical']:
                historical = data['historical']
                
                # Convert to DataFrame
                df = pd.DataFrame(historical)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Rename columns to match yfinance format
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High', 
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Select only the columns we need
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_intraday_data(self, symbol: str, interval: str = '1hour') -> pd.DataFrame:
        """Get intraday data"""
        try:
            # Map interval to FMP format
            interval_map = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1hour',
                '4h': '4hour'
            }
            
            fmp_interval = interval_map.get(interval, '1hour')
            
            url = f"{self.base_url}/historical-chart/{fmp_interval}/{symbol}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Rename columns to match yfinance format
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low', 
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Select only the columns we need
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting intraday data: {str(e)}")
            return pd.DataFrame()
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        try:
            url = f"{self.base_url}/profile/{symbol}"
            params = {'apikey': self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data and len(data) > 0:
                return data[0]
            else:
                return {}
                
        except Exception as e:
            return {'error': f'Failed to get company profile: {str(e)}'}
    
    def get_technical_indicator(self, symbol: str, indicator_type: str, 
                              timeframe: str = '1day', period: int = 14) -> Dict[str, Any]:
        """Get technical indicators from FMP API"""
        try:
            # Map timeframes
            timeframe_map = {
                '1m': '1min',
                '5m': '5min', 
                '15m': '15min',
                '30m': '30min',
                '1h': '1hour',
                '4h': '4hour',
                '1d': '1day',
                'daily': '1day'
            }
            
            fmp_timeframe = timeframe_map.get(timeframe, '1day')
            
            url = f"{self.base_url}/technical_indicator/{fmp_timeframe}/{symbol}"
            params = {
                'apikey': self.api_key,
                'type': indicator_type.lower(),
                'period': period
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                # Convert to DataFrame for easier handling
                df = pd.DataFrame(data)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                
                return {
                    'indicator': indicator_type,
                    'timeframe': timeframe,
                    'period': period,
                    'data': df,
                    'current_value': df.iloc[-1][indicator_type.lower()] if len(df) > 0 and indicator_type.lower() in df.columns else None,
                    'timestamp': df.index[-1] if len(df) > 0 else None
                }
            else:
                return {'error': f'No {indicator_type} data available'}
                
        except Exception as e:
            return {'error': f'Failed to get {indicator_type}: {str(e)}'}
    
    def get_multiple_indicators(self, symbol: str, timeframe: str = '1day') -> Dict[str, Any]:
        """Get multiple technical indicators at once"""
        indicators = {
            'sma_20': self.get_technical_indicator(symbol, 'sma', timeframe, 20),
            'sma_50': self.get_technical_indicator(symbol, 'sma', timeframe, 50),
            'ema_12': self.get_technical_indicator(symbol, 'ema', timeframe, 12),
            'ema_26': self.get_technical_indicator(symbol, 'ema', timeframe, 26),
            'rsi': self.get_technical_indicator(symbol, 'rsi', timeframe, 14),
            'adx': self.get_technical_indicator(symbol, 'adx', timeframe, 14),
            'williams': self.get_technical_indicator(symbol, 'williams', timeframe, 14)
        }
        
        # Extract current values
        current_values = {}
        for name, indicator_data in indicators.items():
            if 'error' not in indicator_data and indicator_data.get('current_value') is not None:
                current_values[name] = indicator_data['current_value']
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'indicators': indicators,
            'current_values': current_values,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_comprehensive_technical_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive technical analysis data using FMP indicators"""
        try:
            # Get real-time quote
            quote = self.get_real_time_quote(symbol)
            if 'error' in quote:
                return quote
            
            # Get historical data
            daily_data = self.get_historical_data(symbol, '3mo')
            hourly_data = self.get_intraday_data(symbol, '1hour')
            
            # Get technical indicators for different timeframes
            daily_indicators = self.get_multiple_indicators(symbol, '1day')
            hourly_indicators = self.get_multiple_indicators(symbol, '1hour')
            
            return {
                'symbol': symbol,
                'quote': quote,
                'historical_data': {
                    'daily': daily_data,
                    'hourly': hourly_data
                },
                'technical_indicators': {
                    'daily': daily_indicators,
                    'hourly': hourly_indicators
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to get comprehensive data: {str(e)}'}

    def test_connection(self) -> bool:
        """Test if the API connection is working"""
        try:
            quote = self.get_real_time_quote('AAPL')
            return 'error' not in quote
        except:
            return False

# Global instance
fmp_provider = FMPMarketDataProvider()
