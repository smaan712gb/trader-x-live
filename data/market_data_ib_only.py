"""
IB-Only Market Data Manager
Provides real-time market data exclusively from IB Gateway
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas_ta as ta
from config.trading_config import TradingConfig, DatabaseConfig
from config.supabase_config import supabase_manager
from core.logger import logger
from data.ib_gateway import ib_gateway
import json
import os
import time

class IBOnlyMarketDataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        self.ib_connected = False
        
        # Initialize IB Gateway connection
        self._initialize_ib_gateway()
    
    def _initialize_ib_gateway(self):
        """Initialize IB Gateway connection"""
        try:
            # Check if IB Gateway settings are configured
            ib_host = os.getenv('IBKR_HOST', '127.0.0.1')
            ib_port = int(os.getenv('IBKR_PORT', '4002'))  # Paper trading port
            ib_client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
            
            # Configure IB Gateway
            ib_gateway.host = ib_host
            ib_gateway.port = ib_port
            ib_gateway.client_id = ib_client_id
            
            # Attempt connection
            if ib_gateway.connect():
                self.ib_connected = True
                logger.info("IB Gateway connected successfully - using real-time data", "MARKET_DATA")
            else:
                logger.error("IB Gateway connection failed - no market data available", "MARKET_DATA")
                self.ib_connected = False
                
        except Exception as e:
            logger.error(f"IB Gateway initialization failed: {e}", "MARKET_DATA")
            self.ib_connected = False
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get stock price data from IB Gateway only"""
        try:
            if not self.ib_connected:
                logger.error(f"IB Gateway not connected - cannot fetch data for {symbol}", "MARKET_DATA")
                return pd.DataFrame()
            
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first (refresh every 5 minutes for intraday data)
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                cache_timeout = timedelta(minutes=5) if interval in ['1m', '5m', '15m', '30m'] else timedelta(minutes=30)
                if datetime.now() - last_update < cache_timeout:
                    return self.cache[cache_key]
            
            # Convert period and interval to IB format
            ib_duration, ib_bar_size = self._convert_to_ib_format(period, interval)
            
            data = ib_gateway.get_historical_data(symbol, ib_duration, ib_bar_size)
            
            if not data.empty:
                # Cache the data
                self.cache[cache_key] = data
                self.last_update[cache_key] = datetime.now()
                
                logger.debug(f"Fetched {len(data)} rows from IB Gateway for {symbol}", "MARKET_DATA")
                return data
            else:
                logger.warning(f"No data from IB Gateway for {symbol}", "MARKET_DATA")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}", "MARKET_DATA")
            return pd.DataFrame()
    
    def _convert_to_ib_format(self, period: str, interval: str) -> Tuple[str, str]:
        """Convert yfinance period/interval to IB Gateway format"""
        # Convert period
        period_map = {
            "1d": "1 D",
            "5d": "5 D", 
            "1mo": "1 M",
            "3mo": "3 M",
            "6mo": "6 M",
            "1y": "1 Y",
            "2y": "2 Y",
            "5y": "5 Y",
            "10y": "10 Y",
            "ytd": "1 Y",
            "max": "10 Y"
        }
        
        # Convert interval
        interval_map = {
            "1m": "1 min",
            "2m": "2 mins",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "60m": "1 hour",
            "90m": "1 hour",
            "1h": "1 hour",
            "1d": "1 day",
            "5d": "1 week",
            "1wk": "1 week",
            "1mo": "1 month",
            "3mo": "1 month"
        }
        
        ib_duration = period_map.get(period, "1 Y")
        ib_bar_size = interval_map.get(interval, "1 day")
        
        return ib_duration, ib_bar_size
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote data from IB Gateway"""
        try:
            if not self.ib_connected:
                logger.error(f"IB Gateway not connected - cannot fetch quote for {symbol}", "MARKET_DATA")
                return {}
            
            # Get real-time data from IB Gateway
            market_data = ib_gateway.get_market_data(symbol)
            
            if market_data:
                return {
                    'symbol': symbol,
                    'last': market_data.get('last', 0),
                    'bid': market_data.get('bid', 0),
                    'ask': market_data.get('ask', 0),
                    'volume': market_data.get('volume', 0),
                    'high': market_data.get('high', 0),
                    'low': market_data.get('low', 0),
                    'close': market_data.get('close', 0),
                    'timestamp': market_data.get('timestamp', datetime.now()),
                    'source': 'IB_GATEWAY'
                }
            else:
                logger.warning(f"No real-time quote available for {symbol}", "MARKET_DATA")
                return {}
            
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a stock from IB Gateway"""
        try:
            if not self.ib_connected:
                logger.warning(f"IB Gateway not connected - cannot fetch fundamental data for {symbol}", "MARKET_DATA")
                return {}
            
            # Try IB Gateway fundamental data
            ib_fundamental = ib_gateway.get_fundamental_data(symbol, timeout=5)
            if ib_fundamental and 'xml_data' in ib_fundamental:
                # Parse IB fundamental data (simplified)
                fundamental_data = self._parse_ib_fundamental_data(symbol, ib_fundamental)
                if fundamental_data:
                    return fundamental_data
            
            logger.warning(f"No fundamental data available for {symbol} from IB Gateway", "MARKET_DATA")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to fetch fundamental data for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def _parse_ib_fundamental_data(self, symbol: str, ib_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse IB Gateway fundamental data (simplified implementation)"""
        try:
            # This would need proper XML parsing for production
            # For now, return basic structure with IB data
            return {
                'symbol': symbol,
                'market_cap': 0,  # Would need to parse from XML
                'revenue_ttm': 0,  # Would need to parse from XML
                'revenue_growth_yoy': 0,  # Would need to parse from XML
                'revenue_growth_qoq': 0,  # Would need to parse from XML
                'profit_margins': 0,  # Would need to parse from XML
                'operating_margins': 0,  # Would need to parse from XML
                'return_on_equity': 0,  # Would need to parse from XML
                'debt_to_equity': 0,  # Would need to parse from XML
                'current_ratio': 0,  # Would need to parse from XML
                'price_to_earnings': 0,  # Would need to parse from XML
                'price_to_sales': 0,  # Would need to parse from XML
                'enterprise_value': 0,  # Would need to parse from XML
                'forward_pe': 0,  # Would need to parse from XML
                'peg_ratio': 0,  # Would need to parse from XML
                'beta': 1.0,  # Would need to parse from XML
                'fifty_two_week_high': 0,  # Would need to parse from XML
                'fifty_two_week_low': 0,  # Would need to parse from XML
                'analyst_target_price': 0,  # Would need to parse from XML
                'recommendation': 0,  # Would need to parse from XML
                'ib_fundamental_data': ib_data.get('xml_data', ''),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'ib_gateway'
            }
        except Exception as e:
            logger.error(f"Failed to parse IB fundamental data for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def get_technical_indicators(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Calculate technical indicators for a stock using IB data"""
        try:
            if not self.ib_connected:
                logger.warning(f"IB Gateway not connected - cannot calculate technical indicators for {symbol}", "MARKET_DATA")
                return {}
            
            data = self.get_stock_data(symbol, period="6mo", interval=timeframe)
            
            if data.empty:
                logger.warning(f"No data available for technical analysis of {symbol}", "MARKET_DATA")
                return {}
            
            # Calculate technical indicators using pandas_ta
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            data.ta.sma(length=200, append=True)
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.macd(fast=12, slow=26, signal=9, append=True)
            data.ta.bbands(length=20, std=2, append=True)
            data.ta.stoch(k=14, d=3, append=True)
            data.ta.adx(length=14, append=True)
            data.ta.atr(length=14, append=True)
            data.ta.obv(append=True)
            
            # Get latest values
            latest = data.iloc[-1]
            
            technical_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': float(latest['Close']),
                'volume': int(latest['Volume']),
                'sma_20': float(latest.get('SMA_20', 0)),
                'sma_50': float(latest.get('SMA_50', 0)),
                'sma_200': float(latest.get('SMA_200', 0)),
                'ema_12': float(latest.get('EMA_12', 0)),
                'ema_26': float(latest.get('EMA_26', 0)),
                'rsi': float(latest.get('RSI_14', 0)),
                'macd': float(latest.get('MACD_12_26_9', 0)),
                'macd_signal': float(latest.get('MACDs_12_26_9', 0)),
                'macd_histogram': float(latest.get('MACDh_12_26_9', 0)),
                'bb_upper': float(latest.get('BBU_20_2.0', 0)),
                'bb_middle': float(latest.get('BBM_20_2.0', 0)),
                'bb_lower': float(latest.get('BBL_20_2.0', 0)),
                'stoch_k': float(latest.get('STOCHk_14_3_3', 0)),
                'stoch_d': float(latest.get('STOCHd_14_3_3', 0)),
                'adx': float(latest.get('ADX_14', 0)),
                'atr': float(latest.get('ATRr_14', 0)),
                'obv': float(latest.get('OBV', 0)),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'ib_gateway'
            }
            
            # Add trend analysis
            technical_data.update(self._analyze_trends(data))
            
            # Store in database
            self._store_technical_data(technical_data)
            
            return technical_data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends and patterns"""
        try:
            latest = data.iloc[-1]
            
            # Price position relative to moving averages
            price = latest['Close']
            sma_20 = latest.get('SMA_20', price)
            sma_50 = latest.get('SMA_50', price)
            sma_200 = latest.get('SMA_200', price)
            
            # Trend strength
            trend_strength = "NEUTRAL"
            if price > sma_20 > sma_50 > sma_200:
                trend_strength = "STRONG_BULLISH"
            elif price > sma_20 > sma_50:
                trend_strength = "BULLISH"
            elif price < sma_20 < sma_50 < sma_200:
                trend_strength = "STRONG_BEARISH"
            elif price < sma_20 < sma_50:
                trend_strength = "BEARISH"
            
            # Support and resistance levels
            high_20 = data['High'].rolling(20).max().iloc[-1]
            low_20 = data['Low'].rolling(20).min().iloc[-1]
            
            # Volume analysis
            avg_volume_20 = data['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = latest['Volume'] / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            # Volume trend
            volume_trend = "neutral"
            if len(data) >= 5:
                recent_volumes = data['Volume'].tail(5)
                if recent_volumes.iloc[-1] > recent_volumes.mean():
                    volume_trend = "increasing"
                elif recent_volumes.iloc[-1] < recent_volumes.mean() * 0.8:
                    volume_trend = "decreasing"
            
            return {
                'trend_strength': trend_strength,
                'price_vs_sma20': (price - sma_20) / sma_20 * 100 if sma_20 > 0 else 0,
                'price_vs_sma50': (price - sma_50) / sma_50 * 100 if sma_50 > 0 else 0,
                'price_vs_sma200': (price - sma_200) / sma_200 * 100 if sma_200 > 0 else 0,
                'resistance_level': float(high_20),
                'support_level': float(low_20),
                'volume_ratio': float(volume_ratio),
                'volume_trend': volume_trend,
                'volatility_20d': float(data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100) if len(data) >= 20 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}", "MARKET_DATA")
            return {}
    
    def _store_technical_data(self, data: Dict[str, Any]):
        """Store technical data in database"""
        try:
            market_data_entry = {
                'symbol': data['symbol'],
                'timeframe': data['timeframe'],
                'timestamp': data['last_updated'],
                'close_price': data['current_price'],
                'volume': data['volume'],
                'technical_indicators': data
            }
            
            supabase_manager.client.table(DatabaseConfig.MARKET_DATA_TABLE).upsert(market_data_entry).execute()
            
        except Exception as e:
            logger.error(f"Failed to store technical data: {e}", "MARKET_DATA")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'ib_gateway_enabled': True,
            'ib_gateway_connected': self.ib_connected,
            'primary_data_source': 'IB_GATEWAY',
            'fallback_available': False,
            'timestamp': datetime.now().isoformat()
        }
    
    def reconnect_ib_gateway(self) -> bool:
        """Attempt to reconnect to IB Gateway"""
        try:
            if ib_gateway.connected:
                ib_gateway.disconnect()
            
            if ib_gateway.connect():
                self.ib_connected = True
                logger.info("IB Gateway reconnected successfully", "MARKET_DATA")
                return True
            else:
                self.ib_connected = False
                logger.error("IB Gateway reconnection failed", "MARKET_DATA")
                return False
                
        except Exception as e:
            logger.error(f"IB Gateway reconnection error: {e}", "MARKET_DATA")
            self.ib_connected = False
            return False
    
    def get_hardcoded_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Provide hardcoded fallback data when IB Gateway is unavailable"""
        # Reasonable fallback values for testing
        fallback_prices = {
            'AVGO': 1850.0,
            'NVDA': 950.0,
            'TSLA': 250.0,
            'TSM': 105.0,
            'PLTR': 45.0,
            'CRWD': 320.0,
            'CEG': 180.0,
            'ANET': 420.0,
            'VRT': 520.0,
            'ASML': 780.0
        }
        
        base_price = fallback_prices.get(symbol, 100.0)
        
        return {
            'current_price': base_price,
            'volume': 1000000,
            'volume_ratio': 1.2,
            'trend_strength': 'BULLISH',
            'rsi': 55.0,
            'support_level': base_price * 0.95,
            'resistance_level': base_price * 1.05,
            'volatility_20d': 25.0,
            'volume_trend': 'increasing',
            'data_source': 'hardcoded_fallback'
        }

# Global IB-only market data manager instance
ib_market_data_manager = IBOnlyMarketDataManager()
