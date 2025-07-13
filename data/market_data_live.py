"""
Live Market Data Manager - PRODUCTION ONLY
Real-time data with strict validation - NO FALLBACK DATA for live trading
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas_ta as ta
from config.trading_config import TradingConfig
from core.logger import logger
from data.ib_gateway import ib_gateway
import time

class LiveMarketDataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        self.ib_connected = False
        self.data_validation_enabled = True
        self.min_data_quality_threshold = 0.95  # 95% data quality required
        
        # Initialize IB Gateway connection - REQUIRED for live trading
        self._initialize_ib_gateway()
    
    def _initialize_ib_gateway(self):
        """Initialize IB Gateway connection - MANDATORY for live trading"""
        try:
            # Attempt connection
            logger.info("Connecting to IB Gateway for live trading...", "LIVE_DATA")
            
            if ib_gateway.connect():
                self.ib_connected = True
                logger.info("IB Gateway connected successfully - LIVE DATA ENABLED", "LIVE_DATA")
                
                # Verify connection quality
                self._verify_connection_quality()
            else:
                self.ib_connected = False
                logger.error("IB Gateway connection FAILED - LIVE TRADING DISABLED", "LIVE_DATA")
                raise ConnectionError("Cannot establish IB Gateway connection for live trading")
                
        except Exception as e:
            self.ib_connected = False
            logger.error(f"IB Gateway initialization FAILED: {e} - LIVE TRADING DISABLED", "LIVE_DATA")
            raise ConnectionError(f"Live trading requires IB Gateway connection: {e}")
    
    def _verify_connection_quality(self):
        """Verify IB Gateway connection quality for live trading"""
        try:
            # Test with a known liquid stock
            test_symbol = "SPY"
            
            # Test market data request
            start_time = time.time()
            test_data = ib_gateway.get_market_data(test_symbol, timeout=10)
            response_time = time.time() - start_time
            
            if not test_data:
                raise ConnectionError("IB Gateway not returning market data")
            
            if response_time > 5.0:  # More than 5 seconds is too slow
                logger.warning(f"IB Gateway response time slow: {response_time:.2f}s", "LIVE_DATA")
            
            logger.info(f"IB Gateway connection verified - Response time: {response_time:.2f}s", "LIVE_DATA")
            
        except Exception as e:
            logger.error(f"IB Gateway connection quality check failed: {e}", "LIVE_DATA")
            raise ConnectionError(f"IB Gateway connection quality insufficient: {e}")
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get stock price data - LIVE DATA ONLY"""
        if not self.ib_connected:
            logger.error(f"Cannot fetch data for {symbol} - IB Gateway not connected", "LIVE_DATA")
            raise ConnectionError("Live trading requires IB Gateway connection")
        
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache with short timeout for live data
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                cache_timeout = timedelta(minutes=1) if interval in ['1m', '5m'] else timedelta(minutes=5)
                if datetime.now() - last_update < cache_timeout:
                    return self.cache[cache_key]
            
            # Get data from IB Gateway
            ib_duration, ib_bar_size = self._convert_to_ib_format(period, interval)
            data = ib_gateway.get_historical_data(symbol, ib_duration, ib_bar_size, timeout=30)
            
            if data.empty:
                logger.error(f"No historical data received for {symbol} from IB Gateway", "LIVE_DATA")
                raise ValueError(f"No historical data available for {symbol}")
            
            # Validate data quality
            if not self._validate_data_quality(data, symbol):
                logger.error(f"Data quality validation failed for {symbol}", "LIVE_DATA")
                raise ValueError(f"Data quality insufficient for {symbol}")
            
            # Cache validated data
            self.cache[cache_key] = data
            self.last_update[cache_key] = datetime.now()
            
            logger.info(f"Fetched {len(data)} rows of validated data for {symbol}", "LIVE_DATA")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch live data for {symbol}: {e}", "LIVE_DATA")
            raise
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate data quality for live trading"""
        try:
            if data.empty:
                logger.error(f"Empty dataset for {symbol}", "LIVE_DATA")
                return False
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing columns for {symbol}: {missing_columns}", "LIVE_DATA")
                return False
            
            # Check for null values
            null_percentage = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if null_percentage > (1 - self.min_data_quality_threshold):
                logger.error(f"Too many null values for {symbol}: {null_percentage:.2%}", "LIVE_DATA")
                return False
            
            # Check for zero/negative prices
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if (data[col] <= 0).any():
                    logger.error(f"Invalid prices detected in {col} for {symbol}", "LIVE_DATA")
                    return False
            
            # Check OHLC relationships
            invalid_ohlc = (
                (data['High'] < data['Low']) |
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            ).any()
            
            if invalid_ohlc:
                logger.error(f"Invalid OHLC relationships detected for {symbol}", "LIVE_DATA")
                return False
            
            # Check for reasonable volume
            if (data['Volume'] < 0).any():
                logger.error(f"Negative volume detected for {symbol}", "LIVE_DATA")
                return False
            
            # Check data recency
            latest_date = data.index[-1]
            if isinstance(latest_date, str):
                latest_date = pd.to_datetime(latest_date)
            
            days_old = (datetime.now() - latest_date).days
            if days_old > 7:  # Data older than 7 days
                logger.warning(f"Data for {symbol} is {days_old} days old", "LIVE_DATA")
            
            logger.debug(f"Data quality validation passed for {symbol}", "LIVE_DATA")
            return True
            
        except Exception as e:
            logger.error(f"Data quality validation error for {symbol}: {e}", "LIVE_DATA")
            return False
    
    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote - LIVE DATA ONLY"""
        if not self.ib_connected:
            logger.error(f"Cannot fetch real-time quote for {symbol} - IB Gateway not connected", "LIVE_DATA")
            raise ConnectionError("Live trading requires IB Gateway connection")
        
        try:
            market_data = ib_gateway.get_market_data(symbol, timeout=10)
            
            if not market_data:
                logger.error(f"No real-time data received for {symbol}", "LIVE_DATA")
                raise ValueError(f"No real-time data available for {symbol}")
            
            # Validate quote data
            if not self._validate_quote_data(market_data, symbol):
                logger.error(f"Quote data validation failed for {symbol}", "LIVE_DATA")
                raise ValueError(f"Quote data quality insufficient for {symbol}")
            
            quote = {
                'symbol': symbol,
                'last': market_data.get('last', 0),
                'bid': market_data.get('bid', 0),
                'ask': market_data.get('ask', 0),
                'volume': market_data.get('volume', 0),
                'high': market_data.get('high', 0),
                'low': market_data.get('low', 0),
                'close': market_data.get('close', 0),
                'timestamp': market_data.get('timestamp', datetime.now()),
                'source': 'IB_GATEWAY_LIVE',
                'validated': True
            }
            
            logger.debug(f"Real-time quote validated for {symbol}: ${quote['last']:.2f}", "LIVE_DATA")
            return quote
            
        except Exception as e:
            logger.error(f"Failed to get real-time quote for {symbol}: {e}", "LIVE_DATA")
            raise
    
    def _validate_quote_data(self, quote_data: Dict[str, Any], symbol: str) -> bool:
        """Validate real-time quote data"""
        try:
            # Check for required fields
            required_fields = ['last', 'bid', 'ask']
            for field in required_fields:
                if field not in quote_data or quote_data[field] <= 0:
                    logger.error(f"Invalid {field} price for {symbol}: {quote_data.get(field, 'missing')}", "LIVE_DATA")
                    return False
            
            # Check bid/ask spread reasonableness
            bid = quote_data['bid']
            ask = quote_data['ask']
            last = quote_data['last']
            
            if ask <= bid:
                logger.error(f"Invalid bid/ask spread for {symbol}: bid={bid}, ask={ask}", "LIVE_DATA")
                return False
            
            # Check if last price is within bid/ask range (with some tolerance)
            spread = ask - bid
            tolerance = spread * 2  # Allow last price to be outside bid/ask by 2x spread
            
            if not (bid - tolerance <= last <= ask + tolerance):
                logger.warning(f"Last price outside bid/ask range for {symbol}: last={last}, bid={bid}, ask={ask}", "LIVE_DATA")
            
            # Check timestamp recency
            timestamp = quote_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            age_seconds = (datetime.now() - timestamp).total_seconds()
            if age_seconds > 60:  # Quote older than 1 minute
                logger.warning(f"Quote data for {symbol} is {age_seconds:.0f} seconds old", "LIVE_DATA")
            
            return True
            
        except Exception as e:
            logger.error(f"Quote validation error for {symbol}: {e}", "LIVE_DATA")
            return False
    
    def get_technical_indicators(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """Calculate technical indicators from LIVE DATA ONLY"""
        try:
            # Get validated historical data
            data = self.get_stock_data(symbol, period="6mo", interval=timeframe)
            
            if data.empty:
                logger.error(f"No data available for technical analysis of {symbol}", "LIVE_DATA")
                raise ValueError(f"No data available for technical analysis of {symbol}")
            
            # Calculate technical indicators
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
                'rsi': float(latest.get('RSI_14', 50)),
                'macd': float(latest.get('MACD_12_26_9', 0)),
                'macd_signal': float(latest.get('MACDs_12_26_9', 0)),
                'macd_histogram': float(latest.get('MACDh_12_26_9', 0)),
                'bb_upper': float(latest.get('BBU_20_2.0', 0)),
                'bb_middle': float(latest.get('BBM_20_2.0', 0)),
                'bb_lower': float(latest.get('BBL_20_2.0', 0)),
                'stoch_k': float(latest.get('STOCHk_14_3_3', 50)),
                'stoch_d': float(latest.get('STOCHd_14_3_3', 50)),
                'adx': float(latest.get('ADX_14', 25)),
                'atr': float(latest.get('ATRr_14', 0)),
                'obv': float(latest.get('OBV', 0)),
                'last_updated': datetime.now().isoformat(),
                'data_source': 'ib_gateway_live',
                'validated': True
            }
            
            # Add trend analysis
            technical_data.update(self._analyze_trends(data))
            
            logger.debug(f"Technical indicators calculated for {symbol}", "LIVE_DATA")
            return technical_data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators for {symbol}: {e}", "LIVE_DATA")
            raise
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data - LIVE DATA ONLY"""
        if not self.ib_connected:
            logger.error(f"Cannot fetch fundamental data for {symbol} - IB Gateway not connected", "LIVE_DATA")
            raise ConnectionError("Live trading requires IB Gateway connection")
        
        try:
            fundamental_data = ib_gateway.get_fundamental_data(symbol, timeout=15)
            
            if not fundamental_data:
                logger.error(f"No fundamental data received for {symbol}", "LIVE_DATA")
                raise ValueError(f"No fundamental data available for {symbol}")
            
            # Validate fundamental data
            if not self._validate_fundamental_data(fundamental_data, symbol):
                logger.error(f"Fundamental data validation failed for {symbol}", "LIVE_DATA")
                raise ValueError(f"Fundamental data quality insufficient for {symbol}")
            
            fundamental_data['validated'] = True
            fundamental_data['data_source'] = 'ib_gateway_live'
            fundamental_data['last_updated'] = datetime.now().isoformat()
            
            logger.debug(f"Fundamental data validated for {symbol}", "LIVE_DATA")
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to get fundamental data for {symbol}: {e}", "LIVE_DATA")
            raise
    
    def _validate_fundamental_data(self, fundamental_data: Dict[str, Any], symbol: str) -> bool:
        """Validate fundamental data quality"""
        try:
            # Check for key metrics
            key_metrics = ['revenue_growth_yoy', 'market_cap']
            for metric in key_metrics:
                if metric not in fundamental_data:
                    logger.warning(f"Missing key metric {metric} for {symbol}", "LIVE_DATA")
            
            # Validate market cap
            market_cap = fundamental_data.get('market_cap', 0)
            if market_cap <= 0:
                logger.error(f"Invalid market cap for {symbol}: {market_cap}", "LIVE_DATA")
                return False
            
            # Validate growth rates (should be reasonable)
            revenue_growth = fundamental_data.get('revenue_growth_yoy', 0)
            if abs(revenue_growth) > 1000:  # More than 1000% growth is suspicious
                logger.warning(f"Suspicious revenue growth for {symbol}: {revenue_growth}%", "LIVE_DATA")
            
            return True
            
        except Exception as e:
            logger.error(f"Fundamental data validation error for {symbol}: {e}", "LIVE_DATA")
            return False
    
    def _convert_to_ib_format(self, period: str, interval: str) -> Tuple[str, str]:
        """Convert period/interval to IB Gateway format"""
        period_map = {
            "1d": "1 D", "5d": "5 D", "1mo": "1 M", "3mo": "3 M", "6mo": "6 M",
            "1y": "1 Y", "2y": "2 Y", "5y": "5 Y", "10y": "10 Y", "ytd": "1 Y", "max": "10 Y"
        }
        
        interval_map = {
            "1m": "1 min", "2m": "2 mins", "5m": "5 mins", "15m": "15 mins", "30m": "30 mins",
            "60m": "1 hour", "90m": "1 hour", "1h": "1 hour", "1d": "1 day",
            "5d": "1 week", "1wk": "1 week", "1mo": "1 month", "3mo": "1 month"
        }
        
        return period_map.get(period, "1 Y"), interval_map.get(interval, "1 day")
    
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
            
            return {
                'trend_strength': trend_strength,
                'price_vs_sma20': (price - sma_20) / sma_20 * 100 if sma_20 > 0 else 0,
                'price_vs_sma50': (price - sma_50) / sma_50 * 100 if sma_50 > 0 else 0,
                'price_vs_sma200': (price - sma_200) / sma_200 * 100 if sma_200 > 0 else 0,
                'resistance_level': float(high_20),
                'support_level': float(low_20),
                'volume_ratio': float(volume_ratio),
                'volatility_20d': float(data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100) if len(data) >= 20 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}", "LIVE_DATA")
            return {}
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            'ib_gateway_connected': self.ib_connected,
            'data_validation_enabled': self.data_validation_enabled,
            'min_data_quality_threshold': self.min_data_quality_threshold,
            'live_trading_ready': self.ib_connected and self.data_validation_enabled,
            'timestamp': datetime.now().isoformat()
        }
    
    def is_ready_for_live_trading(self) -> bool:
        """Check if system is ready for live trading"""
        return self.ib_connected and self.data_validation_enabled

# Global live market data manager instance
live_market_data_manager = LiveMarketDataManager()
