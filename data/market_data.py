"""
Market Data Collection and Management
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas_ta as ta
from config.trading_config import TradingConfig, DatabaseConfig
from config.supabase_config import supabase_manager
from core.logger import logger
import json

class MarketDataManager:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
    
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get stock price data from yfinance"""
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache first (refresh every 5 minutes for intraday data)
            if cache_key in self.cache:
                last_update = self.last_update.get(cache_key, datetime.min)
                if datetime.now() - last_update < timedelta(minutes=5):
                    return self.cache[cache_key]
            
            # Fetch fresh data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}", "MARKET_DATA")
                return pd.DataFrame()
            
            # Cache the data
            self.cache[cache_key] = data
            self.last_update[cache_key] = datetime.now()
            
            logger.debug(f"Fetched {len(data)} rows of data for {symbol}", "MARKET_DATA")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}", "MARKET_DATA")
            return pd.DataFrame()
    
    def get_fundamental_data(self, symbol: str) -> Dict[str, any]:
        """Get fundamental data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get quarterly and annual financials
            quarterly_financials = ticker.quarterly_financials
            annual_financials = ticker.financials
            
            # Calculate growth metrics
            growth_metrics = self._calculate_growth_metrics(quarterly_financials, annual_financials)
            
            fundamental_data = {
                'symbol': symbol,
                'market_cap': info.get('marketCap', 0),
                'revenue_ttm': info.get('totalRevenue', 0),
                'revenue_growth_yoy': growth_metrics.get('revenue_growth_yoy', 0),
                'revenue_growth_qoq': growth_metrics.get('revenue_growth_qoq', 0),
                'profit_margins': info.get('profitMargins', 0),
                'operating_margins': info.get('operatingMargins', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'price_to_earnings': info.get('trailingPE', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'beta': info.get('beta', 1.0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'analyst_target_price': info.get('targetMeanPrice', 0),
                'recommendation': info.get('recommendationMean', 0),
                'last_updated': datetime.now().isoformat()
            }
            
            # Store in database
            self._store_fundamental_data(fundamental_data)
            
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Failed to fetch fundamental data for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def _calculate_growth_metrics(self, quarterly_financials: pd.DataFrame, annual_financials: pd.DataFrame) -> Dict[str, float]:
        """Calculate revenue growth metrics"""
        try:
            growth_metrics = {}
            
            # Revenue growth YoY (from annual data)
            if not annual_financials.empty and 'Total Revenue' in annual_financials.index:
                revenue_data = annual_financials.loc['Total Revenue'].dropna()
                if len(revenue_data) >= 2:
                    latest_revenue = revenue_data.iloc[0]
                    previous_revenue = revenue_data.iloc[1]
                    growth_metrics['revenue_growth_yoy'] = ((latest_revenue - previous_revenue) / previous_revenue) * 100
            
            # Revenue growth QoQ (from quarterly data)
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                quarterly_revenue = quarterly_financials.loc['Total Revenue'].dropna()
                if len(quarterly_revenue) >= 2:
                    latest_quarter = quarterly_revenue.iloc[0]
                    previous_quarter = quarterly_revenue.iloc[1]
                    growth_metrics['revenue_growth_qoq'] = ((latest_quarter - previous_quarter) / previous_quarter) * 100
            
            return growth_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate growth metrics: {e}", "MARKET_DATA")
            return {}
    
    def get_technical_indicators(self, symbol: str, timeframe: str = "1d") -> Dict[str, any]:
        """Calculate technical indicators for a stock"""
        try:
            data = self.get_stock_data(symbol, period="6mo", interval=timeframe)
            
            if data.empty:
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
                'last_updated': datetime.now().isoformat()
            }
            
            # Add trend analysis
            technical_data.update(self._analyze_trends(data))
            
            # Store in database
            self._store_technical_data(technical_data)
            
            return technical_data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators for {symbol}: {e}", "MARKET_DATA")
            return {}
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze price trends and patterns"""
        try:
            latest = data.iloc[-1]
            prev_5 = data.iloc[-6:-1]
            
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
            volume_ratio = latest['Volume'] / avg_volume_20
            
            return {
                'trend_strength': trend_strength,
                'price_vs_sma20': (price - sma_20) / sma_20 * 100,
                'price_vs_sma50': (price - sma_50) / sma_50 * 100,
                'price_vs_sma200': (price - sma_200) / sma_200 * 100,
                'resistance_level': float(high_20),
                'support_level': float(low_20),
                'volume_ratio': float(volume_ratio),
                'volatility_20d': float(data['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}", "MARKET_DATA")
            return {}
    
    def _store_fundamental_data(self, data: Dict[str, any]):
        """Store fundamental data in database"""
        try:
            # This would typically go in a separate fundamentals table
            # For now, we'll store it as market data with a special timeframe
            market_data_entry = {
                'symbol': data['symbol'],
                'timeframe': 'fundamental',
                'timestamp': data['last_updated'],
                'technical_indicators': data
            }
            
            supabase_manager.client.table(DatabaseConfig.MARKET_DATA_TABLE).upsert(market_data_entry).execute()
            
        except Exception as e:
            logger.error(f"Failed to store fundamental data: {e}", "MARKET_DATA")
    
    def _store_technical_data(self, data: Dict[str, any]):
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
    
    def screen_stocks_by_fundamentals(self, stock_list: List[str]) -> List[Dict[str, any]]:
        """Screen stocks based on fundamental criteria"""
        qualified_stocks = []
        
        for symbol in stock_list:
            try:
                logger.info(f"Screening {symbol} for fundamental criteria", "MARKET_DATA")
                
                fundamental_data = self.get_fundamental_data(symbol)
                
                if not fundamental_data:
                    continue
                
                # Check if stock meets fundamental criteria
                revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
                revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
                
                if (revenue_growth_yoy >= TradingConfig.MIN_REVENUE_GROWTH_YOY and 
                    revenue_growth_qoq >= TradingConfig.MIN_REVENUE_GROWTH_QOQ):
                    
                    qualified_stocks.append({
                        'symbol': symbol,
                        'revenue_growth_yoy': revenue_growth_yoy,
                        'revenue_growth_qoq': revenue_growth_qoq,
                        'fundamental_data': fundamental_data
                    })
                    
                    logger.info(f"{symbol} passed fundamental screening", "MARKET_DATA")
                else:
                    logger.debug(f"{symbol} failed fundamental screening: YoY={revenue_growth_yoy:.1f}%, QoQ={revenue_growth_qoq:.1f}%", "MARKET_DATA")
                    
            except Exception as e:
                logger.error(f"Failed to screen {symbol}: {e}", "MARKET_DATA")
                continue
        
        return qualified_stocks

# Global market data manager instance
market_data_manager = MarketDataManager()
