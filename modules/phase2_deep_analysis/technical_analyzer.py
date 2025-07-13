"""
Advanced Technical Analysis Module
Multi-timeframe technical analysis with sophisticated indicators
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from core.logger import logger

class TechnicalAnalyzer:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        
        # Timeframes for multi-timeframe analysis
        self.timeframes = {
            'daily': {'period': '60d', 'interval': '1d'},
            '4hour': {'period': '30d', 'interval': '4h'},
            '1hour': {'period': '7d', 'interval': '1h'},
            '15min': {'period': '2d', 'interval': '15m'}
        }
    
    def analyze_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive multi-timeframe technical analysis
        Returns detailed technical analysis results
        """
        logger.info(f"Starting technical analysis for {symbol}", "TECHNICAL_ANALYZER")
        
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'timeframe_analysis': {},
                'trend_analysis': self._analyze_trend(symbol),
                'momentum_analysis': self._analyze_momentum(symbol),
                'volatility_analysis': self._analyze_volatility(symbol),
                'support_resistance': self._identify_support_resistance(symbol),
                'pattern_recognition': self._detect_patterns(symbol),
                'entry_signals': self._generate_entry_signals(symbol),
                'risk_levels': self._calculate_risk_levels(symbol),
                'overall_technical_score': 0
            }
            
            # Analyze each timeframe
            for timeframe, config in self.timeframes.items():
                try:
                    tf_analysis = self._analyze_timeframe(symbol, timeframe, config)
                    analysis['timeframe_analysis'][timeframe] = tf_analysis
                except Exception as e:
                    logger.warning(f"Timeframe analysis failed for {symbol} {timeframe}: {e}", "TECHNICAL_ANALYZER")
                    analysis['timeframe_analysis'][timeframe] = {'error': str(e)}
            
            # Calculate overall technical score
            analysis['overall_technical_score'] = self._calculate_overall_score(analysis)
            
            logger.info(f"Technical analysis complete for {symbol}", "TECHNICAL_ANALYZER")
            return analysis
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'error': str(e)}
    
    def _analyze_timeframe(self, symbol: str, timeframe: str, config: Dict[str, str]) -> Dict[str, Any]:
        """Analyze a specific timeframe"""
        try:
            # Get data for this timeframe
            data = self._get_timeframe_data(symbol, config['period'], config['interval'])
            
            if data.empty:
                return {'error': 'No data available'}
            
            # Calculate indicators for this timeframe
            indicators = self._calculate_indicators(data)
            
            # Analyze trend for this timeframe
            trend_analysis = self._analyze_timeframe_trend(data, indicators)
            
            # Analyze momentum for this timeframe
            momentum_analysis = self._analyze_timeframe_momentum(data, indicators)
            
            # Identify key levels for this timeframe
            key_levels = self._identify_timeframe_levels(data)
            
            return {
                'timeframe': timeframe,
                'data_points': len(data),
                'current_price': float(data['Close'].iloc[-1]),
                'indicators': indicators,
                'trend': trend_analysis,
                'momentum': momentum_analysis,
                'key_levels': key_levels,
                'signal_strength': self._calculate_timeframe_signal_strength(trend_analysis, momentum_analysis)
            }
            
        except Exception as e:
            logger.error(f"Timeframe analysis failed for {symbol} {timeframe}: {e}", "TECHNICAL_ANALYZER")
            return {'error': str(e)}
    
    def _get_timeframe_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get price data for specific timeframe with caching"""
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                return self.cache[cache_key]
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                # Cache for appropriate duration based on interval
                cache_duration = self._get_cache_duration(interval)
                self.cache[cache_key] = data
                self.cache_expiry[cache_key] = datetime.now() + cache_duration
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get data for {symbol} {period} {interval}: {e}", "TECHNICAL_ANALYZER")
            return pd.DataFrame()
    
    def _get_cache_duration(self, interval: str) -> timedelta:
        """Get appropriate cache duration based on interval"""
        if interval in ['1m', '5m', '15m']:
            return timedelta(minutes=5)
        elif interval in ['30m', '1h']:
            return timedelta(minutes=15)
        elif interval in ['4h']:
            return timedelta(hours=1)
        else:  # Daily and above
            return timedelta(hours=4)
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for the given data"""
        try:
            indicators = {}
            
            # Trend Indicators
            indicators['sma_20'] = ta.sma(data['Close'], length=20).iloc[-1] if len(data) >= 20 else None
            indicators['sma_50'] = ta.sma(data['Close'], length=50).iloc[-1] if len(data) >= 50 else None
            indicators['ema_12'] = ta.ema(data['Close'], length=12).iloc[-1] if len(data) >= 12 else None
            indicators['ema_26'] = ta.ema(data['Close'], length=26).iloc[-1] if len(data) >= 26 else None
            
            # MACD
            macd_data = ta.macd(data['Close'])
            if macd_data is not None and not macd_data.empty:
                indicators['macd'] = macd_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in macd_data.columns else None
                indicators['macd_signal'] = macd_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in macd_data.columns else None
                indicators['macd_histogram'] = macd_data['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in macd_data.columns else None
            
            # Momentum Indicators
            indicators['rsi'] = ta.rsi(data['Close'], length=14).iloc[-1] if len(data) >= 14 else None
            indicators['stoch_k'] = ta.stoch(data['High'], data['Low'], data['Close'])['STOCHk_14_3_3'].iloc[-1] if len(data) >= 14 else None
            indicators['stoch_d'] = ta.stoch(data['High'], data['Low'], data['Close'])['STOCHd_14_3_3'].iloc[-1] if len(data) >= 14 else None
            
            # Volatility Indicators
            bb_data = ta.bbands(data['Close'], length=20)
            if bb_data is not None and not bb_data.empty:
                indicators['bb_upper'] = bb_data['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in bb_data.columns else None
                indicators['bb_middle'] = bb_data['BBM_20_2.0'].iloc[-1] if 'BBM_20_2.0' in bb_data.columns else None
                indicators['bb_lower'] = bb_data['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in bb_data.columns else None
                indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle'] if all(x is not None for x in [indicators['bb_upper'], indicators['bb_lower'], indicators['bb_middle']]) else None
            
            indicators['atr'] = ta.atr(data['High'], data['Low'], data['Close'], length=14).iloc[-1] if len(data) >= 14 else None
            
            # Volume Indicators
            indicators['volume_sma'] = ta.sma(data['Volume'], length=20).iloc[-1] if len(data) >= 20 else None
            indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] else 1
            
            # Price levels
            indicators['current_price'] = float(data['Close'].iloc[-1])
            indicators['high_20'] = float(data['High'].tail(20).max())
            indicators['low_20'] = float(data['Low'].tail(20).min())
            
            return indicators
            
        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}", "TECHNICAL_ANALYZER")
            return {}
    
    def _analyze_trend(self, symbol: str) -> Dict[str, Any]:
        """Analyze overall trend across timeframes"""
        try:
            trend_analysis = {
                'primary_trend': 'neutral',
                'trend_strength': 0,
                'trend_consistency': 0,
                'timeframe_alignment': {},
                'trend_signals': []
            }
            
            timeframe_trends = []
            
            # Analyze trend for each timeframe
            for timeframe, config in self.timeframes.items():
                try:
                    data = self._get_timeframe_data(symbol, config['period'], config['interval'])
                    if not data.empty:
                        indicators = self._calculate_indicators(data)
                        tf_trend = self._determine_timeframe_trend(data, indicators)
                        trend_analysis['timeframe_alignment'][timeframe] = tf_trend
                        timeframe_trends.append(tf_trend['direction'])
                except:
                    continue
            
            # Determine overall trend
            if timeframe_trends:
                bullish_count = timeframe_trends.count('bullish')
                bearish_count = timeframe_trends.count('bearish')
                
                if bullish_count > bearish_count:
                    trend_analysis['primary_trend'] = 'bullish'
                    trend_analysis['trend_strength'] = bullish_count / len(timeframe_trends)
                elif bearish_count > bullish_count:
                    trend_analysis['primary_trend'] = 'bearish'
                    trend_analysis['trend_strength'] = bearish_count / len(timeframe_trends)
                else:
                    trend_analysis['primary_trend'] = 'neutral'
                    trend_analysis['trend_strength'] = 0.5
                
                # Calculate trend consistency
                max_count = max(bullish_count, bearish_count, timeframe_trends.count('neutral'))
                trend_analysis['trend_consistency'] = max_count / len(timeframe_trends)
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'primary_trend': 'neutral', 'trend_strength': 0, 'trend_consistency': 0}
    
    def _determine_timeframe_trend(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Determine trend for a specific timeframe"""
        try:
            current_price = indicators['current_price']
            trend_score = 0
            signals = []
            
            # SMA trend analysis
            if indicators['sma_20'] and indicators['sma_50']:
                if indicators['sma_20'] > indicators['sma_50']:
                    trend_score += 1
                    signals.append('SMA bullish crossover')
                else:
                    trend_score -= 1
                    signals.append('SMA bearish crossover')
            
            # Price vs SMA
            if indicators['sma_20']:
                if current_price > indicators['sma_20']:
                    trend_score += 1
                    signals.append('Price above SMA20')
                else:
                    trend_score -= 1
                    signals.append('Price below SMA20')
            
            # MACD trend
            if indicators['macd'] and indicators['macd_signal']:
                if indicators['macd'] > indicators['macd_signal']:
                    trend_score += 1
                    signals.append('MACD bullish')
                else:
                    trend_score -= 1
                    signals.append('MACD bearish')
            
            # Determine direction
            if trend_score > 1:
                direction = 'bullish'
            elif trend_score < -1:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            return {
                'direction': direction,
                'score': trend_score,
                'signals': signals,
                'strength': abs(trend_score) / 3  # Normalize to 0-1
            }
            
        except Exception as e:
            logger.error(f"Timeframe trend determination failed: {e}", "TECHNICAL_ANALYZER")
            return {'direction': 'neutral', 'score': 0, 'signals': [], 'strength': 0}
    
    def _analyze_momentum(self, symbol: str) -> Dict[str, Any]:
        """Analyze momentum across timeframes"""
        try:
            # Get daily data for momentum analysis
            data = self._get_timeframe_data(symbol, '60d', '1d')
            
            if data.empty:
                return {'error': 'No data available'}
            
            indicators = self._calculate_indicators(data)
            
            momentum_analysis = {
                'rsi_level': indicators.get('rsi', 50),
                'rsi_signal': self._interpret_rsi(indicators.get('rsi', 50)),
                'stoch_level': indicators.get('stoch_k', 50),
                'stoch_signal': self._interpret_stochastic(indicators.get('stoch_k', 50), indicators.get('stoch_d', 50)),
                'macd_momentum': self._interpret_macd(indicators),
                'momentum_divergence': self._detect_momentum_divergence(data, indicators),
                'overall_momentum': 'neutral'
            }
            
            # Determine overall momentum
            momentum_score = 0
            
            if momentum_analysis['rsi_signal'] == 'bullish':
                momentum_score += 1
            elif momentum_analysis['rsi_signal'] == 'bearish':
                momentum_score -= 1
            
            if momentum_analysis['stoch_signal'] == 'bullish':
                momentum_score += 1
            elif momentum_analysis['stoch_signal'] == 'bearish':
                momentum_score -= 1
            
            if momentum_analysis['macd_momentum'] == 'bullish':
                momentum_score += 1
            elif momentum_analysis['macd_momentum'] == 'bearish':
                momentum_score -= 1
            
            if momentum_score > 1:
                momentum_analysis['overall_momentum'] = 'bullish'
            elif momentum_score < -1:
                momentum_analysis['overall_momentum'] = 'bearish'
            
            return momentum_analysis
            
        except Exception as e:
            logger.error(f"Momentum analysis failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'overall_momentum': 'neutral'}
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI signal"""
        if rsi is None:
            return 'neutral'
        
        if rsi > 70:
            return 'overbought'
        elif rsi < 30:
            return 'oversold'
        elif rsi > 50:
            return 'bullish'
        else:
            return 'bearish'
    
    def _interpret_stochastic(self, stoch_k: float, stoch_d: float) -> str:
        """Interpret Stochastic signal"""
        if stoch_k is None or stoch_d is None:
            return 'neutral'
        
        if stoch_k > 80 and stoch_d > 80:
            return 'overbought'
        elif stoch_k < 20 and stoch_d < 20:
            return 'oversold'
        elif stoch_k > stoch_d:
            return 'bullish'
        else:
            return 'bearish'
    
    def _interpret_macd(self, indicators: Dict[str, Any]) -> str:
        """Interpret MACD signal"""
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        macd_histogram = indicators.get('macd_histogram')
        
        if not all(x is not None for x in [macd, macd_signal, macd_histogram]):
            return 'neutral'
        
        if macd > macd_signal and macd_histogram > 0:
            return 'bullish'
        elif macd < macd_signal and macd_histogram < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def _analyze_volatility(self, symbol: str) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        try:
            data = self._get_timeframe_data(symbol, '60d', '1d')
            
            if data.empty:
                return {'error': 'No data available'}
            
            indicators = self._calculate_indicators(data)
            
            # Calculate volatility metrics
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Recent vs historical volatility
            recent_vol = returns.tail(10).std() * np.sqrt(252)
            historical_vol = returns.head(40).std() * np.sqrt(252)
            vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
            
            return {
                'current_volatility': volatility,
                'recent_volatility': recent_vol,
                'historical_volatility': historical_vol,
                'volatility_ratio': vol_ratio,
                'volatility_level': 'high' if vol_ratio > 1.5 else 'low' if vol_ratio < 0.7 else 'normal',
                'atr': indicators.get('atr', 0),
                'bb_width': indicators.get('bb_width', 0),
                'volatility_trend': 'increasing' if vol_ratio > 1.2 else 'decreasing' if vol_ratio < 0.8 else 'stable'
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'volatility_level': 'unknown'}
    
    def _identify_support_resistance(self, symbol: str) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        try:
            data = self._get_timeframe_data(symbol, '60d', '1d')
            
            if data.empty:
                return {'error': 'No data available'}
            
            current_price = float(data['Close'].iloc[-1])
            
            # Find pivot points
            highs = data['High'].rolling(window=5, center=True).max()
            lows = data['Low'].rolling(window=5, center=True).min()
            
            # Identify resistance levels (pivot highs)
            resistance_levels = []
            for i in range(2, len(data) - 2):
                if data['High'].iloc[i] == highs.iloc[i] and data['High'].iloc[i] > current_price:
                    resistance_levels.append(float(data['High'].iloc[i]))
            
            # Identify support levels (pivot lows)
            support_levels = []
            for i in range(2, len(data) - 2):
                if data['Low'].iloc[i] == lows.iloc[i] and data['Low'].iloc[i] < current_price:
                    support_levels.append(float(data['Low'].iloc[i]))
            
            # Sort and get closest levels
            resistance_levels = sorted(set(resistance_levels))[:5]  # Top 5 resistance
            support_levels = sorted(set(support_levels), reverse=True)[:5]  # Top 5 support
            
            return {
                'current_price': current_price,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'nearest_resistance': resistance_levels[0] if resistance_levels else None,
                'nearest_support': support_levels[0] if support_levels else None,
                'resistance_distance': ((resistance_levels[0] - current_price) / current_price * 100) if resistance_levels else None,
                'support_distance': ((current_price - support_levels[0]) / current_price * 100) if support_levels else None
            }
            
        except Exception as e:
            logger.error(f"Support/resistance identification failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'current_price': 0, 'resistance_levels': [], 'support_levels': []}
    
    def _detect_patterns(self, symbol: str) -> Dict[str, Any]:
        """Detect chart patterns"""
        try:
            data = self._get_timeframe_data(symbol, '60d', '1d')
            
            if data.empty:
                return {'error': 'No data available'}
            
            patterns = {
                'detected_patterns': [],
                'pattern_strength': 0,
                'breakout_potential': 'low'
            }
            
            # Simple pattern detection
            # Double top/bottom detection
            if self._detect_double_top(data):
                patterns['detected_patterns'].append('double_top')
                patterns['pattern_strength'] += 0.3
            
            if self._detect_double_bottom(data):
                patterns['detected_patterns'].append('double_bottom')
                patterns['pattern_strength'] += 0.3
            
            # Triangle patterns
            if self._detect_ascending_triangle(data):
                patterns['detected_patterns'].append('ascending_triangle')
                patterns['pattern_strength'] += 0.4
            
            if self._detect_descending_triangle(data):
                patterns['detected_patterns'].append('descending_triangle')
                patterns['pattern_strength'] += 0.4
            
            # Determine breakout potential
            if patterns['pattern_strength'] > 0.5:
                patterns['breakout_potential'] = 'high'
            elif patterns['pattern_strength'] > 0.2:
                patterns['breakout_potential'] = 'medium'
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'detected_patterns': [], 'pattern_strength': 0, 'breakout_potential': 'low'}
    
    def _detect_double_top(self, data: pd.DataFrame) -> bool:
        """Simple double top detection"""
        try:
            if len(data) < 20:
                return False
            
            highs = data['High'].tail(20)
            max_high = highs.max()
            
            # Find peaks
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    if highs.iloc[i] > max_high * 0.95:  # Within 5% of max
                        peaks.append(highs.iloc[i])
            
            return len(peaks) >= 2
            
        except:
            return False
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> bool:
        """Simple double bottom detection"""
        try:
            if len(data) < 20:
                return False
            
            lows = data['Low'].tail(20)
            min_low = lows.min()
            
            # Find troughs
            troughs = []
            for i in range(1, len(lows) - 1):
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    if lows.iloc[i] < min_low * 1.05:  # Within 5% of min
                        troughs.append(lows.iloc[i])
            
            return len(troughs) >= 2
            
        except:
            return False
    
    def _detect_ascending_triangle(self, data: pd.DataFrame) -> bool:
        """Simple ascending triangle detection"""
        try:
            if len(data) < 15:
                return False
            
            recent_data = data.tail(15)
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Check if highs are relatively flat
            high_variance = highs.var() / highs.mean()
            
            # Check if lows are trending up
            low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
            
            return high_variance < 0.01 and low_trend > 0
            
        except:
            return False
    
    def _detect_descending_triangle(self, data: pd.DataFrame) -> bool:
        """Simple descending triangle detection"""
        try:
            if len(data) < 15:
                return False
            
            recent_data = data.tail(15)
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Check if lows are relatively flat
            low_variance = lows.var() / lows.mean()
            
            # Check if highs are trending down
            high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
            
            return low_variance < 0.01 and high_trend < 0
            
        except:
            return False
    
    def _generate_entry_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate entry signals based on technical analysis"""
        try:
            # Get multiple timeframe data
            daily_data = self._get_timeframe_data(symbol, '60d', '1d')
            hourly_data = self._get_timeframe_data(symbol, '7d', '1h')
            
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'signal_strength': 0,
                'entry_recommendation': 'hold'
            }
            
            if not daily_data.empty:
                daily_indicators = self._calculate_indicators(daily_data)
                
                # RSI oversold signal
                if daily_indicators.get('rsi', 50) < 30:
                    signals['buy_signals'].append('RSI oversold')
                    signals['signal_strength'] += 0.3
                
                # RSI overbought signal
                if daily_indicators.get('rsi', 50) > 70:
                    signals['sell_signals'].append('RSI overbought')
                    signals['signal_strength'] -= 0.3
                
                # MACD bullish crossover
                if (daily_indicators.get('macd') and daily_indicators.get('macd_signal') and
                    daily_indicators['macd'] > daily_indicators['macd_signal']):
                    signals['buy_signals'].append('MACD bullish crossover')
                    signals['signal_strength'] += 0.4
                
                # Price above/below moving averages
                current_price = daily_indicators['current_price']
                if daily_indicators.get('sma_20') and current_price > daily_indicators['sma_20']:
                    signals['buy_signals'].append('Price above SMA20')
                    signals['signal_strength'] += 0.2
                
                # Bollinger Band signals
                if (daily_indicators.get('bb_lower') and 
                    current_price < daily_indicators['bb_lower']):
                    signals['buy_signals'].append('Price below BB lower')
                    signals['signal_strength'] += 0.3
            
            # Determine entry recommendation
            if signals['signal_strength'] > 0.5:
                signals['entry_recommendation'] = 'buy'
            elif signals['signal_strength'] < -0.5:
                signals['entry_recommendation'] = 'sell'
            
            return signals
            
        except Exception as e:
            logger.error(f"Entry signal generation failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'buy_signals': [], 'sell_signals': [], 'signal_strength': 0, 'entry_recommendation': 'hold'}
    
    def _calculate_risk_levels(self, symbol: str) -> Dict[str, Any]:
        """Calculate risk management levels"""
        try:
            data = self._get_timeframe_data(symbol, '60d', '1d')
            
            if data.empty:
                return {'error': 'No data available'}
            
            indicators = self._calculate_indicators(data)
            current_price = indicators['current_price']
            atr = indicators.get('atr', 0)
            
            # Calculate stop loss levels
            atr_stop_loss = current_price - (2 * atr)  # 2 ATR stop loss
            percentage_stop_loss = current_price * (1 - TradingConfig.STOP_LOSS_PERCENTAGE)
            
            # Calculate take profit levels
            atr_take_profit = current_price + (3 * atr)  # 3 ATR take profit
            percentage_take_profit = current_price * (1 + TradingConfig.TAKE_PROFIT_PERCENTAGE)
            
            # Support/resistance based levels
            sr_levels = self._identify_support_resistance(symbol)
            
            return {
                'current_price': current_price,
                'atr_stop_loss': atr_stop_loss,
                'percentage_stop_loss': percentage_stop_loss,
                'recommended_stop_loss': max(atr_stop_loss, percentage_stop_loss),
                'atr_take_profit': atr_take_profit,
                'percentage_take_profit': percentage_take_profit,
                'recommended_take_profit': min(atr_take_profit, percentage_take_profit),
                'nearest_support': sr_levels.get('nearest_support'),
                'nearest_resistance': sr_levels.get('nearest_resistance'),
                'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price, atr_stop_loss, atr_take_profit)
            }
            
        except Exception as e:
            logger.error(f"Risk level calculation failed for {symbol}: {e}", "TECHNICAL_ANALYZER")
            return {'current_price': 0}
    
    def _calculate_risk_reward_ratio(self, current_price: float, stop_loss: float, take_profit: float) -> float:
        """Calculate risk/reward ratio"""
        try:
            if stop_loss >= current_price or take_profit <= current_price:
                return 0
            
            risk = current_price - stop_loss
            reward = take_profit - current_price
            
            return reward / risk if risk > 0 else 0
            
        except Exception as e:
            logger.error(f"Risk/reward ratio calculation failed: {e}", "TECHNICAL_ANALYZER")
            return 0
    
    def _detect_momentum_divergence(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> str:
        """Detect momentum divergence"""
        try:
            if len(data) < 20:
                return 'insufficient_data'
            
            # Simple divergence detection
            recent_prices = data['Close'].tail(10)
            recent_rsi = ta.rsi(data['Close'], length=14).tail(10)
            
            if recent_rsi.empty:
                return 'no_divergence'
            
            # Check if price is making higher highs but RSI is making lower highs (bearish divergence)
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            if price_trend > 0 and rsi_trend < 0:
                return 'bearish_divergence'
            elif price_trend < 0 and rsi_trend > 0:
                return 'bullish_divergence'
            else:
                return 'no_divergence'
                
        except Exception as e:
            logger.error(f"Momentum divergence detection failed: {e}", "TECHNICAL_ANALYZER")
            return 'no_divergence'
    
    def _analyze_timeframe_momentum(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum for a specific timeframe"""
        try:
            momentum = {
                'rsi': indicators.get('rsi', 50),
                'rsi_signal': self._interpret_rsi(indicators.get('rsi', 50)),
                'stoch_signal': self._interpret_stochastic(indicators.get('stoch_k', 50), indicators.get('stoch_d', 50)),
                'macd_signal': self._interpret_macd(indicators),
                'momentum_score': 0
            }
            
            # Calculate momentum score
            score = 0
            if momentum['rsi_signal'] == 'bullish':
                score += 1
            elif momentum['rsi_signal'] == 'bearish':
                score -= 1
            
            if momentum['stoch_signal'] == 'bullish':
                score += 1
            elif momentum['stoch_signal'] == 'bearish':
                score -= 1
            
            if momentum['macd_signal'] == 'bullish':
                score += 1
            elif momentum['macd_signal'] == 'bearish':
                score -= 1
            
            momentum['momentum_score'] = score
            return momentum
            
        except Exception as e:
            logger.error(f"Timeframe momentum analysis failed: {e}", "TECHNICAL_ANALYZER")
            return {'momentum_score': 0}
    
    def _identify_timeframe_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key levels for a specific timeframe"""
        try:
            if data.empty:
                return {'support': [], 'resistance': []}
            
            current_price = float(data['Close'].iloc[-1])
            
            # Simple support/resistance based on recent highs and lows
            recent_data = data.tail(20)
            resistance = recent_data['High'].max()
            support = recent_data['Low'].min()
            
            return {
                'support': [support] if support < current_price else [],
                'resistance': [resistance] if resistance > current_price else [],
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Timeframe levels identification failed: {e}", "TECHNICAL_ANALYZER")
            return {'support': [], 'resistance': []}
    
    def _calculate_timeframe_signal_strength(self, trend_analysis: Dict[str, Any], momentum_analysis: Dict[str, Any]) -> float:
        """Calculate signal strength for a timeframe"""
        try:
            trend_score = trend_analysis.get('score', 0)
            momentum_score = momentum_analysis.get('momentum_score', 0)
            
            # Combine trend and momentum scores
            combined_score = (trend_score + momentum_score) / 2
            
            # Normalize to 0-1 scale
            return abs(combined_score) / 3
            
        except Exception as e:
            logger.error(f"Signal strength calculation failed: {e}", "TECHNICAL_ANALYZER")
            return 0
    
    def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall technical score"""
        try:
            score = 0
            components = 0
            
            # Trend component
            trend_analysis = analysis.get('trend_analysis', {})
            if trend_analysis.get('primary_trend') == 'bullish':
                score += trend_analysis.get('trend_strength', 0) * 30
            elif trend_analysis.get('primary_trend') == 'bearish':
                score -= trend_analysis.get('trend_strength', 0) * 30
            components += 30
            
            # Momentum component
            momentum_analysis = analysis.get('momentum_analysis', {})
            if momentum_analysis.get('overall_momentum') == 'bullish':
                score += 25
            elif momentum_analysis.get('overall_momentum') == 'bearish':
                score -= 25
            components += 25
            
            # Entry signals component
            entry_signals = analysis.get('entry_signals', {})
            signal_strength = entry_signals.get('signal_strength', 0)
            score += signal_strength * 25
            components += 25
            
            # Pattern recognition component
            patterns = analysis.get('pattern_recognition', {})
            pattern_strength = patterns.get('pattern_strength', 0)
            score += pattern_strength * 20
            components += 20
            
            # Normalize to 0-100 scale
            return max(0, min(100, 50 + score))
            
        except Exception as e:
            logger.error(f"Overall score calculation failed: {e}", "TECHNICAL_ANALYZER")
            return 50

# Global technical analyzer instance
technical_analyzer = TechnicalAnalyzer()
