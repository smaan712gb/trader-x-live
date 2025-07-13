"""
Technical Analysis Engine for Trader-X
Multi-timeframe analysis with entry/exit signals for $5-10 gains
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import FMP provider
try:
    from data.fmp_market_data import fmp_provider
    FMP_AVAILABLE = True
except ImportError:
    FMP_AVAILABLE = False
    print("⚠️ FMP provider not available")

class TechnicalAnalysisEngine:
    def __init__(self):
        self.timeframes = {
            'daily': '1d',
            '4hour': '1h',  # We'll resample to 4h
            '1hour': '1h',
            '15min': '15m'
        }
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive technical analysis across all timeframes"""
        try:
            # Try to get comprehensive data from FMP first
            if FMP_AVAILABLE:
                fmp_data = fmp_provider.get_comprehensive_technical_data(symbol)
                if 'error' not in fmp_data:
                    # Use FMP data as primary source
                    current_price = fmp_data['quote']['price']
                    daily_data = fmp_data['historical_data']['daily']
                    hourly_data = fmp_data['historical_data']['hourly']
                    
                    # Get minute data separately if needed
                    minute_data = self._get_stock_data(symbol, '15m', '7d')
                    if minute_data.empty and FMP_AVAILABLE:
                        minute_data = fmp_provider.get_intraday_data(symbol, '15min')
                else:
                    # Fallback to original method
                    daily_data = self._get_stock_data(symbol, '1d', '3mo')
                    hourly_data = self._get_stock_data(symbol, '1h', '30d')
                    minute_data = self._get_stock_data(symbol, '15m', '7d')
                    current_price = daily_data['Close'].iloc[-1] if not daily_data.empty else 0
            else:
                # Original method
                daily_data = self._get_stock_data(symbol, '1d', '3mo')
                hourly_data = self._get_stock_data(symbol, '1h', '30d')
                minute_data = self._get_stock_data(symbol, '15m', '7d')
                current_price = daily_data['Close'].iloc[-1] if not daily_data.empty else 0
            
            # Resample hourly to 4-hour
            four_hour_data = self._resample_to_4h(hourly_data) if not hourly_data.empty else pd.DataFrame()
            
            # Analyze each timeframe
            daily_analysis = self._analyze_timeframe(daily_data, 'Daily', current_price)
            four_hour_analysis = self._analyze_timeframe(four_hour_data, '4-Hour', current_price)
            hourly_analysis = self._analyze_timeframe(hourly_data, '1-Hour', current_price)
            minute_analysis = self._analyze_timeframe(minute_data, '15-Min', current_price)
            
            # Get support/resistance levels
            support_resistance = self._calculate_support_resistance(daily_data, current_price)
            
            # Calculate entry/exit signals for $5-10 gains
            entry_signals = self._calculate_entry_signals(
                daily_data, hourly_data, minute_data, current_price, support_resistance
            )
            
            # Get high/low analysis
            high_low_analysis = self._analyze_highs_lows(daily_data, current_price)
            
            # Overall technical score
            technical_score = self._calculate_technical_score(
                daily_analysis, four_hour_analysis, hourly_analysis, minute_analysis
            )
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'timestamp': datetime.now().isoformat(),
                'timeframe_analysis': {
                    'daily': daily_analysis,
                    '4hour': four_hour_analysis,
                    '1hour': hourly_analysis,
                    '15min': minute_analysis
                },
                'support_resistance': support_resistance,
                'entry_signals': entry_signals,
                'high_low_analysis': high_low_analysis,
                'technical_score': technical_score,
                'recommendation': self._get_recommendation(technical_score, entry_signals)
            }
            
        except Exception as e:
            return {'error': f"Technical analysis failed: {str(e)}"}
    
    def _get_stock_data(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        """Get stock data for specified timeframe"""
        try:
            # First try yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                # Try alternative periods if data is empty
                alternative_periods = ['1mo', '2mo', '6mo', '1y']
                for alt_period in alternative_periods:
                    if alt_period != period:
                        data = ticker.history(period=alt_period, interval=interval)
                        if not data.empty:
                            break
            
            # If yfinance fails, try FMP as fallback
            if data.empty and FMP_AVAILABLE:
                print(f"📡 yfinance failed for {symbol}, trying FMP API...")
                if interval in ['1d', 'daily']:
                    data = fmp_provider.get_historical_data(symbol, period)
                else:
                    data = fmp_provider.get_intraday_data(symbol, interval)
            
            # Ensure we have a proper datetime index
            if not data.empty and not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error getting data for {symbol}: {str(e)}")
            # Return empty DataFrame with proper structure if all fails
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def _resample_to_4h(self, hourly_data: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 4-hour timeframe"""
        four_hour = hourly_data.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return four_hour
    
    def _analyze_timeframe(self, data: pd.DataFrame, timeframe: str, current_price: float) -> Dict[str, Any]:
        """Analyze a specific timeframe"""
        if len(data) < 20:
            return {'error': f'Insufficient data for {timeframe} analysis'}
        
        # Calculate technical indicators
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        # Convert to DataFrame for pandas_ta
        df = pd.DataFrame({
            'Open': data['Open'],
            'High': data['High'],
            'Low': data['Low'],
            'Close': data['Close'],
            'Volume': data['Volume']
        })
        
        # Moving averages
        sma_20 = ta.sma(df['Close'], length=20)
        sma_50 = ta.sma(df['Close'], length=50) if len(close) >= 50 else None
        ema_12 = ta.ema(df['Close'], length=12)
        ema_26 = ta.ema(df['Close'], length=26)
        
        # MACD
        macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        macd = macd_data['MACD_12_26_9'] if macd_data is not None else pd.Series([0] * len(close))
        macd_signal = macd_data['MACDs_12_26_9'] if macd_data is not None else pd.Series([0] * len(close))
        macd_hist = macd_data['MACDh_12_26_9'] if macd_data is not None else pd.Series([0] * len(close))
        
        # RSI
        rsi = ta.rsi(df['Close'], length=14)
        
        # Bollinger Bands
        bb_data = ta.bbands(df['Close'], length=20, std=2)
        bb_upper = bb_data['BBU_20_2.0'] if bb_data is not None else pd.Series([close[-1]] * len(close))
        bb_middle = bb_data['BBM_20_2.0'] if bb_data is not None else pd.Series([close[-1]] * len(close))
        bb_lower = bb_data['BBL_20_2.0'] if bb_data is not None else pd.Series([close[-1]] * len(close))
        
        # Stochastic
        stoch_data = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        stoch_k = stoch_data['STOCHk_14_3_3'] if stoch_data is not None else pd.Series([50] * len(close))
        stoch_d = stoch_data['STOCHd_14_3_3'] if stoch_data is not None else pd.Series([50] * len(close))
        
        # Volume indicators
        volume_sma = ta.sma(df['Volume'], length=20)
        
        # Current values - handle pandas Series properly
        current_rsi = rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50
        current_macd = macd.iloc[-1] if not macd.empty and not pd.isna(macd.iloc[-1]) else 0
        current_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty and not pd.isna(macd_signal.iloc[-1]) else 0
        current_stoch_k = stoch_k.iloc[-1] if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else 50
        
        # Trend analysis
        trend = self._determine_trend(close, sma_20, ema_12, ema_26)
        
        # Support/resistance for this timeframe
        recent_highs = data['High'].rolling(window=10).max().iloc[-5:]
        recent_lows = data['Low'].rolling(window=10).min().iloc[-5:]
        
        resistance = recent_highs.max()
        support = recent_lows.min()
        
        # Signal strength
        signal_strength = self._calculate_signal_strength(current_rsi, current_macd, current_stoch_k, trend)
        
        return {
            'timeframe': timeframe,
            'trend': trend,
            'signal_strength': signal_strength,
            'indicators': {
                'rsi': round(current_rsi, 2),
                'macd': round(current_macd, 4),
                'macd_signal': round(current_macd_signal, 4),
                'stochastic_k': round(current_stoch_k, 2),
                'sma_20': round(sma_20.iloc[-1], 2) if not sma_20.empty and not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': round(sma_50.iloc[-1], 2) if sma_50 is not None and not sma_50.empty and not pd.isna(sma_50.iloc[-1]) else None,
                'bb_upper': round(bb_upper.iloc[-1], 2) if not bb_upper.empty and not pd.isna(bb_upper.iloc[-1]) else None,
                'bb_lower': round(bb_lower.iloc[-1], 2) if not bb_lower.empty and not pd.isna(bb_lower.iloc[-1]) else None
            },
            'levels': {
                'resistance': round(resistance, 2),
                'support': round(support, 2),
                'distance_to_resistance': round(((resistance - current_price) / current_price) * 100, 2),
                'distance_to_support': round(((current_price - support) / current_price) * 100, 2)
            }
        }
    
    def _determine_trend(self, close: np.ndarray, sma_20: np.ndarray, ema_12: np.ndarray, ema_26: np.ndarray) -> str:
        """Determine trend direction"""
        current_price = close[-1]
        sma_20_current = sma_20[-1] if not np.isnan(sma_20[-1]) else current_price
        ema_12_current = ema_12[-1] if not np.isnan(ema_12[-1]) else current_price
        ema_26_current = ema_26[-1] if not np.isnan(ema_26[-1]) else current_price
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs moving averages
        if current_price > sma_20_current:
            bullish_signals += 1
        else:
            bearish_signals += 1
            
        if ema_12_current > ema_26_current:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Recent price action
        if len(close) >= 5:
            recent_trend = np.polyfit(range(5), close[-5:], 1)[0]
            if recent_trend > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return 'Bullish'
        elif bearish_signals > bullish_signals:
            return 'Bearish'
        else:
            return 'Neutral'
    
    def _calculate_signal_strength(self, rsi: float, macd: float, stoch_k: float, trend: str) -> str:
        """Calculate signal strength"""
        strength_score = 0
        
        # RSI signals
        if rsi < 30:  # Oversold
            strength_score += 2
        elif rsi > 70:  # Overbought
            strength_score -= 2
        elif 40 <= rsi <= 60:  # Neutral
            strength_score += 1
        
        # MACD signals
        if macd > 0:
            strength_score += 1
        else:
            strength_score -= 1
        
        # Stochastic signals
        if stoch_k < 20:  # Oversold
            strength_score += 1
        elif stoch_k > 80:  # Overbought
            strength_score -= 1
        
        # Trend alignment
        if trend == 'Bullish':
            strength_score += 2
        elif trend == 'Bearish':
            strength_score -= 2
        
        if strength_score >= 4:
            return 'Very Strong'
        elif strength_score >= 2:
            return 'Strong'
        elif strength_score >= 0:
            return 'Moderate'
        elif strength_score >= -2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _calculate_support_resistance(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Calculate key support and resistance levels"""
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find pivot points
        resistance_levels = []
        support_levels = []
        
        # Look for local maxima and minima
        for i in range(2, len(highs) - 2):
            # Resistance (local maxima)
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
            
            # Support (local minima)
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Get the most relevant levels (closest to current price)
        resistance_levels = sorted([r for r in resistance_levels if r > current_price])[:3]
        support_levels = sorted([s for s in support_levels if s < current_price], reverse=True)[:3]
        
        # Add psychological levels (round numbers)
        psychological_levels = []
        for level in range(int(current_price) - 20, int(current_price) + 21, 5):
            if abs(level - current_price) / current_price < 0.1:  # Within 10%
                psychological_levels.append(level)
        
        return {
            'resistance_levels': [round(r, 2) for r in resistance_levels],
            'support_levels': [round(s, 2) for s in support_levels],
            'psychological_levels': psychological_levels,
            'nearest_resistance': round(min(resistance_levels), 2) if resistance_levels else None,
            'nearest_support': round(max(support_levels), 2) if support_levels else None
        }
    
    def _calculate_entry_signals(self, daily_data: pd.DataFrame, hourly_data: pd.DataFrame, 
                               minute_data: pd.DataFrame, current_price: float, 
                               support_resistance: Dict) -> Dict[str, Any]:
        """Calculate specific entry/exit signals for $5-10 gains"""
        
        signals = []
        
        # Get nearest support and resistance
        nearest_resistance = support_resistance.get('nearest_resistance')
        nearest_support = support_resistance.get('nearest_support')
        
        # Calculate potential gains
        if nearest_resistance:
            potential_gain = nearest_resistance - current_price
            if 5 <= potential_gain <= 15:  # $5-15 potential gain
                entry_price = current_price
                target_price = nearest_resistance - 0.50  # Leave some buffer
                stop_loss = nearest_support if nearest_support else current_price * 0.98
                
                signals.append({
                    'type': 'Long',
                    'entry_price': round(entry_price, 2),
                    'target_price': round(target_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'potential_gain': round(potential_gain, 2),
                    'risk_reward_ratio': round(potential_gain / (entry_price - stop_loss), 2) if entry_price > stop_loss else 0,
                    'confidence': self._calculate_entry_confidence(daily_data, hourly_data, 'long'),
                    'timeframe': 'Intraday to 1-3 days',
                    'reasoning': f'Resistance at ${target_price:.2f} offers ${potential_gain:.2f} upside potential'
                })
        
        # Look for short opportunities
        if nearest_support:
            potential_gain = current_price - nearest_support
            if 5 <= potential_gain <= 15:  # $5-15 potential gain
                entry_price = current_price
                target_price = nearest_support + 0.50  # Leave some buffer
                stop_loss = nearest_resistance if nearest_resistance else current_price * 1.02
                
                signals.append({
                    'type': 'Short',
                    'entry_price': round(entry_price, 2),
                    'target_price': round(target_price, 2),
                    'stop_loss': round(stop_loss, 2),
                    'potential_gain': round(potential_gain, 2),
                    'risk_reward_ratio': round(potential_gain / (stop_loss - entry_price), 2) if stop_loss > entry_price else 0,
                    'confidence': self._calculate_entry_confidence(daily_data, hourly_data, 'short'),
                    'timeframe': 'Intraday to 1-3 days',
                    'reasoning': f'Support at ${target_price:.2f} offers ${potential_gain:.2f} downside potential'
                })
        
        # Breakout signals
        breakout_signals = self._identify_breakout_signals(hourly_data, minute_data, current_price)
        signals.extend(breakout_signals)
        
        # Sort by confidence and potential gain
        signals = sorted(signals, key=lambda x: (x['confidence'], x['potential_gain']), reverse=True)
        
        return {
            'signals': signals[:3],  # Top 3 signals
            'total_signals': len(signals),
            'best_signal': signals[0] if signals else None
        }
    
    def _identify_breakout_signals(self, hourly_data: pd.DataFrame, minute_data: pd.DataFrame, 
                                 current_price: float) -> List[Dict]:
        """Identify breakout signals for quick $5-10 gains"""
        signals = []
        
        # Look for consolidation patterns in hourly data
        if len(hourly_data) >= 20:
            recent_highs = hourly_data['High'].tail(20)
            recent_lows = hourly_data['Low'].tail(20)
            
            # Check for tight consolidation (range < 3%)
            range_pct = ((recent_highs.max() - recent_lows.min()) / current_price) * 100
            
            if range_pct < 3:  # Tight consolidation
                breakout_high = recent_highs.max()
                breakout_low = recent_lows.min()
                
                # Upward breakout signal
                if current_price > breakout_high * 0.999:  # Near breakout high
                    target = breakout_high + (breakout_high - breakout_low)  # Measured move
                    potential_gain = target - current_price
                    
                    if 5 <= potential_gain <= 15:
                        signals.append({
                            'type': 'Breakout Long',
                            'entry_price': round(breakout_high + 0.10, 2),
                            'target_price': round(target, 2),
                            'stop_loss': round(breakout_low, 2),
                            'potential_gain': round(potential_gain, 2),
                            'risk_reward_ratio': round(potential_gain / (breakout_high - breakout_low), 2),
                            'confidence': 75,
                            'timeframe': 'Intraday',
                            'reasoning': f'Breakout above ${breakout_high:.2f} consolidation with ${potential_gain:.2f} target'
                        })
                
                # Downward breakout signal
                if current_price < breakout_low * 1.001:  # Near breakout low
                    target = breakout_low - (breakout_high - breakout_low)  # Measured move
                    potential_gain = current_price - target
                    
                    if 5 <= potential_gain <= 15:
                        signals.append({
                            'type': 'Breakout Short',
                            'entry_price': round(breakout_low - 0.10, 2),
                            'target_price': round(target, 2),
                            'stop_loss': round(breakout_high, 2),
                            'potential_gain': round(potential_gain, 2),
                            'risk_reward_ratio': round(potential_gain / (breakout_high - breakout_low), 2),
                            'confidence': 75,
                            'timeframe': 'Intraday',
                            'reasoning': f'Breakdown below ${breakout_low:.2f} consolidation with ${potential_gain:.2f} target'
                        })
        
        return signals
    
    def _calculate_entry_confidence(self, daily_data: pd.DataFrame, hourly_data: pd.DataFrame, direction: str) -> int:
        """Calculate confidence level for entry signal"""
        confidence = 50  # Base confidence
        
        # Volume confirmation
        if len(daily_data) >= 5:
            recent_volume = daily_data['Volume'].tail(5).mean()
            avg_volume = daily_data['Volume'].tail(20).mean()
            
            if recent_volume > avg_volume * 1.2:  # Above average volume
                confidence += 15
        
        # Trend alignment
        if len(hourly_data) >= 10:
            hourly_close = hourly_data['Close'].values
            hourly_trend = np.polyfit(range(10), hourly_close[-10:], 1)[0]
            
            if direction == 'long' and hourly_trend > 0:
                confidence += 20
            elif direction == 'short' and hourly_trend < 0:
                confidence += 20
        
        # RSI confirmation
        if len(daily_data) >= 14:
            df = pd.DataFrame({'Close': daily_data['Close']})
            rsi_series = ta.rsi(df['Close'], length=14)
            rsi = rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50
            
            if direction == 'long' and 30 <= rsi <= 50:  # Not overbought
                confidence += 10
            elif direction == 'short' and 50 <= rsi <= 70:  # Not oversold
                confidence += 10
        
        return min(confidence, 95)  # Cap at 95%
    
    def _analyze_highs_lows(self, data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Analyze current day/week highs and lows"""
        
        # Current day high/low
        today_data = data.tail(1)
        today_high = today_data['High'].iloc[0] if len(today_data) > 0 else current_price
        today_low = today_data['Low'].iloc[0] if len(today_data) > 0 else current_price
        
        # Previous day high/low
        prev_day_data = data.tail(2).head(1) if len(data) >= 2 else today_data
        prev_high = prev_day_data['High'].iloc[0] if len(prev_day_data) > 0 else current_price
        prev_low = prev_day_data['Low'].iloc[0] if len(prev_day_data) > 0 else current_price
        
        # Week high/low (last 5 trading days)
        week_data = data.tail(5)
        week_high = week_data['High'].max()
        week_low = week_data['Low'].min()
        
        # Calculate position within ranges
        today_position = ((current_price - today_low) / (today_high - today_low)) * 100 if today_high != today_low else 50
        week_position = ((current_price - week_low) / (week_high - week_low)) * 100 if week_high != week_low else 50
        
        return {
            'current_day': {
                'high': round(today_high, 2),
                'low': round(today_low, 2),
                'range': round(today_high - today_low, 2),
                'position_pct': round(today_position, 1)
            },
            'previous_day': {
                'high': round(prev_high, 2),
                'low': round(prev_low, 2),
                'range': round(prev_high - prev_low, 2)
            },
            'current_week': {
                'high': round(week_high, 2),
                'low': round(week_low, 2),
                'range': round(week_high - week_low, 2),
                'position_pct': round(week_position, 1)
            },
            'key_levels': {
                'resistance_today': round(today_high, 2),
                'support_today': round(today_low, 2),
                'resistance_week': round(week_high, 2),
                'support_week': round(week_low, 2)
            }
        }
    
    def _calculate_technical_score(self, daily: Dict, four_hour: Dict, hourly: Dict, minute: Dict) -> Dict[str, Any]:
        """Calculate overall technical score"""
        
        scores = []
        weights = {'Daily': 0.4, '4-Hour': 0.3, '1-Hour': 0.2, '15-Min': 0.1}
        
        for timeframe_data, weight in [(daily, weights['Daily']), (four_hour, weights['4-Hour']), 
                                     (hourly, weights['1-Hour']), (minute, weights['15-Min'])]:
            if 'error' not in timeframe_data:
                trend = timeframe_data.get('trend', 'Neutral')
                signal_strength = timeframe_data.get('signal_strength', 'Moderate')
                
                # Convert to numeric score
                trend_score = {'Bullish': 80, 'Neutral': 50, 'Bearish': 20}.get(trend, 50)
                strength_score = {
                    'Very Strong': 90, 'Strong': 75, 'Moderate': 50, 
                    'Weak': 25, 'Very Weak': 10
                }.get(signal_strength, 50)
                
                weighted_score = (trend_score + strength_score) / 2 * weight
                scores.append(weighted_score)
        
        overall_score = sum(scores) if scores else 50
        
        # Determine recommendation
        if overall_score >= 70:
            recommendation = 'Strong Buy'
        elif overall_score >= 60:
            recommendation = 'Buy'
        elif overall_score >= 40:
            recommendation = 'Hold'
        elif overall_score >= 30:
            recommendation = 'Sell'
        else:
            recommendation = 'Strong Sell'
        
        return {
            'overall_score': round(overall_score, 1),
            'recommendation': recommendation,
            'timeframe_alignment': len([s for s in scores if s > 50]) / len(scores) * 100 if scores else 0
        }
    
    def _get_recommendation(self, technical_score: Dict, entry_signals: Dict) -> Dict[str, Any]:
        """Get final trading recommendation"""
        
        best_signal = entry_signals.get('best_signal')
        overall_score = technical_score.get('overall_score', 50)
        
        if best_signal and best_signal['confidence'] >= 70 and best_signal['potential_gain'] >= 5:
            action = f"{best_signal['type']} Signal"
            confidence = best_signal['confidence']
            reasoning = f"Technical analysis suggests {best_signal['type'].lower()} opportunity with ${best_signal['potential_gain']:.2f} potential gain"
        elif overall_score >= 70:
            action = 'Buy'
            confidence = int(overall_score)
            reasoning = "Strong bullish technical signals across multiple timeframes"
        elif overall_score <= 30:
            action = 'Sell'
            confidence = int(100 - overall_score)
            reasoning = "Strong bearish technical signals across multiple timeframes"
        else:
            action = 'Hold'
            confidence = 50
            reasoning = "Mixed technical signals, wait for clearer direction"
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'timeframe': 'Intraday to 3 days' if best_signal else 'Medium term'
        }
    
    def get_grok_enhanced_recommendation(self, symbol: str, technical_data: Dict) -> Dict[str, Any]:
        """Get Grok-4 enhanced trading recommendation with live intelligence"""
        try:
            # Import AI engine for Grok-4 analysis
            from core.ai_engine import ai_engine
            
            # Prepare technical summary for Grok-4
            current_price = technical_data.get('current_price', 0)
            support_resistance = technical_data.get('support_resistance', {})
            entry_signals = technical_data.get('entry_signals', {})
            timeframe_analysis = technical_data.get('timeframe_analysis', {})
            high_low_analysis = technical_data.get('high_low_analysis', {})
            
            # Create comprehensive technical summary
            technical_summary = {
                'symbol': symbol,
                'current_price': current_price,
                'nearest_resistance': support_resistance.get('nearest_resistance'),
                'nearest_support': support_resistance.get('nearest_support'),
                'daily_trend': timeframe_analysis.get('daily', {}).get('trend', 'Neutral'),
                'hourly_trend': timeframe_analysis.get('1hour', {}).get('trend', 'Neutral'),
                'daily_rsi': timeframe_analysis.get('daily', {}).get('indicators', {}).get('rsi'),
                'week_high': high_low_analysis.get('current_week', {}).get('high'),
                'week_low': high_low_analysis.get('current_week', {}).get('low'),
                'technical_score': technical_data.get('technical_score', {}).get('overall_score', 50),
                'best_signal': entry_signals.get('best_signal')
            }
            
            # Get live market intelligence for enhanced context
            live_intel = ai_engine.get_real_time_market_intelligence(symbol)
            
            # Create enhanced prompt for Grok-4
            prompt = f"""
            As an expert institutional trader with access to LIVE PRODUCTION MARKET DATA, analyze {symbol} and provide REAL TRADING RECOMMENDATIONS for actual positions.
            
            CRITICAL: This is LIVE PRODUCTION TRADING INTELLIGENCE - not simulation or demo. Provide actionable recommendations for real money trading decisions.
            
            CURRENT TECHNICAL DATA:
            - Current Price: ${current_price}
            - Nearest Resistance: ${support_resistance.get('nearest_resistance', 'N/A')}
            - Nearest Support: ${support_resistance.get('nearest_support', 'N/A')}
            - Daily Trend: {timeframe_analysis.get('daily', {}).get('trend', 'Neutral')}
            - Hourly Trend: {timeframe_analysis.get('1hour', {}).get('trend', 'Neutral')}
            - Daily RSI: {timeframe_analysis.get('daily', {}).get('indicators', {}).get('rsi', 'N/A')}
            - Week High: ${high_low_analysis.get('current_week', {}).get('high', 'N/A')}
            - Week Low: ${high_low_analysis.get('current_week', {}).get('low', 'N/A')}
            - Technical Score: {technical_data.get('technical_score', {}).get('overall_score', 50)}/100
            
            LIVE MARKET CONTEXT:
            - Social Sentiment: {live_intel.get('social_sentiment', 'neutral')}
            - Breaking News: {len(live_intel.get('breaking_news', []))} items
            - Overall Impact: {live_intel.get('overall_impact', 'neutral')}
            
            Provide a specific trading recommendation with:
            1. EXACT entry price levels
            2. EXACT target prices (for $5-15 gains)
            3. EXACT stop-loss levels
            4. Specific reasoning based on technical levels AND live market sentiment
            5. Expected timeframe for the trade
            
            Format your response as a professional trader would - be specific about price levels and reasoning.
            """
            
            # Get Grok-4 enhanced recommendation
            grok_response = ai_engine.get_trading_recommendation(prompt)
            
            return {
                'grok_enhanced': True,
                'recommendation': grok_response,
                'technical_summary': technical_summary,
                'live_context': {
                    'social_sentiment': live_intel.get('social_sentiment', 'neutral'),
                    'breaking_news_count': len(live_intel.get('breaking_news', [])),
                    'overall_impact': live_intel.get('overall_impact', 'neutral'),
                    'sources_used': live_intel.get('sources_used', 0)
                }
            }
            
        except Exception as e:
            # Fallback to basic recommendation if Grok-4 fails
            return {
                'grok_enhanced': False,
                'error': f"Grok-4 enhancement failed: {str(e)}",
                'fallback_recommendation': self._get_recommendation(
                    technical_data.get('technical_score', {}), 
                    technical_data.get('entry_signals', {})
                )
            }

# Global technical analysis engine instance
technical_engine = TechnicalAnalysisEngine()
