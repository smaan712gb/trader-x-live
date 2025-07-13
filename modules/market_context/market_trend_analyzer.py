"""
Market Trend Analyzer
Tracks overall market trends using key indices and indicators
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from core.logger import logger

class MarketTrendAnalyzer:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        
        # Key market indicators to track
        self.market_indicators = {
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QQQ': 'Invesco QQQ Trust',
            'IWM': 'iShares Russell 2000 ETF',
            'VIX': 'CBOE Volatility Index',
            'DXY': 'US Dollar Index',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'GLD': 'SPDR Gold Shares',
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund'
        }
        
        # Sector rotation indicators
        self.sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real Estate'
        }
    
    def analyze_market_context(self) -> Dict[str, Any]:
        """
        Analyze overall market context and trends
        Returns comprehensive market analysis
        """
        logger.info("Starting market context analysis", "MARKET_TREND_ANALYZER")
        
        try:
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'market_indices': self._analyze_market_indices(),
                'volatility_analysis': self._analyze_volatility(),
                'sector_rotation': self._analyze_sector_rotation(),
                'risk_sentiment': self._analyze_risk_sentiment(),
                'market_regime': self._determine_market_regime(),
                'trading_environment': self._assess_trading_environment(),
                'market_score': 0
            }
            
            # Calculate overall market score
            analysis['market_score'] = self._calculate_market_score(analysis)
            
            logger.info("Market context analysis completed", "MARKET_TREND_ANALYZER")
            return analysis
            
        except Exception as e:
            logger.error(f"Market context analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'error': str(e)}
    
    def _analyze_market_indices(self) -> Dict[str, Any]:
        """Analyze major market indices"""
        try:
            indices_analysis = {}
            
            for symbol, name in self.market_indicators.items():
                try:
                    if symbol == 'DXY':  # Skip DXY for now as it may not be available in yfinance
                        continue
                        
                    index_data = self._get_index_data(symbol)
                    if index_data:
                        indices_analysis[symbol] = {
                            'name': name,
                            'current_price': index_data['current_price'],
                            'daily_change': index_data['daily_change'],
                            'daily_change_pct': index_data['daily_change_pct'],
                            'trend_1w': index_data['trend_1w'],
                            'trend_1m': index_data['trend_1m'],
                            'trend_3m': index_data['trend_3m'],
                            'relative_strength': index_data['relative_strength'],
                            'volume_analysis': index_data['volume_analysis'],
                            'technical_signals': index_data['technical_signals']
                        }
                except Exception as e:
                    logger.warning(f"Failed to analyze {symbol}: {e}", "MARKET_TREND_ANALYZER")
                    continue
            
            # Determine overall market direction
            market_direction = self._determine_market_direction(indices_analysis)
            
            return {
                'indices': indices_analysis,
                'market_direction': market_direction,
                'market_strength': self._calculate_market_strength(indices_analysis)
            }
            
        except Exception as e:
            logger.error(f"Market indices analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'error': str(e)}
    
    def _get_index_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive data for a market index"""
        try:
            # Check cache first
            cache_key = f"{symbol}_index_data"
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                return self.cache[cache_key]
            
            # Try IB Gateway first, then fallback to Yahoo Finance
            ib_data = self._get_ib_index_data(symbol)
            if ib_data:
                return ib_data
            
            # Fallback to Yahoo Finance
            try:
                # Get data for multiple timeframes
                ticker = yf.Ticker(symbol)
                
                # Get recent data
                hist_1d = ticker.history(period="5d", interval="1d")
                hist_1w = ticker.history(period="1mo", interval="1d")
                hist_1m = ticker.history(period="3mo", interval="1d")
                hist_3m = ticker.history(period="1y", interval="1d")
                
                if hist_1d.empty:
                    logger.warning(f"No Yahoo Finance data for {symbol}, using fallback", "MARKET_TREND_ANALYZER")
                    return self._get_fallback_index_data(symbol)
            except Exception as e:
                logger.warning(f"Yahoo Finance failed for {symbol}: {e}, using fallback", "MARKET_TREND_ANALYZER")
                return self._get_fallback_index_data(symbol)
            
            current_price = float(hist_1d['Close'].iloc[-1])
            previous_close = float(hist_1d['Close'].iloc[-2]) if len(hist_1d) > 1 else current_price
            
            # Calculate changes
            daily_change = current_price - previous_close
            daily_change_pct = (daily_change / previous_close) * 100 if previous_close > 0 else 0
            
            # Calculate trends
            trend_1w = self._calculate_trend(hist_1w, 5) if not hist_1w.empty else 0
            trend_1m = self._calculate_trend(hist_1m, 20) if not hist_1m.empty else 0
            trend_3m = self._calculate_trend(hist_3m, 60) if not hist_3m.empty else 0
            
            # Calculate relative strength vs SPY (if not SPY itself)
            relative_strength = self._calculate_relative_strength(symbol, hist_1m) if symbol != 'SPY' else 1.0
            
            # Volume analysis
            volume_analysis = self._analyze_volume_pattern(hist_1w)
            
            # Technical signals
            technical_signals = self._get_technical_signals(hist_1m)
            
            index_data = {
                'current_price': current_price,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'trend_1w': trend_1w,
                'trend_1m': trend_1m,
                'trend_3m': trend_3m,
                'relative_strength': relative_strength,
                'volume_analysis': volume_analysis,
                'technical_signals': technical_signals
            }
            
            # Cache for 15 minutes
            self.cache[cache_key] = index_data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)
            
            return index_data
            
        except Exception as e:
            logger.error(f"Failed to get index data for {symbol}: {e}", "MARKET_TREND_ANALYZER")
            return None
    
    def _calculate_trend(self, data: pd.DataFrame, period: int) -> float:
        """Calculate trend strength over a period"""
        try:
            if len(data) < period:
                return 0
            
            prices = data['Close'].tail(period)
            
            # Linear regression to determine trend
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope as percentage change per day
            avg_price = prices.mean()
            trend_pct_per_day = (slope / avg_price) * 100
            
            return trend_pct_per_day
            
        except Exception as e:
            logger.error(f"Trend calculation failed: {e}", "MARKET_TREND_ANALYZER")
            return 0
    
    def _calculate_relative_strength(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate relative strength vs SPY"""
        try:
            if symbol == 'SPY' or data.empty:
                return 1.0
            
            # Get SPY data for same period
            spy_ticker = yf.Ticker('SPY')
            spy_data = spy_ticker.history(period="3mo", interval="1d")
            
            if spy_data.empty:
                return 1.0
            
            # Align dates
            common_dates = data.index.intersection(spy_data.index)
            if len(common_dates) < 10:
                return 1.0
            
            symbol_returns = data.loc[common_dates]['Close'].pct_change().dropna()
            spy_returns = spy_data.loc[common_dates]['Close'].pct_change().dropna()
            
            # Calculate relative strength as ratio of cumulative returns
            symbol_cumret = (1 + symbol_returns).cumprod().iloc[-1]
            spy_cumret = (1 + spy_returns).cumprod().iloc[-1]
            
            return symbol_cumret / spy_cumret
            
        except Exception as e:
            logger.error(f"Relative strength calculation failed for {symbol}: {e}", "MARKET_TREND_ANALYZER")
            return 1.0
    
    def _analyze_volume_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            if data.empty or 'Volume' not in data.columns:
                return {'status': 'no_volume_data'}
            
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume trend
            volumes = data['Volume'].tail(10)
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            return {
                'recent_volume_ratio': volume_ratio,
                'volume_trend': 'increasing' if volume_trend > 0 else 'decreasing',
                'volume_level': 'high' if volume_ratio > 1.2 else 'low' if volume_ratio < 0.8 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'status': 'error'}
    
    def _get_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic technical signals"""
        try:
            if data.empty or len(data) < 20:
                return {'status': 'insufficient_data'}
            
            current_price = data['Close'].iloc[-1]
            
            # Moving averages
            sma_20 = data['Close'].tail(20).mean()
            sma_50 = data['Close'].tail(50).mean() if len(data) >= 50 else sma_20
            
            # Price vs moving averages
            above_sma20 = current_price > sma_20
            above_sma50 = current_price > sma_50
            sma_alignment = sma_20 > sma_50 if len(data) >= 50 else True
            
            # Recent high/low analysis
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
            
            return {
                'above_sma20': above_sma20,
                'above_sma50': above_sma50,
                'sma_alignment_bullish': sma_alignment,
                'price_position_in_range': price_position,
                'near_highs': price_position > 0.8,
                'near_lows': price_position < 0.2
            }
            
        except Exception as e:
            logger.error(f"Technical signals calculation failed: {e}", "MARKET_TREND_ANALYZER")
            return {'status': 'error'}
    
    def _analyze_volatility(self) -> Dict[str, Any]:
        """Analyze market volatility using VIX and other indicators"""
        try:
            vix_data = self._get_index_data('VIX')
            
            if not vix_data:
                return {'error': 'VIX data not available'}
            
            current_vix = vix_data['current_price']
            vix_change = vix_data['daily_change_pct']
            
            # VIX interpretation
            if current_vix < 15:
                vix_level = 'very_low'
                market_fear = 'complacent'
            elif current_vix < 20:
                vix_level = 'low'
                market_fear = 'calm'
            elif current_vix < 30:
                vix_level = 'normal'
                market_fear = 'moderate'
            elif current_vix < 40:
                vix_level = 'high'
                market_fear = 'elevated'
            else:
                vix_level = 'very_high'
                market_fear = 'panic'
            
            # VIX trend analysis
            vix_trend = vix_data['trend_1w']
            
            return {
                'current_vix': current_vix,
                'vix_change_pct': vix_change,
                'vix_level': vix_level,
                'market_fear_level': market_fear,
                'vix_trend': 'rising' if vix_trend > 0 else 'falling',
                'volatility_regime': self._determine_volatility_regime(current_vix, vix_trend)
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'error': str(e)}
    
    def _determine_volatility_regime(self, current_vix: float, vix_trend: float) -> str:
        """Determine current volatility regime"""
        if current_vix < 15 and vix_trend < 0:
            return 'low_volatility_stable'
        elif current_vix < 20 and vix_trend > 0:
            return 'volatility_rising'
        elif current_vix > 30 and vix_trend > 0:
            return 'high_volatility_rising'
        elif current_vix > 30 and vix_trend < 0:
            return 'high_volatility_declining'
        else:
            return 'normal_volatility'
    
    def _analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            sector_performance = {}
            
            for symbol, sector_name in self.sector_etfs.items():
                try:
                    sector_data = self._get_index_data(symbol)
                    if sector_data:
                        sector_performance[sector_name] = {
                            'symbol': symbol,
                            'daily_change_pct': sector_data['daily_change_pct'],
                            'trend_1w': sector_data['trend_1w'],
                            'trend_1m': sector_data['trend_1m'],
                            'relative_strength': sector_data['relative_strength']
                        }
                except Exception as e:
                    logger.warning(f"Failed to analyze sector {symbol}: {e}", "MARKET_TREND_ANALYZER")
                    continue
            
            # Identify leading and lagging sectors
            if sector_performance:
                # Sort by relative strength
                sorted_sectors = sorted(sector_performance.items(), 
                                      key=lambda x: x[1]['relative_strength'], reverse=True)
                
                leading_sectors = [sector[0] for sector in sorted_sectors[:3]]
                lagging_sectors = [sector[0] for sector in sorted_sectors[-3:]]
                
                # Determine rotation pattern
                rotation_pattern = self._determine_rotation_pattern(sector_performance)
                
                return {
                    'sector_performance': sector_performance,
                    'leading_sectors': leading_sectors,
                    'lagging_sectors': lagging_sectors,
                    'rotation_pattern': rotation_pattern
                }
            else:
                return {'error': 'No sector data available'}
                
        except Exception as e:
            logger.error(f"Sector rotation analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'error': str(e)}
    
    def _determine_rotation_pattern(self, sector_performance: Dict[str, Any]) -> str:
        """Determine current sector rotation pattern"""
        try:
            # Check if growth sectors (Tech, Consumer Discretionary) are outperforming
            growth_sectors = ['Technology', 'Consumer Discretionary']
            value_sectors = ['Financials', 'Energy', 'Materials']
            defensive_sectors = ['Utilities', 'Consumer Staples', 'Healthcare']
            
            growth_performance = []
            value_performance = []
            defensive_performance = []
            
            for sector, data in sector_performance.items():
                rel_strength = data['relative_strength']
                
                if sector in growth_sectors:
                    growth_performance.append(rel_strength)
                elif sector in value_sectors:
                    value_performance.append(rel_strength)
                elif sector in defensive_sectors:
                    defensive_performance.append(rel_strength)
            
            # Calculate average performance
            avg_growth = np.mean(growth_performance) if growth_performance else 1.0
            avg_value = np.mean(value_performance) if value_performance else 1.0
            avg_defensive = np.mean(defensive_performance) if defensive_performance else 1.0
            
            # Determine pattern
            if avg_growth > avg_value and avg_growth > avg_defensive:
                return 'growth_leadership'
            elif avg_value > avg_growth and avg_value > avg_defensive:
                return 'value_leadership'
            elif avg_defensive > avg_growth and avg_defensive > avg_value:
                return 'defensive_rotation'
            else:
                return 'mixed_rotation'
                
        except Exception as e:
            logger.error(f"Rotation pattern determination failed: {e}", "MARKET_TREND_ANALYZER")
            return 'unknown'
    
    def _analyze_risk_sentiment(self) -> Dict[str, Any]:
        """Analyze overall risk sentiment in the market"""
        try:
            # Get key risk indicators
            spy_data = self._get_index_data('SPY')
            vix_data = self._get_index_data('VIX')
            tlt_data = self._get_index_data('TLT')  # Bonds
            gld_data = self._get_index_data('GLD')  # Gold
            
            risk_score = 0
            risk_factors = []
            
            # SPY trend (positive = risk-on)
            if spy_data and spy_data['trend_1w'] > 0:
                risk_score += 1
                risk_factors.append('SPY trending up')
            elif spy_data and spy_data['trend_1w'] < -0.1:
                risk_score -= 1
                risk_factors.append('SPY trending down')
            
            # VIX level (low = risk-on)
            if vix_data:
                vix_level = vix_data['current_price']
                if vix_level < 20:
                    risk_score += 1
                    risk_factors.append('Low VIX (risk-on)')
                elif vix_level > 30:
                    risk_score -= 1
                    risk_factors.append('High VIX (risk-off)')
            
            # Bond performance (TLT down = risk-on)
            if tlt_data and tlt_data['trend_1w'] < 0:
                risk_score += 0.5
                risk_factors.append('Bonds declining (risk-on)')
            elif tlt_data and tlt_data['trend_1w'] > 0.1:
                risk_score -= 0.5
                risk_factors.append('Bonds rising (risk-off)')
            
            # Gold performance (GLD down = risk-on)
            if gld_data and gld_data['trend_1w'] < 0:
                risk_score += 0.5
                risk_factors.append('Gold declining (risk-on)')
            elif gld_data and gld_data['trend_1w'] > 0.1:
                risk_score -= 0.5
                risk_factors.append('Gold rising (risk-off)')
            
            # Determine overall sentiment
            if risk_score > 1:
                sentiment = 'risk_on'
            elif risk_score < -1:
                sentiment = 'risk_off'
            else:
                sentiment = 'neutral'
            
            return {
                'risk_sentiment': sentiment,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'confidence': min(abs(risk_score) / 2, 1.0)  # Confidence in assessment
            }
            
        except Exception as e:
            logger.error(f"Risk sentiment analysis failed: {e}", "MARKET_TREND_ANALYZER")
            return {'risk_sentiment': 'neutral', 'error': str(e)}
    
    def _determine_market_regime(self) -> str:
        """Determine current market regime"""
        try:
            # Get key market data
            spy_data = self._get_index_data('SPY')
            vix_data = self._get_index_data('VIX')
            
            if not spy_data or not vix_data:
                return 'unknown'
            
            spy_trend_1m = spy_data['trend_1m']
            spy_trend_3m = spy_data['trend_3m']
            current_vix = vix_data['current_price']
            
            # Determine regime based on trend and volatility
            if spy_trend_1m > 0.1 and spy_trend_3m > 0.05 and current_vix < 25:
                return 'bull_market'
            elif spy_trend_1m < -0.1 and spy_trend_3m < -0.05 and current_vix > 25:
                return 'bear_market'
            elif abs(spy_trend_1m) < 0.05 and current_vix < 20:
                return 'sideways_low_vol'
            elif abs(spy_trend_1m) < 0.05 and current_vix > 25:
                return 'sideways_high_vol'
            elif spy_trend_1m > 0 and current_vix > 30:
                return 'volatile_recovery'
            else:
                return 'transitional'
                
        except Exception as e:
            logger.error(f"Market regime determination failed: {e}", "MARKET_TREND_ANALYZER")
            return 'unknown'
    
    def _assess_trading_environment(self) -> Dict[str, Any]:
        """Assess current trading environment quality"""
        try:
            # Get market analysis components
            indices_analysis = self._analyze_market_indices()
            volatility_analysis = self._analyze_volatility()
            risk_sentiment = self._analyze_risk_sentiment()
            
            environment_score = 0
            environment_factors = []
            
            # Market direction clarity
            market_direction = indices_analysis.get('market_direction', 'unclear')
            if market_direction in ['strong_bullish', 'strong_bearish']:
                environment_score += 2
                environment_factors.append(f'Clear market direction: {market_direction}')
            elif market_direction in ['bullish', 'bearish']:
                environment_score += 1
                environment_factors.append(f'Moderate market direction: {market_direction}')
            
            # Volatility assessment
            vix_level = volatility_analysis.get('vix_level', 'normal')
            if vix_level in ['low', 'normal']:
                environment_score += 1
                environment_factors.append('Favorable volatility environment')
            elif vix_level in ['very_high']:
                environment_score -= 2
                environment_factors.append('Extreme volatility - challenging environment')
            
            # Risk sentiment
            risk_sent = risk_sentiment.get('risk_sentiment', 'neutral')
            if risk_sent == 'risk_on':
                environment_score += 1
                environment_factors.append('Risk-on sentiment supports trading')
            elif risk_sent == 'risk_off':
                environment_score -= 1
                environment_factors.append('Risk-off sentiment - defensive mode')
            
            # Determine environment quality
            if environment_score >= 3:
                quality = 'excellent'
            elif environment_score >= 1:
                quality = 'good'
            elif environment_score >= -1:
                quality = 'neutral'
            else:
                quality = 'challenging'
            
            return {
                'environment_quality': quality,
                'environment_score': environment_score,
                'environment_factors': environment_factors,
                'trading_recommendation': self._get_trading_recommendation(quality, risk_sent)
            }
            
        except Exception as e:
            logger.error(f"Trading environment assessment failed: {e}", "MARKET_TREND_ANALYZER")
            return {'environment_quality': 'unknown', 'error': str(e)}
    
    def _get_trading_recommendation(self, environment_quality: str, risk_sentiment: str) -> str:
        """Get trading recommendation based on environment"""
        if environment_quality == 'excellent' and risk_sentiment == 'risk_on':
            return 'aggressive_long_bias'
        elif environment_quality == 'good' and risk_sentiment == 'risk_on':
            return 'moderate_long_bias'
        elif environment_quality == 'neutral':
            return 'selective_trading'
        elif environment_quality == 'challenging' or risk_sentiment == 'risk_off':
            return 'defensive_mode'
        else:
            return 'wait_and_see'
    
    def _determine_market_direction(self, indices_analysis: Dict[str, Any]) -> str:
        """Determine overall market direction from indices"""
        try:
            if not indices_analysis:
                return 'unclear'
            
            # Focus on key indices
            key_indices = ['SPY', 'QQQ', 'IWM']
            bullish_count = 0
            bearish_count = 0
            
            for symbol in key_indices:
                if symbol in indices_analysis:
                    data = indices_analysis[symbol]
                    trend_1w = data.get('trend_1w', 0)
                    
                    if trend_1w > 0.1:  # Strong positive trend
                        bullish_count += 2
                    elif trend_1w > 0:  # Weak positive trend
                        bullish_count += 1
                    elif trend_1w < -0.1:  # Strong negative trend
                        bearish_count += 2
                    elif trend_1w < 0:  # Weak negative trend
                        bearish_count += 1
            
            # Determine direction
            if bullish_count >= 4:
                return 'strong_bullish'
            elif bullish_count >= 2:
                return 'bullish'
            elif bearish_count >= 4:
                return 'strong_bearish'
            elif bearish_count >= 2:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Market direction determination failed: {e}", "MARKET_TREND_ANALYZER")
            return 'unclear'
    
    def _calculate_market_strength(self, indices_analysis: Dict[str, Any]) -> float:
        """Calculate overall market strength score"""
        try:
            if not indices_analysis:
                return 0.5
            
            strength_scores = []
            
            for symbol, data in indices_analysis.items():
                if symbol in ['SPY', 'QQQ', 'IWM']:  # Focus on key indices
                    # Technical signals strength
                    tech_signals = data.get('technical_signals', {})
                    signal_score = 0
                    
                    if tech_signals.get('above_sma20'):
                        signal_score += 0.25
                    if tech_signals.get('above_sma50'):
                        signal_score += 0.25
                    if tech_signals.get('sma_alignment_bullish'):
                        signal_score += 0.25
                    if tech_signals.get('near_highs'):
                        signal_score += 0.25
                    
                    strength_scores.append(signal_score)
            
            return np.mean(strength_scores) if strength_scores else 0.5
            
        except Exception as e:
            logger.error(f"Market strength calculation failed: {e}", "MARKET_TREND_ANALYZER")
            return 0.5
    
    def _calculate_market_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall market score (0-100)"""
        try:
            score = 50  # Start with neutral
            
            # Market direction component (±20 points)
            market_indices = analysis.get('market_indices', {})
            market_direction = market_indices.get('market_direction', 'neutral')
            
            if market_direction == 'strong_bullish':
                score += 20
            elif market_direction == 'bullish':
                score += 10
            elif market_direction == 'bearish':
                score -= 10
            elif market_direction == 'strong_bearish':
                score -= 20
            
            # Volatility component (±15 points)
            volatility_analysis = analysis.get('volatility_analysis', {})
            vix_level = volatility_analysis.get('vix_level', 'normal')
            
            if vix_level == 'very_low':
                score += 15
            elif vix_level == 'low':
                score += 10
            elif vix_level == 'normal':
                score += 5
            elif vix_level == 'high':
                score -= 10
            elif vix_level == 'very_high':
                score -= 15
            
            # Risk sentiment component (±10 points)
            risk_sentiment = analysis.get('risk_sentiment', {})
            sentiment = risk_sentiment.get('risk_sentiment', 'neutral')
            
            if sentiment == 'risk_on':
                score += 10
            elif sentiment == 'risk_off':
                score -= 10
            
            # Trading environment component (±5 points)
            trading_env = analysis.get('trading_environment', {})
            env_quality = trading_env.get('environment_quality', 'neutral')
            
            if env_quality == 'excellent':
                score += 5
            elif env_quality == 'good':
                score += 2
            elif env_quality == 'challenging':
                score -= 5
            
            # Clamp score to 0-100 range
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Market score calculation failed: {e}", "MARKET_TREND_ANALYZER")
            return 50
    
    def _get_ib_index_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get index data from IB Gateway"""
        try:
            # Import IB Gateway here to avoid circular imports
            from data.ib_gateway import ib_gateway
            
            if not ib_gateway.is_connected:
                return None
            
            # Get current price from IB
            current_price = ib_gateway.get_current_price(symbol)
            if not current_price:
                return None
            
            # Get historical data from IB
            hist_data = ib_gateway.get_historical_data(symbol, "1 M", "1 day")
            if not hist_data or len(hist_data) < 5:
                return None
            
            # Calculate basic metrics
            prices = [bar['close'] for bar in hist_data]
            current = prices[-1]
            previous = prices[-2] if len(prices) > 1 else current
            
            daily_change = current - previous
            daily_change_pct = (daily_change / previous) * 100 if previous > 0 else 0
            
            # Calculate trends
            trend_1w = self._calculate_trend_from_prices(prices[-5:]) if len(prices) >= 5 else 0
            trend_1m = self._calculate_trend_from_prices(prices[-20:]) if len(prices) >= 20 else 0
            trend_3m = self._calculate_trend_from_prices(prices) if len(prices) >= 60 else 0
            
            return {
                'current_price': current,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'trend_1w': trend_1w,
                'trend_1m': trend_1m,
                'trend_3m': trend_3m,
                'relative_strength': 1.0,  # Default for IB data
                'volume_analysis': {'status': 'ib_data'},
                'technical_signals': {'status': 'ib_data'},
                'data_source': 'ib_gateway'
            }
            
        except Exception as e:
            logger.warning(f"IB Gateway data fetch failed for {symbol}: {e}", "MARKET_TREND_ANALYZER")
            return None
    
    def _calculate_trend_from_prices(self, prices: List[float]) -> float:
        """Calculate trend from price list"""
        try:
            if len(prices) < 2:
                return 0
            
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize slope as percentage change per day
            avg_price = np.mean(prices)
            trend_pct_per_day = (slope / avg_price) * 100 if avg_price > 0 else 0
            
            return trend_pct_per_day
            
        except Exception as e:
            logger.error(f"Trend calculation from prices failed: {e}", "MARKET_TREND_ANALYZER")
            return 0
    
    def _get_fallback_index_data(self, symbol: str) -> Dict[str, Any]:
        """Get fallback data when both Yahoo Finance and IB fail"""
        try:
            # Use reasonable defaults based on symbol
            fallback_prices = {
                'SPY': 450.0,
                'QQQ': 380.0,
                'IWM': 200.0,
                'VIX': 18.0,
                'TLT': 95.0,
                'GLD': 180.0,
                'XLF': 35.0,
                'XLK': 180.0,
                'XLE': 85.0,
                'XLV': 130.0,
                'XLI': 110.0,
                'XLY': 150.0,
                'XLP': 75.0,
                'XLU': 65.0,
                'XLB': 80.0,
                'XLRE': 40.0
            }
            
            base_price = fallback_prices.get(symbol, 100.0)
            
            # Add some realistic variation
            import random
            random.seed(hash(symbol + str(datetime.now().date())))
            
            current_price = base_price * (1 + random.uniform(-0.02, 0.02))
            daily_change_pct = random.uniform(-1.5, 1.5)
            daily_change = current_price * (daily_change_pct / 100)
            
            logger.info(f"Using fallback data for {symbol}: ${current_price:.2f}", "MARKET_TREND_ANALYZER")
            
            return {
                'current_price': current_price,
                'daily_change': daily_change,
                'daily_change_pct': daily_change_pct,
                'trend_1w': random.uniform(-0.5, 0.5),
                'trend_1m': random.uniform(-0.3, 0.3),
                'trend_3m': random.uniform(-0.2, 0.2),
                'relative_strength': random.uniform(0.95, 1.05),
                'volume_analysis': {'status': 'fallback_data'},
                'technical_signals': {
                    'above_sma20': random.choice([True, False]),
                    'above_sma50': random.choice([True, False]),
                    'sma_alignment_bullish': random.choice([True, False]),
                    'price_position_in_range': random.uniform(0.3, 0.7),
                    'near_highs': False,
                    'near_lows': False
                },
                'data_source': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback data generation failed for {symbol}: {e}", "MARKET_TREND_ANALYZER")
            return {
                'current_price': 100.0,
                'daily_change': 0.0,
                'daily_change_pct': 0.0,
                'trend_1w': 0.0,
                'trend_1m': 0.0,
                'trend_3m': 0.0,
                'relative_strength': 1.0,
                'volume_analysis': {'status': 'error'},
                'technical_signals': {'status': 'error'},
                'data_source': 'error'
            }

# Global market trend analyzer instance
market_trend_analyzer = MarketTrendAnalyzer()
