"""
Smart Money Flow Tracker
Tracks institutional money flow through ETF holdings and dark pool activity
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
import json
from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from core.logger import logger

class MoneyFlowTracker:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
        
        # Key ETFs to track for institutional flow
        self.tracked_etfs = {
            'ARKK': 'ARK Innovation ETF',
            'ARKQ': 'ARK Autonomous Technology & Robotics ETF',
            'ARKW': 'ARK Next Generation Internet ETF',
            'ARKG': 'ARK Genomics Revolution ETF',
            'QQQ': 'Invesco QQQ Trust',
            'XLK': 'Technology Select Sector SPDR Fund',
            'SOXX': 'iShares Semiconductor ETF',
            'ICLN': 'iShares Global Clean Energy ETF',
            'FINX': 'Global X FinTech ETF',
            'CLOU': 'Global X Cloud Computing ETF'
        }
    
    def analyze_money_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze smart money flow for a given symbol
        Returns comprehensive money flow analysis
        """
        logger.info(f"Starting money flow analysis for {symbol}", "MONEY_FLOW_TRACKER")
        
        try:
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'etf_holdings_analysis': self._analyze_etf_holdings(symbol),
                'institutional_flow': self._analyze_institutional_flow(symbol),
                'volume_analysis': self._analyze_volume_patterns(symbol),
                'price_volume_correlation': self._analyze_price_volume_correlation(symbol),
                'dark_pool_estimate': self._estimate_dark_pool_activity(symbol),
                'smart_money_sentiment': self._determine_smart_money_sentiment(symbol),
                'flow_strength': self._calculate_flow_strength(symbol)
            }
            
            logger.info(f"Money flow analysis complete for {symbol}", "MONEY_FLOW_TRACKER")
            return analysis
            
        except Exception as e:
            logger.error(f"Money flow analysis failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _analyze_etf_holdings(self, symbol: str) -> Dict[str, Any]:
        """Analyze ETF holdings changes for the symbol"""
        try:
            holdings_analysis = {
                'tracked_etfs': {},
                'new_positions': [],
                'increased_positions': [],
                'decreased_positions': [],
                'total_etf_exposure': 0,
                'weighted_flow_score': 0
            }
            
            for etf_symbol, etf_name in self.tracked_etfs.items():
                try:
                    # Get ETF holdings data
                    etf_data = self._get_etf_holdings(etf_symbol, symbol)
                    
                    if etf_data:
                        holdings_analysis['tracked_etfs'][etf_symbol] = etf_data
                        
                        # Check for position changes
                        if etf_data.get('position_change') == 'new':
                            holdings_analysis['new_positions'].append({
                                'etf': etf_symbol,
                                'weight': etf_data.get('weight', 0),
                                'shares': etf_data.get('shares', 0)
                            })
                        elif etf_data.get('position_change') == 'increased':
                            holdings_analysis['increased_positions'].append({
                                'etf': etf_symbol,
                                'weight_change': etf_data.get('weight_change', 0),
                                'shares_change': etf_data.get('shares_change', 0)
                            })
                        elif etf_data.get('position_change') == 'decreased':
                            holdings_analysis['decreased_positions'].append({
                                'etf': etf_symbol,
                                'weight_change': etf_data.get('weight_change', 0),
                                'shares_change': etf_data.get('shares_change', 0)
                            })
                        
                        # Add to total exposure
                        holdings_analysis['total_etf_exposure'] += etf_data.get('weight', 0)
                        
                except Exception as e:
                    logger.warning(f"Failed to analyze {etf_symbol} holdings for {symbol}: {e}", "MONEY_FLOW_TRACKER")
                    continue
            
            # Calculate weighted flow score
            holdings_analysis['weighted_flow_score'] = self._calculate_etf_flow_score(holdings_analysis)
            
            return holdings_analysis
            
        except Exception as e:
            logger.error(f"ETF holdings analysis failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _get_etf_holdings(self, etf_symbol: str, target_symbol: str) -> Optional[Dict[str, Any]]:
        """Get ETF holdings data for a specific symbol"""
        try:
            # Check cache first
            cache_key = f"{etf_symbol}_{target_symbol}_holdings"
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                return self.cache[cache_key]
            
            # Get ETF data
            etf_ticker = yf.Ticker(etf_symbol)
            
            # Try to get holdings data (this is limited in yfinance)
            # In production, you'd use a dedicated ETF holdings API
            try:
                # Get basic ETF info
                etf_info = etf_ticker.info
                
                # Simulate holdings analysis based on correlation and volume
                # This is a simplified approach - real implementation would use actual holdings data
                holdings_data = self._simulate_holdings_analysis(etf_symbol, target_symbol)
                
                # Cache for 1 hour
                self.cache[cache_key] = holdings_data
                self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=1)
                
                return holdings_data
                
            except Exception as e:
                logger.warning(f"Could not get holdings data for {etf_symbol}: {e}", "MONEY_FLOW_TRACKER")
                return None
                
        except Exception as e:
            logger.error(f"ETF holdings lookup failed for {etf_symbol}: {e}", "MONEY_FLOW_TRACKER")
            return None
    
    def _simulate_holdings_analysis(self, etf_symbol: str, target_symbol: str) -> Dict[str, Any]:
        """
        Simulate holdings analysis based on price correlation and volume patterns
        In production, this would use actual ETF holdings APIs
        """
        try:
            # Get price data for both ETF and target symbol
            etf_data = yf.download(etf_symbol, period="30d", interval="1d")
            target_data = yf.download(target_symbol, period="30d", interval="1d")
            
            if etf_data.empty or target_data.empty:
                return None
            
            # Calculate correlation
            etf_returns = etf_data['Close'].pct_change().dropna()
            target_returns = target_data['Close'].pct_change().dropna()
            
            # Align the data
            common_dates = etf_returns.index.intersection(target_returns.index)
            etf_aligned = etf_returns.loc[common_dates]
            target_aligned = target_returns.loc[common_dates]
            
            correlation = etf_aligned.corr(target_aligned)
            
            # Analyze volume patterns
            recent_etf_volume = etf_data['Volume'].tail(5).mean()
            historical_etf_volume = etf_data['Volume'].head(20).mean()
            volume_ratio = recent_etf_volume / max(historical_etf_volume, 1)
            
            # Simulate position data based on correlation and known ETF focus
            estimated_weight = self._estimate_etf_weight(etf_symbol, target_symbol, correlation)
            
            # Determine position change based on volume and correlation patterns
            position_change = self._determine_position_change(correlation, volume_ratio)
            
            return {
                'weight': estimated_weight,
                'correlation': correlation,
                'volume_ratio': volume_ratio,
                'position_change': position_change,
                'confidence': min(abs(correlation) * 100, 95),  # Confidence in the analysis
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Holdings simulation failed for {etf_symbol}/{target_symbol}: {e}", "MONEY_FLOW_TRACKER")
            return None
    
    def _estimate_etf_weight(self, etf_symbol: str, target_symbol: str, correlation: float) -> float:
        """Estimate the weight of target symbol in ETF based on correlation and ETF type"""
        base_weights = {
            'ARKK': 0.05,  # ARK typically holds 5% positions
            'ARKQ': 0.04,
            'ARKW': 0.04,
            'ARKG': 0.04,
            'QQQ': 0.02,   # QQQ has broader holdings
            'XLK': 0.03,
            'SOXX': 0.06,  # More concentrated
            'ICLN': 0.04,
            'FINX': 0.05,
            'CLOU': 0.05
        }
        
        base_weight = base_weights.get(etf_symbol, 0.03)
        
        # Adjust based on correlation strength
        correlation_multiplier = abs(correlation) if not np.isnan(correlation) else 0.5
        
        return base_weight * correlation_multiplier
    
    def _determine_position_change(self, correlation: float, volume_ratio: float) -> str:
        """Determine if position is new, increased, decreased, or unchanged"""
        if np.isnan(correlation):
            return 'unknown'
        
        # High correlation + high volume suggests increased position
        if correlation > 0.7 and volume_ratio > 1.2:
            return 'increased'
        # Low correlation + high volume suggests decreased position
        elif correlation < 0.3 and volume_ratio > 1.2:
            return 'decreased'
        # Moderate correlation + very high volume suggests new position
        elif 0.4 < correlation < 0.8 and volume_ratio > 1.5:
            return 'new'
        else:
            return 'unchanged'
    
    def _calculate_etf_flow_score(self, holdings_analysis: Dict[str, Any]) -> float:
        """Calculate weighted flow score based on ETF position changes"""
        try:
            score = 0
            
            # Positive scores for new and increased positions
            for position in holdings_analysis['new_positions']:
                score += position['weight'] * 100  # New positions get full weight
            
            for position in holdings_analysis['increased_positions']:
                score += abs(position.get('weight_change', 0)) * 50  # Increases get half weight
            
            # Negative scores for decreased positions
            for position in holdings_analysis['decreased_positions']:
                score -= abs(position.get('weight_change', 0)) * 50
            
            return max(-100, min(100, score))  # Clamp between -100 and 100
            
        except Exception as e:
            logger.error(f"ETF flow score calculation failed: {e}", "MONEY_FLOW_TRACKER")
            return 0
    
    def _analyze_institutional_flow(self, symbol: str) -> Dict[str, Any]:
        """Analyze institutional flow patterns"""
        try:
            # Get recent price and volume data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty:
                return {'error': 'No price data available'}
            
            # Calculate institutional flow indicators
            analysis = {
                'volume_weighted_price': self._calculate_vwap(hist),
                'institutional_volume_ratio': self._calculate_institutional_volume_ratio(hist),
                'block_trade_activity': self._detect_block_trades(hist),
                'accumulation_distribution': self._calculate_accumulation_distribution(hist),
                'money_flow_index': self._calculate_money_flow_index(hist),
                'institutional_sentiment': 'neutral'
            }
            
            # Determine institutional sentiment
            analysis['institutional_sentiment'] = self._determine_institutional_sentiment(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Institutional flow analysis failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _calculate_vwap(self, hist: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (hist['High'] + hist['Low'] + hist['Close']) / 3
            vwap = (typical_price * hist['Volume']).sum() / hist['Volume'].sum()
            return float(vwap)
        except:
            return 0
    
    def _calculate_institutional_volume_ratio(self, hist: pd.DataFrame) -> float:
        """Calculate ratio of institutional volume (estimated)"""
        try:
            # Estimate institutional volume as volume on days with low volatility but high volume
            avg_volume = hist['Volume'].mean()
            high_volume_days = hist[hist['Volume'] > avg_volume * 1.2]
            
            if high_volume_days.empty:
                return 0
            
            # Calculate volatility for high volume days
            high_volume_days = high_volume_days.copy()
            high_volume_days['volatility'] = (high_volume_days['High'] - high_volume_days['Low']) / high_volume_days['Close']
            
            # Institutional volume is high volume + low volatility
            institutional_days = high_volume_days[high_volume_days['volatility'] < high_volume_days['volatility'].median()]
            
            if institutional_days.empty:
                return 0
            
            institutional_volume = institutional_days['Volume'].sum()
            total_volume = hist['Volume'].sum()
            
            return institutional_volume / total_volume
            
        except Exception as e:
            logger.error(f"Institutional volume ratio calculation failed: {e}", "MONEY_FLOW_TRACKER")
            return 0
    
    def _detect_block_trades(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential block trades (large institutional trades)"""
        try:
            avg_volume = hist['Volume'].mean()
            volume_threshold = avg_volume * 2  # 2x average volume
            
            block_trade_days = hist[hist['Volume'] > volume_threshold]
            
            return {
                'block_trade_count': len(block_trade_days),
                'avg_block_volume': block_trade_days['Volume'].mean() if not block_trade_days.empty else 0,
                'block_trade_frequency': len(block_trade_days) / len(hist),
                'recent_block_trades': len(block_trade_days.tail(5))
            }
            
        except Exception as e:
            logger.error(f"Block trade detection failed: {e}", "MONEY_FLOW_TRACKER")
            return {'block_trade_count': 0, 'avg_block_volume': 0, 'block_trade_frequency': 0, 'recent_block_trades': 0}
    
    def _calculate_accumulation_distribution(self, hist: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution Line"""
        try:
            # Money Flow Multiplier
            mfm = ((hist['Close'] - hist['Low']) - (hist['High'] - hist['Close'])) / (hist['High'] - hist['Low'])
            mfm = mfm.fillna(0)
            
            # Money Flow Volume
            mfv = mfm * hist['Volume']
            
            # Accumulation/Distribution Line
            ad_line = mfv.cumsum()
            
            # Return the trend (positive = accumulation, negative = distribution)
            return float(ad_line.iloc[-1] - ad_line.iloc[0])
            
        except Exception as e:
            logger.error(f"A/D calculation failed: {e}", "MONEY_FLOW_TRACKER")
            return 0
    
    def _calculate_money_flow_index(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index"""
        try:
            # Typical Price
            tp = (hist['High'] + hist['Low'] + hist['Close']) / 3
            
            # Raw Money Flow
            rmf = tp * hist['Volume']
            
            # Positive and Negative Money Flow
            positive_mf = rmf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
            negative_mf = rmf.where(tp < tp.shift(1), 0).rolling(window=period).sum()
            
            # Money Flow Index
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            
            return float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50
            
        except Exception as e:
            logger.error(f"MFI calculation failed: {e}", "MONEY_FLOW_TRACKER")
            return 50
    
    def _analyze_volume_patterns(self, symbol: str) -> Dict[str, Any]:
        """Analyze volume patterns for institutional activity"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="60d", interval="1d")
            
            if hist.empty:
                return {'error': 'No volume data available'}
            
            # Calculate volume metrics
            avg_volume_20d = hist['Volume'].tail(20).mean()
            avg_volume_60d = hist['Volume'].mean()
            recent_volume_trend = hist['Volume'].tail(10).mean() / hist['Volume'].head(10).mean()
            
            # Volume spikes
            volume_spikes = len(hist[hist['Volume'] > avg_volume_60d * 2])
            
            return {
                'avg_volume_20d': avg_volume_20d,
                'avg_volume_60d': avg_volume_60d,
                'volume_ratio_20_60': avg_volume_20d / avg_volume_60d,
                'recent_volume_trend': recent_volume_trend,
                'volume_spikes_60d': volume_spikes,
                'volume_consistency': hist['Volume'].std() / hist['Volume'].mean()  # Lower = more consistent
            }
            
        except Exception as e:
            logger.error(f"Volume pattern analysis failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _analyze_price_volume_correlation(self, symbol: str) -> Dict[str, Any]:
        """Analyze correlation between price and volume"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty:
                return {'error': 'No data available'}
            
            # Calculate price changes and volume
            price_changes = hist['Close'].pct_change().dropna()
            volume_changes = hist['Volume'].pct_change().dropna()
            
            # Align the data
            common_index = price_changes.index.intersection(volume_changes.index)
            price_aligned = price_changes.loc[common_index]
            volume_aligned = volume_changes.loc[common_index]
            
            # Calculate correlations
            overall_correlation = price_aligned.corr(volume_aligned)
            
            # Separate up and down days
            up_days = price_aligned > 0
            down_days = price_aligned < 0
            
            up_day_correlation = price_aligned[up_days].corr(volume_aligned[up_days]) if up_days.sum() > 1 else 0
            down_day_correlation = price_aligned[down_days].corr(volume_aligned[down_days]) if down_days.sum() > 1 else 0
            
            return {
                'overall_pv_correlation': overall_correlation if not np.isnan(overall_correlation) else 0,
                'up_day_pv_correlation': up_day_correlation if not np.isnan(up_day_correlation) else 0,
                'down_day_pv_correlation': down_day_correlation if not np.isnan(down_day_correlation) else 0,
                'correlation_strength': 'strong' if abs(overall_correlation) > 0.5 else 'weak' if abs(overall_correlation) < 0.2 else 'moderate'
            }
            
        except Exception as e:
            logger.error(f"Price-volume correlation analysis failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _estimate_dark_pool_activity(self, symbol: str) -> Dict[str, Any]:
        """Estimate dark pool activity based on volume and price patterns"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if hist.empty:
                return {'error': 'No data available'}
            
            # Dark pool indicators
            # 1. High volume with minimal price movement
            price_volatility = hist['Close'].pct_change().std()
            volume_avg = hist['Volume'].mean()
            
            # 2. Volume spikes without corresponding price spikes
            volume_spikes = hist[hist['Volume'] > volume_avg * 1.5]
            price_moves_on_spikes = abs(volume_spikes['Close'].pct_change()).mean()
            
            # 3. Estimate dark pool percentage
            dark_pool_estimate = min(50, max(0, (1 - price_moves_on_spikes / price_volatility) * 100))
            
            return {
                'estimated_dark_pool_percentage': dark_pool_estimate,
                'volume_price_disconnect': price_moves_on_spikes / price_volatility if price_volatility > 0 else 1,
                'dark_pool_activity_level': 'high' if dark_pool_estimate > 30 else 'low' if dark_pool_estimate < 10 else 'moderate',
                'confidence': 'low'  # This is a rough estimate
            }
            
        except Exception as e:
            logger.error(f"Dark pool estimation failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'error': str(e)}
    
    def _determine_smart_money_sentiment(self, symbol: str) -> str:
        """Determine overall smart money sentiment"""
        try:
            # Get all analysis components
            etf_analysis = self._analyze_etf_holdings(symbol)
            institutional_analysis = self._analyze_institutional_flow(symbol)
            volume_analysis = self._analyze_volume_patterns(symbol)
            
            sentiment_score = 0
            
            # ETF flow sentiment
            etf_flow_score = etf_analysis.get('weighted_flow_score', 0)
            sentiment_score += etf_flow_score / 100  # Normalize to -1 to 1
            
            # Institutional flow sentiment
            mfi = institutional_analysis.get('money_flow_index', 50)
            if mfi > 60:
                sentiment_score += 0.5
            elif mfi < 40:
                sentiment_score -= 0.5
            
            # Volume pattern sentiment
            volume_trend = volume_analysis.get('recent_volume_trend', 1)
            if volume_trend > 1.2:
                sentiment_score += 0.3
            elif volume_trend < 0.8:
                sentiment_score -= 0.3
            
            # Determine final sentiment
            if sentiment_score > 0.5:
                return 'bullish'
            elif sentiment_score < -0.5:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Smart money sentiment determination failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return 'neutral'
    
    def _calculate_flow_strength(self, symbol: str) -> Dict[str, Any]:
        """Calculate overall flow strength score"""
        try:
            # Get analysis components
            etf_analysis = self._analyze_etf_holdings(symbol)
            institutional_analysis = self._analyze_institutional_flow(symbol)
            volume_analysis = self._analyze_volume_patterns(symbol)
            
            # Calculate component scores
            etf_score = abs(etf_analysis.get('weighted_flow_score', 0)) / 100
            institutional_score = abs(institutional_analysis.get('money_flow_index', 50) - 50) / 50
            volume_score = min(1, volume_analysis.get('volume_ratio_20_60', 1))
            
            # Weighted average
            overall_strength = (etf_score * 0.4 + institutional_score * 0.4 + volume_score * 0.2)
            
            return {
                'overall_strength': overall_strength,
                'etf_component': etf_score,
                'institutional_component': institutional_score,
                'volume_component': volume_score,
                'strength_level': 'strong' if overall_strength > 0.7 else 'weak' if overall_strength < 0.3 else 'moderate'
            }
            
        except Exception as e:
            logger.error(f"Flow strength calculation failed for {symbol}: {e}", "MONEY_FLOW_TRACKER")
            return {'overall_strength': 0, 'strength_level': 'unknown'}

# Global money flow tracker instance
money_flow_tracker = MoneyFlowTracker()
