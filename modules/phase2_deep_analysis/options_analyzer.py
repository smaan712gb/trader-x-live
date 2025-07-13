"""
Options Market Analysis Module
Analyzes options chain data to identify key price levels and market sentiment
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import requests
from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from core.logger import logger

class OptionsAnalyzer:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = {}
    
    def analyze_options_chain(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze options chain to identify key price levels and sentiment
        Returns comprehensive options analysis
        """
        logger.info(f"Starting options analysis for {symbol}", "OPTIONS_ANALYZER")
        
        try:
            # Get options data
            options_data = self._get_options_data(symbol)
            
            if not options_data:
                return {'error': 'No options data available'}
            
            # Analyze the options chain
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': options_data.get('current_price', 0),
                'max_pain': self._calculate_max_pain(options_data),
                'high_oi_strikes': self._find_high_oi_strikes(options_data),
                'put_call_ratio': self._calculate_put_call_ratio(options_data),
                'implied_volatility': self._analyze_implied_volatility(options_data),
                'support_resistance': self._identify_support_resistance(options_data),
                'gamma_levels': self._calculate_gamma_levels(options_data),
                'options_sentiment': self._determine_options_sentiment(options_data),
                'key_levels': self._identify_key_levels(options_data)
            }
            
            logger.info(f"Options analysis complete for {symbol}", "OPTIONS_ANALYZER")
            return analysis
            
        except Exception as e:
            logger.error(f"Options analysis failed for {symbol}: {e}", "OPTIONS_ANALYZER")
            return {'error': str(e)}
    
    def _get_options_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get options chain data from yfinance"""
        try:
            # Check cache first
            cache_key = f"{symbol}_options"
            if (cache_key in self.cache and 
                cache_key in self.cache_expiry and 
                datetime.now() < self.cache_expiry[cache_key]):
                return self.cache[cache_key]
            
            ticker = yf.Ticker(symbol)
            
            # Get current stock price
            hist = ticker.history(period="1d")
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Get options expiration dates
            expirations = ticker.options
            if not expirations:
                return None
            
            # Use the nearest expiration (typically weekly or monthly)
            nearest_expiration = expirations[0]
            
            # Get options chain for nearest expiration
            options_chain = ticker.option_chain(nearest_expiration)
            
            calls = options_chain.calls
            puts = options_chain.puts
            
            options_data = {
                'symbol': symbol,
                'current_price': current_price,
                'expiration': nearest_expiration,
                'calls': calls,
                'puts': puts,
                'timestamp': datetime.now()
            }
            
            # Cache the data for 15 minutes
            self.cache[cache_key] = options_data
            self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=15)
            
            return options_data
            
        except Exception as e:
            logger.error(f"Failed to get options data for {symbol}: {e}", "OPTIONS_ANALYZER")
            return None
    
    def _calculate_max_pain(self, options_data: Dict[str, Any]) -> float:
        """Calculate max pain point - strike where most options expire worthless"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Get all unique strike prices
            all_strikes = set(calls['strike'].tolist() + puts['strike'].tolist())
            
            max_pain_values = {}
            
            for strike in all_strikes:
                total_pain = 0
                
                # Calculate pain for calls (ITM calls cause pain to writers)
                itm_calls = calls[calls['strike'] < strike]
                call_pain = (itm_calls['openInterest'] * (strike - itm_calls['strike'])).sum()
                
                # Calculate pain for puts (ITM puts cause pain to writers)
                itm_puts = puts[puts['strike'] > strike]
                put_pain = (itm_puts['openInterest'] * (itm_puts['strike'] - strike)).sum()
                
                total_pain = call_pain + put_pain
                max_pain_values[strike] = total_pain
            
            # Find strike with minimum pain (max pain point)
            max_pain_strike = min(max_pain_values.keys(), key=lambda k: max_pain_values[k])
            
            return float(max_pain_strike)
            
        except Exception as e:
            logger.error(f"Max pain calculation failed: {e}", "OPTIONS_ANALYZER")
            return 0
    
    def _find_high_oi_strikes(self, options_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Find strikes with highest open interest"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Sort by open interest and get top strikes
            top_call_strikes = calls.nlargest(5, 'openInterest')['strike'].tolist()
            top_put_strikes = puts.nlargest(5, 'openInterest')['strike'].tolist()
            
            return {
                'call_strikes': top_call_strikes,
                'put_strikes': top_put_strikes,
                'combined_strikes': list(set(top_call_strikes + top_put_strikes))
            }
            
        except Exception as e:
            logger.error(f"High OI strikes calculation failed: {e}", "OPTIONS_ANALYZER")
            return {'call_strikes': [], 'put_strikes': [], 'combined_strikes': []}
    
    def _calculate_put_call_ratio(self, options_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate put/call ratios for volume and open interest"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Volume ratio
            total_call_volume = calls['volume'].fillna(0).sum()
            total_put_volume = puts['volume'].fillna(0).sum()
            volume_ratio = total_put_volume / max(total_call_volume, 1)
            
            # Open interest ratio
            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()
            oi_ratio = total_put_oi / max(total_call_oi, 1)
            
            return {
                'volume_ratio': volume_ratio,
                'open_interest_ratio': oi_ratio,
                'sentiment': 'bearish' if volume_ratio > 1.2 else 'bullish' if volume_ratio < 0.8 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Put/call ratio calculation failed: {e}", "OPTIONS_ANALYZER")
            return {'volume_ratio': 1.0, 'open_interest_ratio': 1.0, 'sentiment': 'neutral'}
    
    def _analyze_implied_volatility(self, options_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze implied volatility patterns"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            current_price = options_data['current_price']
            
            # Get ATM options (closest to current price)
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]
            
            atm_call_iv = atm_call['impliedVolatility'].iloc[0] if not atm_call.empty else 0
            atm_put_iv = atm_put['impliedVolatility'].iloc[0] if not atm_put.empty else 0
            
            # Calculate average IV
            avg_call_iv = calls['impliedVolatility'].mean()
            avg_put_iv = puts['impliedVolatility'].mean()
            
            return {
                'atm_call_iv': atm_call_iv,
                'atm_put_iv': atm_put_iv,
                'avg_call_iv': avg_call_iv,
                'avg_put_iv': avg_put_iv,
                'iv_skew': avg_put_iv - avg_call_iv,
                'iv_level': 'high' if avg_call_iv > 0.4 else 'low' if avg_call_iv < 0.2 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"IV analysis failed: {e}", "OPTIONS_ANALYZER")
            return {'atm_call_iv': 0, 'atm_put_iv': 0, 'avg_call_iv': 0, 'avg_put_iv': 0, 'iv_skew': 0, 'iv_level': 'unknown'}
    
    def _identify_support_resistance(self, options_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Identify support and resistance levels from options data"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            current_price = options_data['current_price']
            
            # High OI put strikes often act as support
            put_support = puts[puts['openInterest'] > puts['openInterest'].quantile(0.8)]
            support_levels = put_support[put_support['strike'] < current_price]['strike'].tolist()
            
            # High OI call strikes often act as resistance
            call_resistance = calls[calls['openInterest'] > calls['openInterest'].quantile(0.8)]
            resistance_levels = call_resistance[call_resistance['strike'] > current_price]['strike'].tolist()
            
            return {
                'support_levels': sorted(support_levels, reverse=True)[:3],  # Top 3 closest support
                'resistance_levels': sorted(resistance_levels)[:3]  # Top 3 closest resistance
            }
            
        except Exception as e:
            logger.error(f"Support/resistance identification failed: {e}", "OPTIONS_ANALYZER")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _calculate_gamma_levels(self, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate gamma exposure levels"""
        try:
            calls = options_data['calls']
            puts = options_data['puts']
            
            # Simplified gamma calculation (would need more sophisticated model in production)
            # Using delta as proxy for gamma exposure
            
            call_gamma_exposure = (calls['openInterest'] * calls.get('delta', 0.5)).sum()
            put_gamma_exposure = (puts['openInterest'] * puts.get('delta', -0.5)).sum()
            
            net_gamma = call_gamma_exposure + put_gamma_exposure
            
            return {
                'call_gamma_exposure': call_gamma_exposure,
                'put_gamma_exposure': put_gamma_exposure,
                'net_gamma_exposure': net_gamma,
                'gamma_sentiment': 'positive' if net_gamma > 0 else 'negative'
            }
            
        except Exception as e:
            logger.error(f"Gamma calculation failed: {e}", "OPTIONS_ANALYZER")
            return {'call_gamma_exposure': 0, 'put_gamma_exposure': 0, 'net_gamma_exposure': 0, 'gamma_sentiment': 'neutral'}
    
    def _determine_options_sentiment(self, options_data: Dict[str, Any]) -> str:
        """Determine overall options sentiment"""
        try:
            put_call_ratio = self._calculate_put_call_ratio(options_data)
            iv_data = self._analyze_implied_volatility(options_data)
            
            sentiment_score = 0
            
            # Put/call ratio sentiment
            if put_call_ratio['volume_ratio'] < 0.8:
                sentiment_score += 1  # Bullish
            elif put_call_ratio['volume_ratio'] > 1.2:
                sentiment_score -= 1  # Bearish
            
            # IV sentiment
            if iv_data['iv_level'] == 'high':
                sentiment_score -= 0.5  # High IV often bearish
            elif iv_data['iv_level'] == 'low':
                sentiment_score += 0.5  # Low IV often bullish
            
            # IV skew sentiment
            if iv_data['iv_skew'] > 0.05:
                sentiment_score -= 0.5  # Put skew bearish
            elif iv_data['iv_skew'] < -0.05:
                sentiment_score += 0.5  # Call skew bullish
            
            if sentiment_score > 0.5:
                return 'bullish'
            elif sentiment_score < -0.5:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Options sentiment determination failed: {e}", "OPTIONS_ANALYZER")
            return 'neutral'
    
    def _identify_key_levels(self, options_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key price levels from options data"""
        try:
            current_price = options_data['current_price']
            max_pain = self._calculate_max_pain(options_data)
            high_oi = self._find_high_oi_strikes(options_data)
            support_resistance = self._identify_support_resistance(options_data)
            
            key_levels = []
            
            # Add max pain level
            key_levels.append({
                'level': max_pain,
                'type': 'max_pain',
                'importance': 'high',
                'distance_pct': abs(max_pain - current_price) / current_price * 100
            })
            
            # Add high OI levels
            for strike in high_oi['combined_strikes'][:5]:
                level_type = 'resistance' if strike > current_price else 'support'
                key_levels.append({
                    'level': strike,
                    'type': f'high_oi_{level_type}',
                    'importance': 'medium',
                    'distance_pct': abs(strike - current_price) / current_price * 100
                })
            
            # Add support/resistance levels
            for level in support_resistance['support_levels']:
                key_levels.append({
                    'level': level,
                    'type': 'support',
                    'importance': 'medium',
                    'distance_pct': abs(level - current_price) / current_price * 100
                })
            
            for level in support_resistance['resistance_levels']:
                key_levels.append({
                    'level': level,
                    'type': 'resistance',
                    'importance': 'medium',
                    'distance_pct': abs(level - current_price) / current_price * 100
                })
            
            # Sort by distance from current price
            key_levels.sort(key=lambda x: x['distance_pct'])
            
            return key_levels[:10]  # Return top 10 closest levels
            
        except Exception as e:
            logger.error(f"Key levels identification failed: {e}", "OPTIONS_ANALYZER")
            return []

# Global options analyzer instance
options_analyzer = OptionsAnalyzer()
