"""
Phase 1: Fundamental Screener Module
Identifies companies with high, improving growth
"""
from typing import Dict, List, Any
from data.market_data_production import production_market_data_manager
from config.trading_config import TradingConfig
from core.logger import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class FundamentalScreener:
    def __init__(self):
        self.last_screen_time = None
        self.cached_results = {}
    
    def screen_universe(self, stock_universe: List[str] = None) -> List[Dict[str, Any]]:
        """
        Screen the entire market universe for fundamental criteria
        Returns list of stocks that pass fundamental screening
        """
        if stock_universe is None:
            stock_universe = TradingConfig.TEST_STOCKS
        
        logger.info(f"Starting fundamental screening of {len(stock_universe)} stocks", "FUNDAMENTAL_SCREENER")
        
        qualified_stocks = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._screen_single_stock, symbol): symbol for symbol in stock_universe}
            
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per stock
                    if result:
                        qualified_stocks.append(result)
                except Exception as e:
                    symbol = futures[future]
                    logger.error(f"Failed to screen {symbol}: {e}", "FUNDAMENTAL_SCREENER")
        
        # Sort by revenue growth (highest first)
        qualified_stocks.sort(key=lambda x: x.get('revenue_growth_score', 0), reverse=True)
        
        logger.info(f"Fundamental screening complete: {len(qualified_stocks)} stocks qualified", "FUNDAMENTAL_SCREENER")
        
        return qualified_stocks
    
    def _screen_single_stock(self, symbol: str) -> Dict[str, Any]:
        """Screen a single stock for fundamental criteria"""
        try:
            logger.debug(f"Screening {symbol} fundamentals", "FUNDAMENTAL_SCREENER")
            
            # Get fundamental data
            fundamental_data = production_market_data_manager.get_fundamental_data(symbol)
            
            if not fundamental_data:
                logger.warning(f"No fundamental data available for {symbol}, using fallback screening", "FUNDAMENTAL_SCREENER")
                # Use fallback screening based on technical data
                return self._fallback_screening(symbol)
            
            # Extract key metrics
            revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
            revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
            market_cap = fundamental_data.get('market_cap', 0)
            profit_margins = fundamental_data.get('profit_margins', 0)
            debt_to_equity = fundamental_data.get('debt_to_equity', 0)
            
            # Apply screening criteria
            screening_results = self._apply_screening_criteria(
                symbol, revenue_growth_yoy, revenue_growth_qoq, 
                market_cap, profit_margins, debt_to_equity
            )
            
            if not screening_results['passes_screen']:
                logger.debug(f"{symbol} failed fundamental screening: {screening_results['failure_reason']}", "FUNDAMENTAL_SCREENER")
                return None
            
            # Calculate composite score
            growth_score = self._calculate_growth_score(revenue_growth_yoy, revenue_growth_qoq)
            quality_score = self._calculate_quality_score(fundamental_data)
            
            result = {
                'symbol': symbol,
                'revenue_growth_yoy': revenue_growth_yoy,
                'revenue_growth_qoq': revenue_growth_qoq,
                'revenue_growth_score': growth_score,
                'quality_score': quality_score,
                'composite_score': (growth_score * 0.7) + (quality_score * 0.3),
                'fundamental_data': fundamental_data,
                'screening_results': screening_results,
                'screened_at': time.time()
            }
            
            logger.info(f"{symbol} passed fundamental screening - Growth Score: {growth_score:.2f}, Quality Score: {quality_score:.2f}", "FUNDAMENTAL_SCREENER")
            
            return result
            
        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}", "FUNDAMENTAL_SCREENER")
            return None
    
    def _apply_screening_criteria(self, symbol: str, revenue_growth_yoy: float, 
                                revenue_growth_qoq: float, market_cap: float,
                                profit_margins: float, debt_to_equity: float) -> Dict[str, Any]:
        """Apply fundamental screening criteria"""
        
        # Primary criteria: Revenue growth
        if revenue_growth_yoy < TradingConfig.MIN_REVENUE_GROWTH_YOY:
            return {
                'passes_screen': False,
                'failure_reason': f'YoY revenue growth {revenue_growth_yoy:.1f}% below minimum {TradingConfig.MIN_REVENUE_GROWTH_YOY}%'
            }
        
        if revenue_growth_qoq < TradingConfig.MIN_REVENUE_GROWTH_QOQ:
            return {
                'passes_screen': False,
                'failure_reason': f'QoQ revenue growth {revenue_growth_qoq:.1f}% below minimum {TradingConfig.MIN_REVENUE_GROWTH_QOQ}%'
            }
        
        # Secondary criteria: Company quality filters
        quality_checks = []
        
        # Market cap filter (avoid micro-caps for liquidity)
        if market_cap > 0 and market_cap < 1_000_000_000:  # $1B minimum
            quality_checks.append('Market cap below $1B')
        
        # Debt filter (avoid over-leveraged companies)
        if debt_to_equity > 0 and debt_to_equity > 2.0:  # D/E ratio above 2.0
            quality_checks.append('High debt-to-equity ratio')
        
        # Check for accelerating growth
        growth_acceleration = self._check_growth_acceleration(revenue_growth_yoy, revenue_growth_qoq)
        
        return {
            'passes_screen': True,
            'quality_warnings': quality_checks,
            'growth_acceleration': growth_acceleration,
            'criteria_met': {
                'revenue_growth_yoy': revenue_growth_yoy >= TradingConfig.MIN_REVENUE_GROWTH_YOY,
                'revenue_growth_qoq': revenue_growth_qoq >= TradingConfig.MIN_REVENUE_GROWTH_QOQ,
                'market_cap_adequate': market_cap >= 1_000_000_000,
                'debt_manageable': debt_to_equity <= 2.0 if debt_to_equity > 0 else True
            }
        }
    
    def _check_growth_acceleration(self, revenue_growth_yoy: float, revenue_growth_qoq: float) -> str:
        """Check if growth is accelerating"""
        # Annualized quarterly growth vs yearly growth
        annualized_qoq = ((1 + revenue_growth_qoq/100) ** 4 - 1) * 100
        
        if annualized_qoq > revenue_growth_yoy * 1.2:  # 20% higher
            return "ACCELERATING"
        elif annualized_qoq > revenue_growth_yoy * 0.8:  # Within 20%
            return "STABLE"
        else:
            return "DECELERATING"
    
    def _calculate_growth_score(self, revenue_growth_yoy: float, revenue_growth_qoq: float) -> float:
        """Calculate a composite growth score (0-100)"""
        try:
            # Base score from YoY growth
            yoy_score = min(revenue_growth_yoy / 100 * 50, 50)  # Max 50 points
            
            # Bonus from QoQ growth
            qoq_score = min(revenue_growth_qoq / 50 * 30, 30)  # Max 30 points
            
            # Acceleration bonus
            acceleration = self._check_growth_acceleration(revenue_growth_yoy, revenue_growth_qoq)
            acceleration_bonus = {
                "ACCELERATING": 20,
                "STABLE": 10,
                "DECELERATING": 0
            }.get(acceleration, 0)
            
            total_score = yoy_score + qoq_score + acceleration_bonus
            return min(total_score, 100)
            
        except Exception as e:
            logger.error(f"Error calculating growth score: {e}", "FUNDAMENTAL_SCREENER")
            return 0
    
    def _calculate_quality_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate a quality score based on financial health (0-100)"""
        try:
            score = 50  # Base score
            
            # Profitability
            profit_margins = fundamental_data.get('profit_margins', 0)
            if profit_margins > 0.2:  # 20%+ margins
                score += 20
            elif profit_margins > 0.1:  # 10%+ margins
                score += 10
            elif profit_margins < 0:  # Negative margins
                score -= 20
            
            # Financial strength
            debt_to_equity = fundamental_data.get('debt_to_equity', 0)
            if debt_to_equity > 0:
                if debt_to_equity < 0.5:  # Low debt
                    score += 15
                elif debt_to_equity > 2.0:  # High debt
                    score -= 15
            
            # Valuation reasonableness
            pe_ratio = fundamental_data.get('price_to_earnings', 0)
            if pe_ratio > 0:
                if pe_ratio < 25:  # Reasonable valuation
                    score += 10
                elif pe_ratio > 100:  # Very expensive
                    score -= 10
            
            # Market position
            market_cap = fundamental_data.get('market_cap', 0)
            if market_cap > 10_000_000_000:  # Large cap stability
                score += 5
            
            return max(0, min(score, 100))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}", "FUNDAMENTAL_SCREENER")
            return 50
    
    def get_top_candidates(self, max_candidates: int = 10) -> List[Dict[str, Any]]:
        """Get top fundamental candidates from recent screening"""
        try:
            # Run fresh screening
            candidates = self.screen_universe()
            
            # Return top candidates
            return candidates[:max_candidates]
            
        except Exception as e:
            logger.error(f"Error getting top candidates: {e}", "FUNDAMENTAL_SCREENER")
            return []
    
    def _fallback_screening(self, symbol: str) -> Dict[str, Any]:
        """Fallback screening when fundamental data is unavailable"""
        try:
            logger.info(f"Using fallback screening for {symbol}", "FUNDAMENTAL_SCREENER")
            
            # Get basic price and volume data from IB Gateway
            technical_data = production_market_data_manager.get_technical_indicators(symbol)
            
            if not technical_data:
                logger.warning(f"No technical data available for {symbol}", "FUNDAMENTAL_SCREENER")
                return None
            
            current_price = technical_data.get('current_price', 0)
            volume_ratio = technical_data.get('volume_ratio', 1.0)
            trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
            
            # Basic screening criteria for fallback
            if current_price < 5:  # Avoid penny stocks
                logger.debug(f"{symbol} failed fallback screening: price too low ({current_price})", "FUNDAMENTAL_SCREENER")
                return None
            
            if volume_ratio < 0.5:  # Avoid low volume stocks
                logger.debug(f"{symbol} failed fallback screening: low volume ratio ({volume_ratio})", "FUNDAMENTAL_SCREENER")
                return None
            
            # Calculate fallback scores based on technical momentum
            momentum_score = self._calculate_momentum_score(technical_data)
            volume_score = self._calculate_volume_score(technical_data)
            
            # Create fallback result
            result = {
                'symbol': symbol,
                'revenue_growth_yoy': 25.0,  # Assume moderate growth for fallback
                'revenue_growth_qoq': 8.0,   # Assume moderate growth for fallback
                'revenue_growth_score': momentum_score,
                'quality_score': volume_score,
                'composite_score': (momentum_score * 0.6) + (volume_score * 0.4),
                'fundamental_data': {
                    'current_price': current_price,
                    'volume_ratio': volume_ratio,
                    'trend_strength': trend_strength,
                    'data_source': 'fallback_technical'
                },
                'screening_results': {
                    'passes_screen': True,
                    'screening_type': 'fallback',
                    'criteria_met': {
                        'price_adequate': current_price >= 5,
                        'volume_adequate': volume_ratio >= 0.5,
                        'technical_momentum': momentum_score >= 50
                    }
                },
                'screened_at': time.time()
            }
            
            logger.info(f"{symbol} passed fallback screening - Momentum: {momentum_score:.1f}, Volume: {volume_score:.1f}", "FUNDAMENTAL_SCREENER")
            return result
            
        except Exception as e:
            logger.error(f"Fallback screening failed for {symbol}: {e}", "FUNDAMENTAL_SCREENER")
            return None
    
    def _calculate_momentum_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate momentum score from technical data"""
        try:
            score = 50  # Base score
            
            # Trend strength
            trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
            if trend_strength == 'STRONG_BULLISH':
                score += 30
            elif trend_strength == 'BULLISH':
                score += 20
            elif trend_strength == 'BEARISH':
                score -= 20
            elif trend_strength == 'STRONG_BEARISH':
                score -= 30
            
            # RSI momentum
            rsi = technical_data.get('rsi', 50)
            if 40 <= rsi <= 60:  # Neutral zone
                score += 10
            elif rsi > 70:  # Overbought
                score -= 10
            elif rsi < 30:  # Oversold but could bounce
                score += 5
            
            # Price position relative to support/resistance
            support_level = technical_data.get('support_level', 0)
            resistance_level = technical_data.get('resistance_level', 0)
            current_price = technical_data.get('current_price', 0)
            
            if support_level > 0 and resistance_level > 0 and current_price > 0:
                price_position = (current_price - support_level) / (resistance_level - support_level)
                if 0.3 <= price_position <= 0.7:  # Good position
                    score += 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Momentum score calculation failed: {e}", "FUNDAMENTAL_SCREENER")
            return 50
    
    def _calculate_volume_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate volume score from technical data"""
        try:
            score = 50  # Base score
            
            # Volume ratio (current vs average)
            volume_ratio = technical_data.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:  # High volume
                score += 25
            elif volume_ratio > 1.5:  # Above average volume
                score += 15
            elif volume_ratio < 0.5:  # Low volume
                score -= 20
            
            # Volume trend
            volume_trend = technical_data.get('volume_trend', 'neutral')
            if volume_trend == 'increasing':
                score += 15
            elif volume_trend == 'decreasing':
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Volume score calculation failed: {e}", "FUNDAMENTAL_SCREENER")
            return 50

    def analyze_sector_trends(self, qualified_stocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across qualified stocks"""
        try:
            if not qualified_stocks:
                return {}
            
            # Calculate aggregate metrics
            avg_yoy_growth = sum(stock['revenue_growth_yoy'] for stock in qualified_stocks) / len(qualified_stocks)
            avg_qoq_growth = sum(stock['revenue_growth_qoq'] for stock in qualified_stocks) / len(qualified_stocks)
            
            # Growth distribution
            high_growth_count = len([s for s in qualified_stocks if s['revenue_growth_yoy'] > 50])
            accelerating_count = len([s for s in qualified_stocks 
                                    if s['screening_results']['growth_acceleration'] == 'ACCELERATING'])
            
            analysis = {
                'total_qualified': len(qualified_stocks),
                'average_yoy_growth': avg_yoy_growth,
                'average_qoq_growth': avg_qoq_growth,
                'high_growth_stocks': high_growth_count,
                'accelerating_stocks': accelerating_count,
                'top_performers': [s['symbol'] for s in qualified_stocks[:5]]
            }
            
            logger.info(f"Sector analysis: {analysis}", "FUNDAMENTAL_SCREENER")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {e}", "FUNDAMENTAL_SCREENER")
            return {}

# Global fundamental screener instance
fundamental_screener = FundamentalScreener()
