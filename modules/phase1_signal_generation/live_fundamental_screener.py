"""
Live Fundamental Screener - PRODUCTION ONLY
Only operates with validated real-time data - NO FALLBACK for live trading
"""
from typing import Dict, List, Any
from data.market_data_live import live_market_data_manager
from config.trading_config import TradingConfig
from core.logger import logger
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class LiveFundamentalScreener:
    def __init__(self):
        self.last_screen_time = None
        self.cached_results = {}
        self.data_validation_required = True
        
        # Verify live data connection on initialization
        if not live_market_data_manager.is_ready_for_live_trading():
            raise ConnectionError("Live fundamental screener requires validated IB Gateway connection")
    
    def screen_universe(self, stock_universe: List[str] = None) -> List[Dict[str, Any]]:
        """
        Screen stocks using LIVE DATA ONLY - NO FALLBACK
        Returns list of stocks that pass fundamental screening with validated data
        """
        if not live_market_data_manager.is_ready_for_live_trading():
            logger.error("Cannot perform live screening - IB Gateway not ready", "LIVE_SCREENER")
            raise ConnectionError("Live screening requires validated IB Gateway connection")
        
        if stock_universe is None:
            stock_universe = TradingConfig.TEST_STOCKS
        
        logger.info(f"Starting LIVE fundamental screening of {len(stock_universe)} stocks", "LIVE_SCREENER")
        logger.info("USING REAL DATA ONLY - NO FALLBACK", "LIVE_SCREENER")
        
        qualified_stocks = []
        failed_stocks = []
        
        # Use ThreadPoolExecutor for parallel processing with timeout
        with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers for stability
            # Submit all screening tasks
            future_to_symbol = {
                executor.submit(self._screen_single_stock_live, symbol): symbol 
                for symbol in stock_universe
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol, timeout=300):  # 5 minute total timeout
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)  # 60 second timeout per stock
                    if result:
                        qualified_stocks.append(result)
                        logger.info(f"✓ {symbol} qualified with live data", "LIVE_SCREENER")
                    else:
                        failed_stocks.append(symbol)
                        logger.warning(f"✗ {symbol} failed screening criteria", "LIVE_SCREENER")
                        
                except Exception as e:
                    failed_stocks.append(symbol)
                    logger.error(f"✗ {symbol} screening failed: {e}", "LIVE_SCREENER")
        
        # Sort by composite score (highest first)
        qualified_stocks.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        logger.info(f"LIVE screening complete: {len(qualified_stocks)} qualified, {len(failed_stocks)} failed", "LIVE_SCREENER")
        
        if len(failed_stocks) > 0:
            logger.warning(f"Failed stocks: {failed_stocks}", "LIVE_SCREENER")
        
        return qualified_stocks
    
    def _screen_single_stock_live(self, symbol: str) -> Dict[str, Any]:
        """Screen a single stock using LIVE DATA ONLY"""
        try:
            logger.debug(f"Screening {symbol} with live data", "LIVE_SCREENER")
            
            # Get VALIDATED fundamental data - will raise exception if not available
            fundamental_data = live_market_data_manager.get_fundamental_data(symbol)
            
            if not fundamental_data.get('validated', False):
                logger.error(f"Fundamental data for {symbol} not validated", "LIVE_SCREENER")
                raise ValueError(f"Unvalidated fundamental data for {symbol}")
            
            # Extract key metrics
            revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
            revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
            market_cap = fundamental_data.get('market_cap', 0)
            profit_margins = fundamental_data.get('profit_margins', 0)
            debt_to_equity = fundamental_data.get('debt_to_equity', 0)
            
            # Apply strict screening criteria for live trading
            screening_results = self._apply_live_screening_criteria(
                symbol, revenue_growth_yoy, revenue_growth_qoq, 
                market_cap, profit_margins, debt_to_equity
            )
            
            if not screening_results['passes_screen']:
                logger.debug(f"{symbol} failed live screening: {screening_results['failure_reason']}", "LIVE_SCREENER")
                return None
            
            # Get VALIDATED technical data for additional confirmation
            technical_data = live_market_data_manager.get_technical_indicators(symbol)
            
            if not technical_data.get('validated', False):
                logger.error(f"Technical data for {symbol} not validated", "LIVE_SCREENER")
                raise ValueError(f"Unvalidated technical data for {symbol}")
            
            # Calculate scores with live data
            growth_score = self._calculate_growth_score(revenue_growth_yoy, revenue_growth_qoq)
            quality_score = self._calculate_quality_score(fundamental_data)
            technical_score = self._calculate_technical_score(technical_data)
            
            # Composite score with technical confirmation
            composite_score = (growth_score * 0.5) + (quality_score * 0.3) + (technical_score * 0.2)
            
            result = {
                'symbol': symbol,
                'revenue_growth_yoy': revenue_growth_yoy,
                'revenue_growth_qoq': revenue_growth_qoq,
                'revenue_growth_score': growth_score,
                'quality_score': quality_score,
                'technical_score': technical_score,
                'composite_score': composite_score,
                'fundamental_data': fundamental_data,
                'technical_data': technical_data,
                'screening_results': screening_results,
                'screened_at': time.time(),
                'data_validated': True,
                'live_data_source': True
            }
            
            logger.info(f"{symbol} passed LIVE screening - Composite: {composite_score:.1f}", "LIVE_SCREENER")
            
            return result
            
        except Exception as e:
            logger.error(f"Live screening failed for {symbol}: {e}", "LIVE_SCREENER")
            return None
    
    def _apply_live_screening_criteria(self, symbol: str, revenue_growth_yoy: float, 
                                     revenue_growth_qoq: float, market_cap: float,
                                     profit_margins: float, debt_to_equity: float) -> Dict[str, Any]:
        """Apply STRICT screening criteria for live trading"""
        
        # STRICTER criteria for live trading
        MIN_REVENUE_GROWTH_YOY_LIVE = TradingConfig.MIN_REVENUE_GROWTH_YOY * 1.2  # 20% higher threshold
        MIN_REVENUE_GROWTH_QOQ_LIVE = TradingConfig.MIN_REVENUE_GROWTH_QOQ * 1.2  # 20% higher threshold
        MIN_MARKET_CAP_LIVE = 5_000_000_000  # $5B minimum for live trading
        
        # Primary criteria: Revenue growth (stricter for live trading)
        if revenue_growth_yoy < MIN_REVENUE_GROWTH_YOY_LIVE:
            return {
                'passes_screen': False,
                'failure_reason': f'YoY revenue growth {revenue_growth_yoy:.1f}% below live trading minimum {MIN_REVENUE_GROWTH_YOY_LIVE:.1f}%'
            }
        
        if revenue_growth_qoq < MIN_REVENUE_GROWTH_QOQ_LIVE:
            return {
                'passes_screen': False,
                'failure_reason': f'QoQ revenue growth {revenue_growth_qoq:.1f}% below live trading minimum {MIN_REVENUE_GROWTH_QOQ_LIVE:.1f}%'
            }
        
        # Market cap filter (stricter for live trading)
        if market_cap > 0 and market_cap < MIN_MARKET_CAP_LIVE:
            return {
                'passes_screen': False,
                'failure_reason': f'Market cap ${market_cap:,.0f} below live trading minimum ${MIN_MARKET_CAP_LIVE:,.0f}'
            }
        
        # Profitability requirement for live trading
        if profit_margins < 0.05:  # Minimum 5% profit margins
            return {
                'passes_screen': False,
                'failure_reason': f'Profit margins {profit_margins:.1%} below 5% minimum for live trading'
            }
        
        # Debt filter (stricter for live trading)
        if debt_to_equity > 0 and debt_to_equity > 1.5:  # Lower debt tolerance
            return {
                'passes_screen': False,
                'failure_reason': f'Debt-to-equity ratio {debt_to_equity:.1f} above 1.5 maximum for live trading'
            }
        
        # Check for accelerating growth
        growth_acceleration = self._check_growth_acceleration(revenue_growth_yoy, revenue_growth_qoq)
        
        return {
            'passes_screen': True,
            'screening_type': 'live_trading',
            'growth_acceleration': growth_acceleration,
            'criteria_met': {
                'revenue_growth_yoy': revenue_growth_yoy >= MIN_REVENUE_GROWTH_YOY_LIVE,
                'revenue_growth_qoq': revenue_growth_qoq >= MIN_REVENUE_GROWTH_QOQ_LIVE,
                'market_cap_adequate': market_cap >= MIN_MARKET_CAP_LIVE,
                'profitable': profit_margins >= 0.05,
                'debt_manageable': debt_to_equity <= 1.5 if debt_to_equity > 0 else True
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
        """Calculate growth score for live trading"""
        try:
            # Base score from YoY growth (higher standards)
            yoy_score = min(revenue_growth_yoy / 150 * 50, 50)  # Max 50 points, higher threshold
            
            # Bonus from QoQ growth
            qoq_score = min(revenue_growth_qoq / 75 * 30, 30)  # Max 30 points, higher threshold
            
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
            logger.error(f"Error calculating growth score: {e}", "LIVE_SCREENER")
            return 0
    
    def _calculate_quality_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate quality score for live trading (stricter standards)"""
        try:
            score = 40  # Lower base score for stricter standards
            
            # Profitability (higher standards)
            profit_margins = fundamental_data.get('profit_margins', 0)
            if profit_margins > 0.3:  # 30%+ margins
                score += 25
            elif profit_margins > 0.2:  # 20%+ margins
                score += 20
            elif profit_margins > 0.1:  # 10%+ margins
                score += 10
            elif profit_margins < 0.05:  # Below 5% margins
                score -= 30
            
            # Financial strength (stricter)
            debt_to_equity = fundamental_data.get('debt_to_equity', 0)
            if debt_to_equity > 0:
                if debt_to_equity < 0.3:  # Very low debt
                    score += 20
                elif debt_to_equity < 0.8:  # Low debt
                    score += 10
                elif debt_to_equity > 1.5:  # High debt
                    score -= 25
            
            # Valuation reasonableness (stricter)
            pe_ratio = fundamental_data.get('price_to_earnings', 0)
            if pe_ratio > 0:
                if pe_ratio < 20:  # Very reasonable valuation
                    score += 15
                elif pe_ratio < 35:  # Reasonable valuation
                    score += 10
                elif pe_ratio > 80:  # Very expensive
                    score -= 20
            
            # Market position (higher threshold)
            market_cap = fundamental_data.get('market_cap', 0)
            if market_cap > 50_000_000_000:  # Large cap stability
                score += 10
            elif market_cap > 10_000_000_000:  # Mid-large cap
                score += 5
            
            return max(0, min(score, 100))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}", "LIVE_SCREENER")
            return 0
    
    def _calculate_technical_score(self, technical_data: Dict[str, Any]) -> float:
        """Calculate technical score for additional confirmation"""
        try:
            score = 50  # Base score
            
            # Trend strength
            trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
            if trend_strength == 'STRONG_BULLISH':
                score += 25
            elif trend_strength == 'BULLISH':
                score += 15
            elif trend_strength == 'BEARISH':
                score -= 15
            elif trend_strength == 'STRONG_BEARISH':
                score -= 25
            
            # RSI momentum (avoid overbought)
            rsi = technical_data.get('rsi', 50)
            if 30 <= rsi <= 70:  # Good momentum zone
                score += 15
            elif rsi > 80:  # Very overbought
                score -= 20
            elif rsi < 20:  # Very oversold
                score -= 10
            
            # Volume confirmation
            volume_ratio = technical_data.get('volume_ratio', 1.0)
            if volume_ratio > 1.5:  # Above average volume
                score += 10
            elif volume_ratio < 0.7:  # Below average volume
                score -= 10
            
            return max(0, min(score, 100))
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}", "LIVE_SCREENER")
            return 50
    
    def get_top_live_candidates(self, max_candidates: int = 5) -> List[Dict[str, Any]]:
        """Get top candidates for live trading (smaller, more selective list)"""
        try:
            # Run live screening
            candidates = self.screen_universe()
            
            # Return top candidates (smaller list for live trading)
            top_candidates = candidates[:max_candidates]
            
            logger.info(f"Selected {len(top_candidates)} top candidates for live trading", "LIVE_SCREENER")
            
            return top_candidates
            
        except Exception as e:
            logger.error(f"Error getting top live candidates: {e}", "LIVE_SCREENER")
            raise
    
    def validate_screening_readiness(self) -> Dict[str, Any]:
        """Validate that the system is ready for live screening"""
        try:
            status = live_market_data_manager.get_connection_status()
            
            readiness_check = {
                'ib_gateway_connected': status['ib_gateway_connected'],
                'data_validation_enabled': status['data_validation_enabled'],
                'live_trading_ready': status['live_trading_ready'],
                'min_data_quality': status['min_data_quality_threshold'],
                'ready_for_screening': status['live_trading_ready'],
                'timestamp': status['timestamp']
            }
            
            if not readiness_check['ready_for_screening']:
                logger.error("System not ready for live screening", "LIVE_SCREENER")
                raise ConnectionError("Live screening system not ready")
            
            logger.info("Live screening system validated and ready", "LIVE_SCREENER")
            return readiness_check
            
        except Exception as e:
            logger.error(f"Screening readiness validation failed: {e}", "LIVE_SCREENER")
            raise

# Global live fundamental screener instance
live_fundamental_screener = LiveFundamentalScreener()
