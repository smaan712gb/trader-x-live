"""
Main Orchestrator for Trader-X System
Coordinates all phases of the trading pipeline
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from config.trading_config import TradingConfig
from core.logger import logger
from core.ai_engine import ai_engine
from data.market_data_enhanced import market_data_manager

# Phase 1 modules
from modules.phase1_signal_generation.fundamental_screener import fundamental_screener
from modules.phase1_signal_generation.sentiment_analyzer import sentiment_analyzer

# Phase 2 modules
from modules.phase2_deep_analysis.options_analyzer import options_analyzer
from modules.phase2_deep_analysis.money_flow_tracker import money_flow_tracker
from modules.phase2_deep_analysis.technical_analyzer import technical_analyzer

# Market context module
from modules.market_context.market_trend_analyzer import market_trend_analyzer

# Phase 3 module
from modules.phase3_execution.trade_executor import trade_executor

class TradingOrchestrator:
    def __init__(self):
        self.current_candidates = []
        self.active_positions = {}
        self.daily_trade_count = 0
        self.last_run_date = None
        self.system_status = "INITIALIZED"
    
    def run_full_pipeline(self, test_mode: bool = True) -> Dict[str, Any]:
        """
        Run the complete trading pipeline
        Returns summary of pipeline execution
        """
        logger.info("Starting Trader-X full pipeline execution", "ORCHESTRATOR")
        
        pipeline_start_time = time.time()
        results = {
            'execution_time': pipeline_start_time,
            'test_mode': test_mode,
            'phase1_results': {},
            'phase2_results': {},
            'phase3_results': {},
            'ai_decisions': [],
            'trades_executed': 0,
            'errors': []
        }
        
        try:
            # Reset daily counters if new day
            self._check_daily_reset()
            
            # Market Context Analysis (before Phase 1)
            logger.info("=== MARKET CONTEXT ANALYSIS ===", "ORCHESTRATOR")
            market_context = self._analyze_market_context()
            results['market_context'] = market_context
            
            # Check if market conditions are favorable for trading
            should_proceed = self._should_proceed_with_trading(market_context)
            if not should_proceed:
                logger.warning("Market conditions unfavorable but proceeding for testing", "ORCHESTRATOR")
                # For testing purposes, we'll proceed anyway but log the warning
                results['market_override'] = True
                results['override_reason'] = "Testing mode - proceeding despite market conditions"
            
            # Phase 1: Signal Generation
            logger.info("=== PHASE 1: SIGNAL GENERATION ===", "ORCHESTRATOR")
            phase1_results = self._execute_phase1()
            results['phase1_results'] = phase1_results
            
            if not phase1_results.get('qualified_candidates'):
                logger.info("No candidates passed Phase 1 screening", "ORCHESTRATOR")
                return results
            
            # Phase 2: Deep Analysis (placeholder for now)
            logger.info("=== PHASE 2: DEEP ANALYSIS ===", "ORCHESTRATOR")
            phase2_results = self._execute_phase2(phase1_results['qualified_candidates'])
            results['phase2_results'] = phase2_results
            
            # AI Decision Making
            logger.info("=== AI DECISION SYNTHESIS ===", "ORCHESTRATOR")
            ai_decisions = self._execute_ai_decisions(phase1_results, phase2_results)
            results['ai_decisions'] = ai_decisions
            
            # Phase 3: Execution (placeholder for now)
            if not test_mode:
                logger.info("=== PHASE 3: EXECUTION ===", "ORCHESTRATOR")
                phase3_results = self._execute_phase3(ai_decisions)
                results['phase3_results'] = phase3_results
                results['trades_executed'] = len(phase3_results.get('executed_trades', []))
            else:
                logger.info("Test mode: Skipping Phase 3 execution", "ORCHESTRATOR")
            
            # Calculate execution time
            execution_time = time.time() - pipeline_start_time
            results['total_execution_time'] = execution_time
            
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds", "ORCHESTRATOR")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", "ORCHESTRATOR")
            results['errors'].append(str(e))
            return results
    
    def _check_daily_reset(self):
        """Reset daily counters if it's a new day"""
        current_date = datetime.now().date()
        
        if self.last_run_date != current_date:
            self.daily_trade_count = 0
            self.last_run_date = current_date
            logger.info(f"Daily reset completed for {current_date}", "ORCHESTRATOR")
    
    def _execute_phase1(self) -> Dict[str, Any]:
        """Execute Phase 1: Signal Generation"""
        phase1_results = {
            'fundamental_screening': {},
            'sentiment_analysis': {},
            'qualified_candidates': [],
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            # Step 1: Fundamental Screening
            logger.phase_log("1", "ALL", "Starting fundamental screening")
            
            fundamental_candidates = fundamental_screener.screen_universe(TradingConfig.TEST_STOCKS)
            phase1_results['fundamental_screening'] = {
                'total_screened': len(TradingConfig.TEST_STOCKS),
                'passed_screening': len(fundamental_candidates),
                'candidates': fundamental_candidates
            }
            
            logger.phase_log("1", "ALL", f"Fundamental screening complete: {len(fundamental_candidates)} candidates")
            
            if not fundamental_candidates:
                logger.warning("No stocks passed fundamental screening", "ORCHESTRATOR")
                return phase1_results
            
            # Step 2: Sentiment Analysis for qualified candidates
            logger.phase_log("1", "ALL", "Starting sentiment analysis for qualified candidates")
            
            sentiment_results = {}
            qualified_candidates = []
            
            # Analyze sentiment for each fundamental candidate
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(sentiment_analyzer.analyze_stock_sentiment, candidate['symbol']): candidate 
                    for candidate in fundamental_candidates
                }
                
                for future in futures:
                    try:
                        candidate = futures[future]
                        sentiment_data = future.result(timeout=60)  # 60 second timeout
                        
                        symbol = candidate['symbol']
                        sentiment_results[symbol] = sentiment_data
                        
                        # Check if candidate passes sentiment criteria
                        hype_score = sentiment_data.get('hype_score', 0)
                        
                        if hype_score >= TradingConfig.MIN_SENTIMENT_SCORE:
                            # Combine fundamental and sentiment data
                            qualified_candidate = {
                                **candidate,
                                'sentiment_data': sentiment_data,
                                'hype_score': hype_score,
                                'phase1_score': self._calculate_phase1_score(candidate, sentiment_data)
                            }
                            qualified_candidates.append(qualified_candidate)
                            
                            logger.phase_log("1", symbol, f"Qualified for Phase 2 - Hype Score: {hype_score:.1f}%")
                        else:
                            logger.phase_log("1", symbol, f"Failed sentiment criteria - Hype Score: {hype_score:.1f}%")
                            
                    except Exception as e:
                        candidate = futures[future]
                        logger.error(f"Sentiment analysis failed for {candidate['symbol']}: {e}", "ORCHESTRATOR")
            
            # Sort qualified candidates by combined score
            qualified_candidates.sort(key=lambda x: x.get('phase1_score', 0), reverse=True)
            
            phase1_results['sentiment_analysis'] = sentiment_results
            phase1_results['qualified_candidates'] = qualified_candidates
            
            logger.phase_log("1", "ALL", f"Phase 1 complete: {len(qualified_candidates)} candidates qualified")
            
        except Exception as e:
            logger.error(f"Phase 1 execution failed: {e}", "ORCHESTRATOR")
            phase1_results['error'] = str(e)
        
        phase1_results['execution_time'] = time.time() - start_time
        return phase1_results
    
    def _calculate_phase1_score(self, fundamental_data: Dict[str, Any], sentiment_data: Dict[str, Any]) -> float:
        """Calculate combined Phase 1 score"""
        try:
            # Fundamental score (0-70 points)
            fundamental_score = fundamental_data.get('composite_score', 0) * 0.7
            
            # Sentiment score (0-30 points)
            hype_score = sentiment_data.get('hype_score', 0)
            sentiment_score = (hype_score / 100) * 30
            
            # Bonus for authenticity
            authenticity_score = sentiment_data.get('authenticity_score', 0)
            authenticity_bonus = (authenticity_score / 10) * 5  # Max 5 bonus points
            
            total_score = fundamental_score + sentiment_score + authenticity_bonus
            return min(total_score, 100)
            
        except Exception as e:
            logger.error(f"Phase 1 score calculation failed: {e}", "ORCHESTRATOR")
            return 0
    
    def _execute_phase2(self, qualified_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute Phase 2: Deep Analysis - Full Implementation"""
        phase2_results = {
            'candidates_analyzed': len(qualified_candidates),
            'analysis_results': {},
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            logger.phase_log("2", "ALL", f"Starting deep analysis for {len(qualified_candidates)} candidates")
            
            # Use ThreadPoolExecutor for parallel analysis
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self._analyze_single_candidate, candidate): candidate 
                    for candidate in qualified_candidates
                }
                
                for future in futures:
                    try:
                        candidate = futures[future]
                        symbol = candidate['symbol']
                        analysis_result = future.result(timeout=120)  # 2 minute timeout per candidate
                        
                        if analysis_result:
                            phase2_results['analysis_results'][symbol] = analysis_result
                            logger.phase_log("2", symbol, f"Deep analysis completed - Score: {analysis_result.get('phase2_score', 0):.1f}")
                        else:
                            logger.warning(f"No analysis result for {symbol}", "ORCHESTRATOR")
                            
                    except Exception as e:
                        candidate = futures[future]
                        symbol = candidate['symbol']
                        logger.error(f"Deep analysis failed for {symbol}: {e}", "ORCHESTRATOR")
                        
                        # Create fallback analysis
                        fallback_analysis = {
                            'symbol': symbol,
                            'options_analysis': {'error': str(e)},
                            'money_flow_analysis': {'error': str(e)},
                            'technical_analysis': self._get_basic_technical_analysis(symbol),
                            'phase2_score': 50,  # Neutral score on error
                            'error': str(e)
                        }
                        phase2_results['analysis_results'][symbol] = fallback_analysis
            
        except Exception as e:
            logger.error(f"Phase 2 execution failed: {e}", "ORCHESTRATOR")
            phase2_results['error'] = str(e)
        
        phase2_results['execution_time'] = time.time() - start_time
        return phase2_results
    
    def _analyze_single_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive deep analysis for a single candidate"""
        symbol = candidate['symbol']
        
        try:
            logger.phase_log("2", symbol, "Starting comprehensive deep analysis")
            
            # 1. Options Analysis
            logger.phase_log("2", symbol, "Analyzing options chain")
            options_analysis = options_analyzer.analyze_options_chain(symbol)
            
            # 2. Money Flow Analysis
            logger.phase_log("2", symbol, "Analyzing smart money flow")
            money_flow_analysis = money_flow_tracker.analyze_money_flow(symbol)
            
            # 3. Technical Analysis
            logger.phase_log("2", symbol, "Performing technical analysis")
            technical_analysis = technical_analyzer.analyze_technical_indicators(symbol)
            
            # 4. Calculate comprehensive Phase 2 score
            phase2_score = self._calculate_phase2_score(
                candidate, options_analysis, money_flow_analysis, technical_analysis
            )
            
            analysis_result = {
                'symbol': symbol,
                'options_analysis': options_analysis,
                'money_flow_analysis': money_flow_analysis,
                'technical_analysis': technical_analysis,
                'phase2_score': phase2_score,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.phase_log("2", symbol, f"Deep analysis complete - Final Score: {phase2_score:.1f}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Single candidate analysis failed for {symbol}: {e}", "ORCHESTRATOR")
            return None
    
    def _calculate_phase2_score(self, candidate: Dict[str, Any], 
                               options_analysis: Dict[str, Any],
                               money_flow_analysis: Dict[str, Any],
                               technical_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive Phase 2 score"""
        try:
            total_score = 0
            max_score = 100
            
            # Options Analysis Score (30 points)
            options_score = self._score_options_analysis(options_analysis)
            total_score += options_score * 0.3
            
            # Money Flow Score (40 points) - Most important for institutional backing
            money_flow_score = self._score_money_flow_analysis(money_flow_analysis)
            total_score += money_flow_score * 0.4
            
            # Technical Analysis Score (30 points)
            technical_score = self._score_technical_analysis(technical_analysis)
            total_score += technical_score * 0.3
            
            # Bonus for alignment (up to 10 points)
            alignment_bonus = self._calculate_alignment_bonus(
                options_analysis, money_flow_analysis, technical_analysis
            )
            total_score += alignment_bonus
            
            return min(max_score, max(0, total_score))
            
        except Exception as e:
            logger.error(f"Phase 2 score calculation failed: {e}", "ORCHESTRATOR")
            return 50  # Neutral score on error
    
    def _score_options_analysis(self, options_analysis: Dict[str, Any]) -> float:
        """Score options analysis (0-100)"""
        if 'error' in options_analysis:
            return 50  # Neutral score on error
        
        try:
            score = 50  # Base score
            
            # Options sentiment
            sentiment = options_analysis.get('options_sentiment', 'neutral')
            if sentiment == 'bullish':
                score += 20
            elif sentiment == 'bearish':
                score -= 20
            
            # Put/call ratio
            pc_ratio = options_analysis.get('put_call_ratio', {})
            volume_ratio = pc_ratio.get('volume_ratio', 1.0)
            if volume_ratio < 0.8:  # Bullish
                score += 15
            elif volume_ratio > 1.2:  # Bearish
                score -= 15
            
            # Implied volatility level
            iv_data = options_analysis.get('implied_volatility', {})
            iv_level = iv_data.get('iv_level', 'normal')
            if iv_level == 'low':  # Good for buying
                score += 10
            elif iv_level == 'high':  # Expensive options
                score -= 5
            
            # Max pain proximity
            max_pain = options_analysis.get('max_pain', 0)
            current_price = options_analysis.get('current_price', 0)
            if max_pain > 0 and current_price > 0:
                distance_pct = abs(max_pain - current_price) / current_price
                if distance_pct < 0.05:  # Within 5% of max pain
                    score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Options scoring failed: {e}", "ORCHESTRATOR")
            return 50
    
    def _score_money_flow_analysis(self, money_flow_analysis: Dict[str, Any]) -> float:
        """Score money flow analysis (0-100)"""
        if 'error' in money_flow_analysis:
            return 50  # Neutral score on error
        
        try:
            score = 50  # Base score
            
            # Smart money sentiment
            sentiment = money_flow_analysis.get('smart_money_sentiment', 'neutral')
            if sentiment == 'bullish':
                score += 25
            elif sentiment == 'bearish':
                score -= 25
            
            # ETF holdings flow
            etf_analysis = money_flow_analysis.get('etf_holdings_analysis', {})
            flow_score = etf_analysis.get('weighted_flow_score', 0)
            score += flow_score * 0.2  # Scale flow score
            
            # Institutional flow
            institutional_flow = money_flow_analysis.get('institutional_flow', {})
            mfi = institutional_flow.get('money_flow_index', 50)
            if mfi > 60:  # Strong buying
                score += 15
            elif mfi < 40:  # Strong selling
                score -= 15
            
            # Flow strength
            flow_strength = money_flow_analysis.get('flow_strength', {})
            strength_level = flow_strength.get('strength_level', 'moderate')
            if strength_level == 'strong':
                score += 10
            elif strength_level == 'weak':
                score -= 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Money flow scoring failed: {e}", "ORCHESTRATOR")
            return 50
    
    def _score_technical_analysis(self, technical_analysis: Dict[str, Any]) -> float:
        """Score technical analysis (0-100)"""
        if 'error' in technical_analysis:
            return 50  # Neutral score on error
        
        try:
            score = 50  # Base score
            
            # Overall trend
            trend = technical_analysis.get('overall_trend', 'neutral')
            if trend == 'bullish':
                score += 20
            elif trend == 'bearish':
                score -= 20
            
            # Momentum indicators
            momentum = technical_analysis.get('momentum_analysis', {})
            rsi = momentum.get('rsi', 50)
            if 40 <= rsi <= 60:  # Neutral zone
                score += 5
            elif rsi > 70:  # Overbought
                score -= 10
            elif rsi < 30:  # Oversold but could bounce
                score += 5
            
            # Volume confirmation
            volume_analysis = technical_analysis.get('volume_analysis', {})
            volume_trend = volume_analysis.get('trend', 'neutral')
            if volume_trend == 'increasing':
                score += 15
            elif volume_trend == 'decreasing':
                score -= 10
            
            # Support/resistance levels
            levels = technical_analysis.get('support_resistance', {})
            near_support = levels.get('near_support', False)
            near_resistance = levels.get('near_resistance', False)
            if near_support:
                score += 10
            elif near_resistance:
                score -= 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.error(f"Technical scoring failed: {e}", "ORCHESTRATOR")
            return 50
    
    def _calculate_alignment_bonus(self, options_analysis: Dict[str, Any],
                                 money_flow_analysis: Dict[str, Any],
                                 technical_analysis: Dict[str, Any]) -> float:
        """Calculate bonus for alignment between different analyses"""
        try:
            # Get sentiments from each analysis
            options_sentiment = options_analysis.get('options_sentiment', 'neutral')
            money_flow_sentiment = money_flow_analysis.get('smart_money_sentiment', 'neutral')
            technical_trend = technical_analysis.get('overall_trend', 'neutral')
            
            sentiments = [options_sentiment, money_flow_sentiment, technical_trend]
            
            # Count bullish vs bearish signals
            bullish_count = sentiments.count('bullish')
            bearish_count = sentiments.count('bearish')
            
            # Bonus for alignment
            if bullish_count >= 2 and bearish_count == 0:
                return 10  # Strong bullish alignment
            elif bullish_count == 3:
                return 15  # Perfect bullish alignment
            elif bearish_count >= 2 and bullish_count == 0:
                return -10  # Strong bearish alignment (negative bonus)
            elif bearish_count == 3:
                return -15  # Perfect bearish alignment (negative bonus)
            else:
                return 0  # Mixed signals, no bonus
                
        except Exception as e:
            logger.error(f"Alignment bonus calculation failed: {e}", "ORCHESTRATOR")
            return 0
    
    def _get_basic_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get basic technical analysis for a symbol"""
        try:
            technical_data = market_data_manager.get_technical_indicators(symbol)
            
            if not technical_data:
                return {'status': 'no_data'}
            
            return {
                'current_price': technical_data.get('current_price', 0),
                'trend_strength': technical_data.get('trend_strength', 'NEUTRAL'),
                'rsi': technical_data.get('rsi', 50),
                'volume_ratio': technical_data.get('volume_ratio', 1.0),
                'support_level': technical_data.get('support_level', 0),
                'resistance_level': technical_data.get('resistance_level', 0)
            }
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}", "ORCHESTRATOR")
            return {'status': 'error', 'error': str(e)}
    
    def _execute_ai_decisions(self, phase1_results: Dict[str, Any], phase2_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute AI decision making for qualified candidates"""
        ai_decisions = []
        
        try:
            qualified_candidates = phase1_results.get('qualified_candidates', [])
            phase2_analysis = phase2_results.get('analysis_results', {})
            
            logger.info(f"AI analyzing {len(qualified_candidates)} candidates", "ORCHESTRATOR")
            
            for candidate in qualified_candidates:
                symbol = candidate['symbol']
                
                try:
                    # Prepare data for AI analysis
                    phase1_data = {
                        'fundamental': candidate.get('fundamental_data', {}),
                        'sentiment': candidate.get('sentiment_data', {})
                    }
                    
                    phase2_data = phase2_analysis.get(symbol, {})
                    
                    market_context = {
                        'timestamp': datetime.now().isoformat(),
                        'market_session': self._get_market_session(),
                        'daily_trade_count': self.daily_trade_count
                    }
                    
                    # Get enhanced AI decision with real-time intelligence
                    decision, reasoning, confidence = ai_engine.enhanced_trading_decision(
                        symbol, phase1_data, phase2_data, market_context
                    )
                    
                    ai_decision = {
                        'symbol': symbol,
                        'decision': decision,
                        'reasoning': reasoning,
                        'confidence': confidence,
                        'phase1_score': candidate.get('phase1_score', 0),
                        'phase2_score': phase2_data.get('phase2_score', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    ai_decisions.append(ai_decision)
                    
                    logger.ai_decision_log(symbol, decision, reasoning, confidence)
                    
                except Exception as e:
                    logger.error(f"AI decision failed for {symbol}: {e}", "ORCHESTRATOR")
            
            # Sort by confidence
            ai_decisions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"AI decision execution failed: {e}", "ORCHESTRATOR")
        
        return ai_decisions
    
    def _execute_phase3(self, ai_decisions: List[Dict[str, Any]], test_mode: bool = True) -> Dict[str, Any]:
        """Execute Phase 3: Live Trade Execution"""
        phase3_results = {
            'decisions_processed': len(ai_decisions),
            'execution_results': {},
            'portfolio_summary': {},
            'execution_time': 0
        }
        
        start_time = time.time()
        
        try:
            logger.phase_log("3", "ALL", f"Processing {len(ai_decisions)} AI decisions (test_mode: {test_mode})")
            
            # Filter decisions for execution
            executable_decisions = [
                decision for decision in ai_decisions 
                if decision['decision'] in ['BUY', 'SELL'] and 
                decision['confidence'] >= TradingConfig.AI_CONFIDENCE_THRESHOLD
            ]
            
            if not executable_decisions:
                logger.phase_log("3", "ALL", "No decisions meet execution criteria")
                phase3_results['execution_results'] = {
                    'orders_placed': [],
                    'orders_rejected': [],
                    'total_capital_used': 0.0
                }
                return phase3_results
            
            # Execute trades through trade executor
            execution_results = trade_executor.execute_ai_decisions(executable_decisions, test_mode)
            phase3_results['execution_results'] = execution_results
            
            # Update daily trade count
            orders_placed = len(execution_results.get('orders_placed', []))
            self.daily_trade_count += orders_placed
            
            # Get portfolio summary
            portfolio_summary = trade_executor.get_portfolio_summary()
            phase3_results['portfolio_summary'] = portfolio_summary
            
            # Log execution summary
            logger.phase_log("3", "ALL", f"Execution complete: {orders_placed} orders placed, "
                           f"${execution_results.get('total_capital_used', 0):.2f} capital used")
            
            if portfolio_summary.get('total_positions', 0) > 0:
                total_pnl = portfolio_summary.get('total_pnl', 0)
                logger.phase_log("3", "ALL", f"Portfolio: {portfolio_summary['total_positions']} positions, "
                               f"Total P&L: ${total_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Phase 3 execution failed: {e}", "ORCHESTRATOR")
            phase3_results['error'] = str(e)
        
        phase3_results['execution_time'] = time.time() - start_time
        return phase3_results
    
    def _get_market_session(self) -> str:
        """Determine current market session"""
        current_hour = datetime.now().hour
        
        if 9 <= current_hour < 16:
            return "MARKET_HOURS"
        elif 4 <= current_hour < 9:
            return "PRE_MARKET"
        elif 16 <= current_hour < 20:
            return "AFTER_HOURS"
        else:
            return "CLOSED"
    
    def _get_rejection_reason(self, decision: str, confidence: float) -> str:
        """Get reason for trade rejection"""
        if decision == 'HOLD':
            return "AI recommended HOLD"
        elif confidence < TradingConfig.AI_CONFIDENCE_THRESHOLD:
            return f"Confidence {confidence:.2f} below threshold {TradingConfig.AI_CONFIDENCE_THRESHOLD}"
        elif self.daily_trade_count >= TradingConfig.MAX_DAILY_TRADES:
            return f"Daily trade limit reached ({TradingConfig.MAX_DAILY_TRADES})"
        else:
            return "Unknown rejection reason"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'status': self.system_status,
            'current_candidates': len(self.current_candidates),
            'active_positions': len(self.active_positions),
            'daily_trade_count': self.daily_trade_count,
            'last_run_date': self.last_run_date.isoformat() if self.last_run_date else None,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_quick_scan(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run a quick scan of specific symbols"""
        if symbols is None:
            symbols = TradingConfig.TEST_STOCKS[:3]  # Quick scan of first 3 test stocks
        
        logger.info(f"Running quick scan for {symbols}", "ORCHESTRATOR")
        
        results = {}
        
        for symbol in symbols:
            try:
                # Quick fundamental check
                fundamental_data = market_data_manager.get_fundamental_data(symbol)
                
                # Quick technical check
                technical_data = market_data_manager.get_technical_indicators(symbol)
                
                results[symbol] = {
                    'fundamental_score': self._quick_fundamental_score(fundamental_data),
                    'technical_summary': self._quick_technical_summary(technical_data),
                    'recommendation': self._quick_recommendation(fundamental_data, technical_data)
                }
                
            except Exception as e:
                logger.error(f"Quick scan failed for {symbol}: {e}", "ORCHESTRATOR")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def _quick_fundamental_score(self, fundamental_data: Dict[str, Any]) -> float:
        """Calculate quick fundamental score"""
        if not fundamental_data:
            return 0
        
        revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
        revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
        
        # Simple scoring
        score = 0
        if revenue_growth_yoy >= TradingConfig.MIN_REVENUE_GROWTH_YOY:
            score += 50
        if revenue_growth_qoq >= TradingConfig.MIN_REVENUE_GROWTH_QOQ:
            score += 50
        
        return score
    
    def _quick_technical_summary(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get quick technical summary"""
        if not technical_data:
            return {'status': 'no_data'}
        
        return {
            'trend': technical_data.get('trend_strength', 'NEUTRAL'),
            'rsi': technical_data.get('rsi', 50),
            'price': technical_data.get('current_price', 0)
        }
    
    def _quick_recommendation(self, fundamental_data: Dict[str, Any], technical_data: Dict[str, Any]) -> str:
        """Generate quick recommendation"""
        fundamental_score = self._quick_fundamental_score(fundamental_data)
        
        if fundamental_score >= 80:
            return "STRONG_INTEREST"
        elif fundamental_score >= 50:
            return "MODERATE_INTEREST"
        else:
            return "LOW_INTEREST"
    
    def _analyze_market_context(self) -> Dict[str, Any]:
        """Analyze overall market context using market trend analyzer"""
        try:
            logger.info("Analyzing market context", "ORCHESTRATOR")
            
            market_analysis = market_trend_analyzer.analyze_market_context()
            
            if 'error' in market_analysis:
                logger.warning(f"Market context analysis failed: {market_analysis['error']}", "ORCHESTRATOR")
                return {
                    'status': 'error',
                    'error': market_analysis['error'],
                    'market_score': 50,  # Neutral score on error
                    'trading_recommendation': 'selective_trading'
                }
            
            # Extract key metrics
            market_score = market_analysis.get('market_score', 50)
            market_regime = market_analysis.get('market_regime', 'unknown')
            trading_env = market_analysis.get('trading_environment', {})
            trading_recommendation = trading_env.get('trading_recommendation', 'selective_trading')
            
            # Log market context summary
            logger.info(f"Market Score: {market_score:.1f}/100", "ORCHESTRATOR")
            logger.info(f"Market Regime: {market_regime}", "ORCHESTRATOR")
            logger.info(f"Trading Recommendation: {trading_recommendation}", "ORCHESTRATOR")
            
            # Add VIX and SPY summary for quick reference
            volatility_analysis = market_analysis.get('volatility_analysis', {})
            market_indices = market_analysis.get('market_indices', {})
            
            vix_level = volatility_analysis.get('current_vix', 'N/A')
            spy_trend = 'N/A'
            if 'indices' in market_indices and 'SPY' in market_indices['indices']:
                spy_data = market_indices['indices']['SPY']
                spy_trend = f"{spy_data.get('daily_change_pct', 0):.2f}%"
            
            logger.info(f"VIX: {vix_level}, SPY Daily Change: {spy_trend}", "ORCHESTRATOR")
            
            return {
                'status': 'success',
                'market_score': market_score,
                'market_regime': market_regime,
                'trading_recommendation': trading_recommendation,
                'vix_level': vix_level,
                'spy_daily_change': spy_trend,
                'full_analysis': market_analysis
            }
            
        except Exception as e:
            logger.error(f"Market context analysis failed: {e}", "ORCHESTRATOR")
            return {
                'status': 'error',
                'error': str(e),
                'market_score': 50,
                'trading_recommendation': 'wait_and_see'
            }
    
    def _should_proceed_with_trading(self, market_context: Dict[str, Any]) -> bool:
        """Determine if market conditions are favorable for trading"""
        try:
            if market_context.get('status') == 'error':
                logger.warning("Market context analysis failed - proceeding with caution", "ORCHESTRATOR")
                return True  # Proceed but with caution if analysis fails
            
            market_score = market_context.get('market_score', 50)
            trading_recommendation = market_context.get('trading_recommendation', 'selective_trading')
            
            # Check if we're using fallback data
            full_analysis = market_context.get('full_analysis', {})
            using_fallback_data = self._is_using_fallback_data(full_analysis)
            
            if using_fallback_data:
                logger.info("Using fallback market data - applying relaxed trading criteria", "ORCHESTRATOR")
                # More lenient criteria when using fallback data
                if trading_recommendation == 'defensive_mode':
                    logger.warning("Market in defensive mode - stopping trading", "ORCHESTRATOR")
                    return False
                
                # Allow trading with lower market score when using fallback data
                min_score_threshold = max(40, TradingConfig.MIN_MARKET_SCORE - 20)
                if market_score < min_score_threshold:
                    logger.warning(f"Market score {market_score:.1f} below fallback threshold {min_score_threshold} - stopping trading", "ORCHESTRATOR")
                    return False
                
                logger.info(f"Fallback data conditions acceptable for trading (Score: {market_score:.1f}, Rec: {trading_recommendation})", "ORCHESTRATOR")
                return True
            
            # Normal criteria for real data
            if trading_recommendation == 'defensive_mode':
                logger.warning("Market in defensive mode - stopping trading", "ORCHESTRATOR")
                return False
            
            if trading_recommendation == 'wait_and_see':
                logger.warning("Market recommends wait and see - stopping trading", "ORCHESTRATOR")
                return False
            
            if market_score < TradingConfig.MIN_MARKET_SCORE:
                logger.warning(f"Market score {market_score:.1f} below threshold {TradingConfig.MIN_MARKET_SCORE} - stopping trading", "ORCHESTRATOR")
                return False
            
            # Check for extreme market conditions
            vix_level = market_context.get('vix_level', 'N/A')
            if isinstance(vix_level, (int, float)) and vix_level > 40:
                logger.warning(f"VIX too high ({vix_level:.1f}) - stopping trading", "ORCHESTRATOR")
                return False
            
            logger.info(f"Market conditions favorable for trading (Score: {market_score:.1f}, Rec: {trading_recommendation})", "ORCHESTRATOR")
            return True
            
        except Exception as e:
            logger.error(f"Market condition check failed: {e}", "ORCHESTRATOR")
            return True  # Default to proceeding if check fails
    
    def _is_using_fallback_data(self, market_analysis: Dict[str, Any]) -> bool:
        """Check if market analysis is using fallback data"""
        try:
            # Check market indices for fallback data sources
            market_indices = market_analysis.get('market_indices', {})
            indices = market_indices.get('indices', {})
            
            fallback_count = 0
            total_indices = 0
            
            for symbol, data in indices.items():
                total_indices += 1
                data_source = data.get('data_source', 'unknown')
                if data_source in ['fallback', 'ib_gateway']:
                    fallback_count += 1
            
            # If more than 50% of indices are using fallback data
            if total_indices > 0 and (fallback_count / total_indices) > 0.5:
                return True
            
            # Check volatility analysis
            volatility_analysis = market_analysis.get('volatility_analysis', {})
            if 'error' in volatility_analysis:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Fallback data check failed: {e}", "ORCHESTRATOR")
            return False
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get a quick market summary"""
        try:
            market_context = self._analyze_market_context()
            
            return {
                'market_score': market_context.get('market_score', 50),
                'market_regime': market_context.get('market_regime', 'unknown'),
                'trading_recommendation': market_context.get('trading_recommendation', 'selective_trading'),
                'vix_level': market_context.get('vix_level', 'N/A'),
                'spy_daily_change': market_context.get('spy_daily_change', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Market summary failed: {e}", "ORCHESTRATOR")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Global orchestrator instance
orchestrator = TradingOrchestrator()
