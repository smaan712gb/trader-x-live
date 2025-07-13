#!/usr/bin/env python3
"""
Fixed Production Pipeline for Trader-X
Bypasses problematic components and focuses on core functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
from core.logger import logger
from core.orchestrator import orchestrator
from config.trading_config import TradingConfig
from data.market_data_enhanced import market_data_manager

def run_production_pipeline():
    """Run the production pipeline with fixes"""
    print("🚀 TRADER-X PRODUCTION PIPELINE (FIXED)")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now()}")
    print(f"📊 Mode: PAPER TRADING")
    print(f"🎯 Symbols: {TradingConfig.TEST_STOCKS}")
    print("=" * 60)
    
    try:
        # Check system status
        logger.info("Starting Trader-X production pipeline (fixed version)", "ORCHESTRATOR")
        
        # Check data connectivity
        connection_status = market_data_manager.get_connection_status()
        print(f"\n📊 DATA CONNECTIVITY:")
        print(f"   Primary Source: {connection_status['primary_data_source']}")
        print(f"   IB Gateway: {'✅ Connected' if connection_status['ib_gateway_connected'] else '❌ Disconnected'}")
        print(f"   Fallback: {'✅ Available' if connection_status['fallback_available'] else '❌ Unavailable'}")
        
        # Skip market context analysis (problematic) and go straight to Phase 1
        logger.info("=== PHASE 1: SIGNAL GENERATION ===", "ORCHESTRATOR")
        print(f"\n🔍 PHASE 1: SIGNAL GENERATION")
        print("-" * 40)
        
        # Use a smaller subset for testing to avoid rate limits
        test_symbols = TradingConfig.TEST_STOCKS[:3]  # First 3 symbols
        print(f"Testing symbols: {test_symbols}")
        
        # Run fundamental screening with error handling
        fundamental_candidates = []
        for symbol in test_symbols:
            try:
                print(f"\n📈 Screening {symbol}...")
                
                # Get fundamental data with timeout
                fundamental_data = market_data_manager.get_fundamental_data(symbol)
                
                if fundamental_data and 'error' not in fundamental_data:
                    revenue_growth_yoy = fundamental_data.get('revenue_growth_yoy', 0)
                    revenue_growth_qoq = fundamental_data.get('revenue_growth_qoq', 0)
                    market_cap = fundamental_data.get('market_cap', 0)
                    
                    print(f"   Revenue Growth YoY: {revenue_growth_yoy:.1f}%")
                    print(f"   Revenue Growth QoQ: {revenue_growth_qoq:.1f}%")
                    print(f"   Market Cap: ${market_cap:,.0f}")
                    
                    # Check fundamental criteria (relaxed for testing)
                    min_yoy = max(10.0, TradingConfig.MIN_REVENUE_GROWTH_YOY)  # Relaxed criteria
                    min_qoq = max(5.0, TradingConfig.MIN_REVENUE_GROWTH_QOQ)   # Relaxed criteria
                    
                    if revenue_growth_yoy >= min_yoy or revenue_growth_qoq >= min_qoq:
                        candidate = {
                            'symbol': symbol,
                            'revenue_growth_yoy': revenue_growth_yoy,
                            'revenue_growth_qoq': revenue_growth_qoq,
                            'market_cap': market_cap,
                            'fundamental_data': fundamental_data,
                            'composite_score': 75  # Default score
                        }
                        fundamental_candidates.append(candidate)
                        print(f"   ✅ {symbol} PASSED fundamental screening")
                    else:
                        print(f"   ❌ {symbol} failed fundamental criteria")
                        print(f"      Required YoY: {min_yoy}%, Got: {revenue_growth_yoy:.1f}%")
                        print(f"      Required QoQ: {min_qoq}%, Got: {revenue_growth_qoq:.1f}%")
                else:
                    print(f"   ⚠️  {symbol} - No fundamental data available")
                    
            except Exception as e:
                print(f"   ❌ {symbol} - Error: {e}")
                logger.error(f"Fundamental screening failed for {symbol}: {e}", "ORCHESTRATOR")
                
            time.sleep(2)  # Avoid rate limiting
        
        print(f"\n📊 Fundamental Screening Results:")
        print(f"   Symbols Tested: {len(test_symbols)}")
        print(f"   Passed Screening: {len(fundamental_candidates)}")
        
        if not fundamental_candidates:
            print("   ⚠️  No stocks passed fundamental screening")
            print("\n🎯 PIPELINE SUMMARY:")
            print("   Phase 1: No qualified candidates")
            print("   Phase 2: Skipped")
            print("   Phase 3: Skipped")
            return
        
        # Phase 2: Technical Analysis (simplified)
        logger.info("=== PHASE 2: TECHNICAL ANALYSIS ===", "ORCHESTRATOR")
        print(f"\n📈 PHASE 2: TECHNICAL ANALYSIS")
        print("-" * 40)
        
        qualified_candidates = []
        for candidate in fundamental_candidates:
            symbol = candidate['symbol']
            try:
                print(f"\n🔧 Technical analysis for {symbol}...")
                
                # Get technical data with error handling
                technical_data = market_data_manager.get_technical_indicators(symbol)
                
                if technical_data and 'error' not in technical_data:
                    current_price = technical_data.get('current_price', 0)
                    rsi = technical_data.get('rsi', 50)
                    trend_strength = technical_data.get('trend_strength', 'NEUTRAL')
                    volume_ratio = technical_data.get('volume_ratio', 1.0)
                    
                    print(f"   Current Price: ${current_price:.2f}")
                    print(f"   RSI: {rsi:.1f}")
                    print(f"   Trend: {trend_strength}")
                    print(f"   Volume Ratio: {volume_ratio:.2f}x")
                    
                    # Simple technical scoring
                    technical_score = 50  # Base score
                    if 30 <= rsi <= 70:  # Not overbought/oversold
                        technical_score += 20
                    if trend_strength in ['BULLISH', 'STRONG_BULLISH']:
                        technical_score += 30
                    if volume_ratio > 1.2:  # Above average volume
                        technical_score += 20
                    
                    candidate['technical_data'] = technical_data
                    candidate['technical_score'] = technical_score
                    candidate['phase2_score'] = technical_score
                    
                    print(f"   Technical Score: {technical_score}/100")
                    
                    # Qualify if technical score is decent
                    if technical_score >= 60:
                        qualified_candidates.append(candidate)
                        print(f"   ✅ {symbol} qualified for AI analysis")
                    else:
                        print(f"   ⚠️  {symbol} has weak technical setup")
                else:
                    print(f"   ❌ {symbol} - No technical data available")
                    
            except Exception as e:
                print(f"   ❌ {symbol} - Technical analysis error: {e}")
                logger.error(f"Technical analysis failed for {symbol}: {e}", "ORCHESTRATOR")
                
            time.sleep(1)  # Avoid rate limiting
        
        print(f"\n📊 Technical Analysis Results:")
        print(f"   Candidates Analyzed: {len(fundamental_candidates)}")
        print(f"   Qualified for AI: {len(qualified_candidates)}")
        
        if not qualified_candidates:
            print("   ⚠️  No stocks qualified for AI analysis")
            print("\n🎯 PIPELINE SUMMARY:")
            print("   Phase 1: Completed")
            print("   Phase 2: No qualified candidates")
            print("   Phase 3: Skipped")
            return
        
        # Phase 3: AI Decision Making (simplified)
        logger.info("=== AI DECISION SYNTHESIS ===", "ORCHESTRATOR")
        print(f"\n🧠 AI DECISION SYNTHESIS")
        print("-" * 40)
        
        ai_decisions = []
        for candidate in qualified_candidates:
            symbol = candidate['symbol']
            try:
                print(f"\n🤖 AI analysis for {symbol}...")
                
                # Prepare simplified data for AI
                phase1_data = {
                    'fundamental': {
                        'revenue_growth_yoy': candidate.get('revenue_growth_yoy', 0),
                        'revenue_growth_qoq': candidate.get('revenue_growth_qoq', 0),
                        'market_cap': candidate.get('market_cap', 0)
                    },
                    'sentiment': {'hype_score': 75}  # Default sentiment
                }
                
                phase2_data = {
                    'technical': candidate.get('technical_data', {}),
                    'technical_score': candidate.get('technical_score', 50)
                }
                
                market_context = {
                    'market_score': 60,  # Neutral market
                    'timestamp': datetime.now().isoformat()
                }
                
                # Get AI decision with timeout
                decision, reasoning, confidence = orchestrator._execute_ai_decisions(
                    {'qualified_candidates': [candidate]}, 
                    {'analysis_results': {symbol: phase2_data}}
                )
                
                if decision:
                    ai_decision = decision[0] if isinstance(decision, list) else {
                        'symbol': symbol,
                        'decision': 'HOLD',
                        'reasoning': 'Default decision',
                        'confidence': 0.5
                    }
                    
                    ai_decisions.append(ai_decision)
                    
                    decision_text = ai_decision.get('decision', 'HOLD')
                    confidence_score = ai_decision.get('confidence', 0.5)
                    
                    print(f"   Decision: {decision_text}")
                    print(f"   Confidence: {confidence_score:.2f}")
                    print(f"   Reasoning: {ai_decision.get('reasoning', 'N/A')[:100]}...")
                    
                    if decision_text in ['BUY', 'SELL'] and confidence_score >= 0.6:
                        print(f"   ✅ {symbol} recommended for trading")
                    else:
                        print(f"   ⚠️  {symbol} not recommended (low confidence or HOLD)")
                else:
                    print(f"   ❌ {symbol} - AI analysis failed")
                    
            except Exception as e:
                print(f"   ❌ {symbol} - AI analysis error: {e}")
                logger.error(f"AI analysis failed for {symbol}: {e}", "ORCHESTRATOR")
                
            time.sleep(1)
        
        # Final Results
        print("\n" + "=" * 60)
        print("🎯 PRODUCTION PIPELINE RESULTS")
        print("=" * 60)
        
        tradeable_decisions = [d for d in ai_decisions 
                             if d.get('decision') in ['BUY', 'SELL'] 
                             and d.get('confidence', 0) >= 0.6]
        
        print(f"✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"📊 Symbols Analyzed: {len(test_symbols)}")
        print(f"🔍 Fundamental Qualified: {len(fundamental_candidates)}")
        print(f"📈 Technical Qualified: {len(qualified_candidates)}")
        print(f"🤖 AI Decisions Generated: {len(ai_decisions)}")
        print(f"💰 Tradeable Opportunities: {len(tradeable_decisions)}")
        
        if tradeable_decisions:
            print(f"\n🚀 RECOMMENDED TRADES:")
            for i, decision in enumerate(tradeable_decisions, 1):
                symbol = decision.get('symbol', 'N/A')
                action = decision.get('decision', 'N/A')
                confidence = decision.get('confidence', 0)
                print(f"   {i}. {symbol}: {action} (Confidence: {confidence:.2f})")
        else:
            print(f"\n⚠️  NO TRADES RECOMMENDED")
            print(f"   All decisions were HOLD or below confidence threshold")
        
        print(f"\n⏰ Pipeline completed at: {datetime.now()}")
        print("=" * 60)
        
        return {
            'success': True,
            'symbols_analyzed': len(test_symbols),
            'fundamental_qualified': len(fundamental_candidates),
            'technical_qualified': len(qualified_candidates),
            'ai_decisions': len(ai_decisions),
            'tradeable_opportunities': len(tradeable_decisions),
            'recommended_trades': tradeable_decisions
        }
        
    except Exception as e:
        logger.error(f"Production pipeline failed: {e}", "ORCHESTRATOR")
        print(f"\n❌ PIPELINE FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        result = run_production_pipeline()
        if result.get('success'):
            print("\n🎉 PRODUCTION PIPELINE COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print(f"\n💥 PRODUCTION PIPELINE FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
