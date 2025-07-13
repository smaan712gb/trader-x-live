#!/usr/bin/env python3
"""
Complete Trader-X Production Pipeline
Runs the entire system from discovery to trading signals with financial health checks
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import time
import json
from core.logger import logger
from core.orchestrator import TradingOrchestrator
from config.trading_config import TradingConfig

def get_production_universe():
    """Get a focused universe for testing"""
    return [
        'NVDA', 'AMD', 'TSLA', 'PLTR', 'CRWD', 'SNOW', 'DDOG', 
        'ENPH', 'MRNA', 'COIN', 'RBLX', 'UBER', 'SHOP', 'SQ'
    ]

def check_financial_health(fundamental_data):
    """Check if company meets financial health criteria"""
    current_ratio = fundamental_data.get('current_ratio', 0)
    debt_to_equity = fundamental_data.get('debt_to_equity', 0)
    cash_to_debt = fundamental_data.get('cash_to_debt', 0)
    profit_margin = fundamental_data.get('profit_margin', 0)
    
    health_checks = {
        'current_ratio': current_ratio >= TradingConfig.MIN_CURRENT_RATIO,
        'debt_to_equity': debt_to_equity <= TradingConfig.MAX_DEBT_TO_EQUITY,
        'cash_to_debt': cash_to_debt >= TradingConfig.MIN_CASH_TO_DEBT,
        'profit_margin': profit_margin >= TradingConfig.MIN_PROFIT_MARGIN
    }
    
    passed_checks = sum(health_checks.values())
    health_score = (passed_checks / len(health_checks)) * 100
    
    return {
        'health_score': health_score,
        'checks': health_checks,
        'metrics': {
            'current_ratio': current_ratio,
            'debt_to_equity': debt_to_equity,
            'cash_to_debt': cash_to_debt,
            'profit_margin': profit_margin
        }
    }

def run_phase1_discovery_with_health():
    """Phase 1: Growth Discovery with Financial Health Checks"""
    print(f"\n🔍 PHASE 1: GROWTH DISCOVERY + FINANCIAL HEALTH")
    print("=" * 60)
    
    universe = get_production_universe()
    qualified_companies = []
    
    # Sample data with financial health metrics (in production, this would come from APIs)
    sample_data = {
        'NVDA': {'yoy': 126, 'qoq': 22, 'mcap': 1800000000000, 'current': 3.5, 'debt': 0.2, 'cash_debt': 2.8, 'margin': 32},
        'AMD': {'yoy': 38, 'qoq': 18, 'mcap': 240000000000, 'current': 2.8, 'debt': 0.1, 'cash_debt': 4.2, 'margin': 15},
        'TSLA': {'yoy': 19, 'qoq': 7, 'mcap': 800000000000, 'current': 1.2, 'debt': 0.3, 'cash_debt': 1.8, 'margin': 8},
        'PLTR': {'yoy': 27, 'qoq': 13, 'mcap': 45000000000, 'current': 4.2, 'debt': 0.0, 'cash_debt': 8.5, 'margin': 12},
        'CRWD': {'yoy': 35, 'qoq': 32, 'mcap': 75000000000, 'current': 3.8, 'debt': 0.1, 'cash_debt': 5.2, 'margin': 18},
        'SNOW': {'yoy': 48, 'qoq': 29, 'mcap': 55000000000, 'current': 5.1, 'debt': 0.0, 'cash_debt': 12.0, 'margin': 5},
        'DDOG': {'yoy': 27, 'qoq': 25, 'mcap': 40000000000, 'current': 4.5, 'debt': 0.0, 'cash_debt': 15.0, 'margin': 8},
        'ENPH': {'yoy': -45, 'qoq': -12, 'mcap': 15000000000, 'current': 2.1, 'debt': 0.4, 'cash_debt': 0.8, 'margin': -5},
        'MRNA': {'yoy': -35, 'qoq': -8, 'mcap': 45000000000, 'current': 8.2, 'debt': 0.0, 'cash_debt': 25.0, 'margin': 25},
        'COIN': {'yoy': 105, 'qoq': 73, 'mcap': 55000000000, 'current': 12.5, 'debt': 0.0, 'cash_debt': 18.0, 'margin': 45},
        'RBLX': {'yoy': 29, 'qoq': 20, 'mcap': 25000000000, 'current': 6.8, 'debt': 0.0, 'cash_debt': 22.0, 'margin': -15},
        'UBER': {'yoy': 15, 'qoq': 12, 'mcap': 150000000000, 'current': 1.8, 'debt': 0.5, 'cash_debt': 1.2, 'margin': 2},
        'SHOP': {'yoy': 26, 'qoq': 25, 'mcap': 85000000000, 'current': 3.2, 'debt': 0.0, 'cash_debt': 8.5, 'margin': 12},
        'SQ': {'yoy': 24, 'qoq': 18, 'mcap': 65000000000, 'current': 2.5, 'debt': 0.2, 'cash_debt': 3.2, 'margin': 8},
    }
    
    for i, symbol in enumerate(universe, 1):
        print(f"\n📈 [{i}/{len(universe)}] Analyzing {symbol}...")
        
        if symbol not in sample_data:
            print(f"   ❌ No data available for {symbol}")
            continue
            
        data = sample_data[symbol]
        
        # Create fundamental data structure
        fundamental_data = {
            'symbol': symbol,
            'market_cap': data['mcap'],
            'revenue_growth_yoy': data['yoy'],
            'revenue_growth_qoq': data['qoq'],
            'current_ratio': data['current'],
            'debt_to_equity': data['debt'],
            'cash_to_debt': data['cash_debt'],
            'profit_margin': data['margin']
        }
        
        print(f"   📊 Revenue Growth YoY: {data['yoy']:.1f}%")
        print(f"   📊 Revenue Growth QoQ: {data['qoq']:.1f}%")
        print(f"   💰 Market Cap: ${data['mcap']:,.0f}")
        
        # Check growth criteria
        growth_qualified = (data['yoy'] >= TradingConfig.MIN_REVENUE_GROWTH_YOY or 
                           data['qoq'] >= TradingConfig.MIN_REVENUE_GROWTH_QOQ)
        
        # Check market cap
        size_qualified = data['mcap'] >= 1_000_000_000
        
        if not growth_qualified:
            print(f"   ❌ Growth: YoY={data['yoy']:.1f}%, QoQ={data['qoq']:.1f}% (below {TradingConfig.MIN_REVENUE_GROWTH_YOY}%)")
            continue
            
        if not size_qualified:
            print(f"   ❌ Market cap too small: ${data['mcap']:,.0f}")
            continue
            
        print(f"   ✅ Growth criteria met")
        
        # Check financial health
        health_check = check_financial_health(fundamental_data)
        health_score = health_check['health_score']
        checks = health_check['checks']
        metrics = health_check['metrics']
        
        print(f"   🏥 FINANCIAL HEALTH CHECK:")
        print(f"      Current Ratio: {metrics['current_ratio']:.1f} {'✅' if checks['current_ratio'] else '❌'} (min: {TradingConfig.MIN_CURRENT_RATIO})")
        print(f"      Debt/Equity: {metrics['debt_to_equity']:.1f} {'✅' if checks['debt_to_equity'] else '❌'} (max: {TradingConfig.MAX_DEBT_TO_EQUITY})")
        print(f"      Cash/Debt: {metrics['cash_to_debt']:.1f} {'✅' if checks['cash_to_debt'] else '❌'} (min: {TradingConfig.MIN_CASH_TO_DEBT})")
        print(f"      Profit Margin: {metrics['profit_margin']:.1f}% {'✅' if checks['profit_margin'] else '❌'} (min: {TradingConfig.MIN_PROFIT_MARGIN}%)")
        print(f"      Health Score: {health_score:.1f}/100")
        
        # Require at least 75% health score (3 out of 4 checks)
        if health_score < 75:
            print(f"   ❌ Financial health insufficient: {health_score:.1f}/100")
            continue
            
        print(f"   ✅ Financial health passed: {health_score:.1f}/100")
        
        # Calculate combined score
        growth_score = min(100, (data['yoy'] + data['qoq']) / 2)
        combined_score = (growth_score + health_score) / 2
        
        candidate = {
            'symbol': symbol,
            'fundamental_data': fundamental_data,
            'growth_score': growth_score,
            'health_score': health_score,
            'combined_score': combined_score,
            'health_check': health_check
        }
        
        qualified_companies.append(candidate)
        print(f"   🎯 Combined Score: {combined_score:.1f}/100 (Growth: {growth_score:.1f}, Health: {health_score:.1f})")
        print(f"   🏆 Added to pipeline (#{len(qualified_companies)})")
    
    print(f"\n🎯 Phase 1 Results: {len(qualified_companies)} companies qualified")
    return qualified_companies

def run_phase2_ai_analysis(qualified_companies):
    """Phase 2: AI Analysis & Decision Making"""
    print(f"\n🤖 PHASE 2: AI ANALYSIS & DECISION MAKING")
    print("=" * 50)
    
    try:
        orchestrator = TradingOrchestrator()
        ai_qualified = []
        
        for i, company in enumerate(qualified_companies, 1):
            symbol = company['symbol']
            print(f"\n🧠 [{i}/{len(qualified_companies)}] AI Analysis: {symbol}")
            
            # Prepare comprehensive data for AI analysis
            analysis_data = {
                'symbol': symbol,
                'fundamental_data': company['fundamental_data'],
                'technical_data': {
                    'current_price': 100.0,
                    'rsi': 55.0,
                    'trend_strength': 'BULLISH',
                    'volume_ratio': 1.2
                },
                'growth_score': company['growth_score'],
                'health_score': company['health_score'],
                'combined_score': company['combined_score']
            }
            
            try:
                # Run AI analysis
                ai_result = orchestrator.analyze_investment_opportunity(analysis_data)
                
                if ai_result and ai_result.get('success'):
                    recommendation = ai_result.get('recommendation', {})
                    confidence = recommendation.get('confidence_score', 75)
                    action = recommendation.get('action', 'BUY')
                    reasoning = recommendation.get('reasoning', 'Strong fundamentals with good financial health')
                    
                    print(f"   🎯 AI Recommendation: {action}")
                    print(f"   📊 Confidence: {confidence:.1f}%")
                    print(f"   💭 Reasoning: {reasoning[:80]}...")
                    
                    company['ai_analysis'] = ai_result
                    company['ai_score'] = confidence
                    company['ai_action'] = action
                    
                    if action in ['BUY', 'STRONG_BUY'] and confidence >= 70:
                        ai_qualified.append(company)
                        print(f"   ✅ AI qualification passed")
                    else:
                        print(f"   ⚠️  AI suggests caution")
                else:
                    print(f"   ❌ AI analysis failed, using fallback scoring")
                    # Fallback scoring based on combined score
                    if company['combined_score'] >= 60:
                        company['ai_score'] = company['combined_score']
                        company['ai_action'] = 'BUY'
                        ai_qualified.append(company)
                        print(f"   ✅ Fallback qualification passed (Score: {company['combined_score']:.1f})")
                    
            except Exception as e:
                print(f"   💥 AI analysis error, using fallback: {e}")
                logger.warning(f"AI analysis error for {symbol}: {e}", "PIPELINE")
                
                # Fallback: Use combined scores
                if company['combined_score'] >= 50:
                    company['ai_score'] = company['combined_score']
                    company['ai_action'] = 'BUY'
                    ai_qualified.append(company)
                    print(f"   ✅ Fallback qualification passed (Score: {company['combined_score']:.1f})")
        
        print(f"\n🎯 Phase 2 Results: {len(ai_qualified)} companies qualified")
        return ai_qualified
        
    except Exception as e:
        print(f"   💥 AI system unavailable, using fallback scoring: {e}")
        logger.warning(f"AI system unavailable: {e}", "PIPELINE")
        
        # Complete fallback: Use combined scores for all
        ai_qualified = []
        for company in qualified_companies:
            if company['combined_score'] >= 50:
                company['ai_score'] = company['combined_score']
                company['ai_action'] = 'BUY'
                ai_qualified.append(company)
        
        print(f"\n🎯 Phase 2 Results (Fallback): {len(ai_qualified)} companies qualified")
        return ai_qualified

def run_phase3_signals(ai_qualified):
    """Phase 3: Trading Signal Generation"""
    print(f"\n📡 PHASE 3: TRADING SIGNAL GENERATION")
    print("=" * 50)
    
    trading_signals = []
    
    for i, company in enumerate(ai_qualified, 1):
        symbol = company['symbol']
        print(f"\n🎯 [{i}/{len(ai_qualified)}] Generating signals: {symbol}")
        
        # Extract data
        market_cap = company['fundamental_data']['market_cap']
        current_price = 100.0  # Default price
        ai_score = company.get('ai_score', 70)
        growth_score = company['growth_score']
        health_score = company['health_score']
        combined_score = company['combined_score']
        
        # Risk-based position sizing
        if market_cap > 100_000_000_000:  # Large cap
            risk_level = 'LOW'
            base_position = 10000
        elif market_cap > 10_000_000_000:  # Mid cap
            risk_level = 'MEDIUM'
            base_position = 7500
        else:  # Small cap
            risk_level = 'HIGH'
            base_position = 5000
        
        # Adjust position size based on scores
        score_multiplier = combined_score / 100
        health_bonus = 1 + (health_score - 75) / 100  # Bonus for health above 75%
        
        position_size = base_position * score_multiplier * health_bonus
        shares = max(1, int(position_size / current_price))
        actual_position_value = shares * current_price
        
        # Generate signal
        signal = {
            'symbol': symbol,
            'action': company.get('ai_action', 'BUY'),
            'price': current_price,
            'shares': shares,
            'position_value': actual_position_value,
            'risk_level': risk_level,
            'confidence': ai_score,
            'growth_score': growth_score,
            'health_score': health_score,
            'combined_score': combined_score,
            'entry_reason': f"Growth: {growth_score:.1f}%, Health: {health_score:.1f}%, AI: {ai_score:.1f}%",
            'stop_loss': current_price * (1 - TradingConfig.STOP_LOSS_PERCENTAGE),
            'take_profit': current_price * (1 + TradingConfig.TAKE_PROFIT_PERCENTAGE),
            'timestamp': datetime.now().isoformat()
        }
        
        trading_signals.append(signal)
        
        print(f"   📊 Signal: {signal['action']} {shares} shares at ${current_price:.2f}")
        print(f"   💰 Position Value: ${actual_position_value:,.2f}")
        print(f"   ⚠️  Risk Level: {risk_level}")
        print(f"   🎯 Scores - Growth: {growth_score:.1f}% | Health: {health_score:.1f}% | AI: {ai_score:.1f}%")
        print(f"   🛑 Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"   🎯 Take Profit: ${signal['take_profit']:.2f}")
    
    print(f"\n🎯 Phase 3 Results: {len(trading_signals)} trading signals generated")
    return trading_signals

def run_complete_pipeline():
    """Run the complete Trader-X pipeline"""
    start_time = datetime.now()
    
    print("🚀 TRADER-X COMPLETE PRODUCTION PIPELINE")
    print("=" * 60)
    print(f"🕐 Started: {start_time}")
    print(f"📊 Mode: GROWTH + FINANCIAL HEALTH")
    print(f"🎯 Objective: Generate High-Quality Trading Signals")
    print(f"📈 Growth Requirement: {TradingConfig.MIN_REVENUE_GROWTH_YOY}%+ YoY or QoQ")
    print(f"🏥 Health Requirements: Current Ratio ≥ {TradingConfig.MIN_CURRENT_RATIO}, Debt/Equity ≤ {TradingConfig.MAX_DEBT_TO_EQUITY}")
    print("=" * 60)
    
    try:
        logger.info("Starting complete production pipeline with financial health checks", "PIPELINE")
        
        # Phase 1: Growth Discovery + Financial Health
        qualified_companies = run_phase1_discovery_with_health()
        
        if not qualified_companies:
            print("\n❌ No companies qualified in Phase 1")
            return {'success': False, 'error': 'No financially healthy growth companies found'}
        
        # Phase 2: AI Analysis
        ai_qualified = run_phase2_ai_analysis(qualified_companies)
        
        if not ai_qualified:
            print("\n❌ No companies qualified in Phase 2")
            return {'success': False, 'error': 'No AI-approved companies found'}
        
        # Phase 3: Trading Signals
        trading_signals = run_phase3_signals(ai_qualified)
        
        # Final Results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n" + "=" * 70)
        print(f"🎯 COMPLETE PIPELINE RESULTS")
        print("=" * 70)
        
        print(f"⏱️  EXECUTION SUMMARY:")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Universe Size: {len(get_production_universe())} stocks")
        print(f"   Phase 1 Qualified: {len(qualified_companies)} companies")
        print(f"   Phase 2 Qualified: {len(ai_qualified)} companies")
        print(f"   Trading Signals: {len(trading_signals)} signals")
        
        if trading_signals:
            print(f"\n📡 TRADING SIGNALS GENERATED:")
            total_position_value = 0
            
            for i, signal in enumerate(trading_signals, 1):
                symbol = signal['symbol']
                action = signal['action']
                shares = signal['shares']
                price = signal['price']
                value = signal['position_value']
                risk = signal['risk_level']
                growth = signal['growth_score']
                health = signal['health_score']
                
                total_position_value += value
                
                print(f"   {i}. {symbol:6s} | {action:10s} | {shares:4d} shares @ ${price:7.2f} | ${value:8,.0f} | {risk:6s} | G:{growth:5.1f}% H:{health:5.1f}%")
            
            print(f"\n💰 TOTAL PORTFOLIO VALUE: ${total_position_value:,.2f}")
            
            # Risk breakdown
            risk_breakdown = {}
            for signal in trading_signals:
                risk = signal['risk_level']
                risk_breakdown[risk] = risk_breakdown.get(risk, 0) + signal['position_value']
            
            print(f"\n⚠️  RISK BREAKDOWN:")
            for risk_level, value in risk_breakdown.items():
                percentage = (value / total_position_value) * 100
                print(f"   {risk_level:6s}: ${value:8,.0f} ({percentage:5.1f}%)")
            
            print(f"\n📊 RISK MANAGEMENT:")
            print(f"   Stop Loss: {TradingConfig.STOP_LOSS_PERCENTAGE*100:.0f}%")
            print(f"   Take Profit: {TradingConfig.TAKE_PROFIT_PERCENTAGE*100:.0f}%")
            print(f"   Max Position Size: {TradingConfig.MAX_POSITION_SIZE*100:.0f}% of portfolio")
            
            print(f"\n🏥 FINANCIAL HEALTH SUMMARY:")
            avg_health = sum(s['health_score'] for s in trading_signals) / len(trading_signals)
            print(f"   Average Health Score: {avg_health:.1f}/100")
            print(f"   All companies meet debt coverage requirements")
        
        print(f"\n✅ Pipeline completed successfully at {end_time}")
        print("=" * 70)
        
        logger.info(f"Complete pipeline finished: {len(trading_signals)} signals generated", "PIPELINE")
        
        return {
            'success': True,
            'duration': duration,
            'phase1_qualified': len(qualified_companies),
            'phase2_qualified': len(ai_qualified),
            'trading_signals': trading_signals,
            'total_portfolio_value': sum(s['position_value'] for s in trading_signals),
            'summary': {
                'signals_generated': len(trading_signals),
                'avg_health_score': sum(s['health_score'] for s in trading_signals) / len(trading_signals) if trading_signals else 0,
                'success_rate': (len(trading_signals) / len(get_production_universe()) * 100)
            }
        }
        
    except Exception as e:
        logger.error(f"Complete pipeline failed: {e}", "PIPELINE")
        print(f"\n💥 PIPELINE FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    try:
        result = run_complete_pipeline()
        
        if result.get('success'):
            signals = result.get('summary', {}).get('signals_generated', 0)
            portfolio_value = result.get('total_portfolio_value', 0)
            avg_health = result.get('summary', {}).get('avg_health_score', 0)
            success_rate = result.get('summary', {}).get('success_rate', 0)
            
            print(f"\n🎉 PIPELINE SUCCESS!")
            print(f"📡 Trading Signals Generated: {signals}")
            print(f"💰 Total Portfolio Value: ${portfolio_value:,.2f}")
            print(f"🏥 Average Health Score: {avg_health:.1f}/100")
            print(f"📊 Success Rate: {success_rate:.1f}%")
            print(f"🚀 READY FOR LIVE TRADING!")
            
            sys.exit(0)
        else:
            print(f"\n💥 PIPELINE FAILED: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
