#!/usr/bin/env python3
"""
Production Runner for Trader-X
Runs the complete trading pipeline with real market data
"""
import sys
import argparse
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any
from core.orchestrator import orchestrator
from core.logger import logger
from config.trading_config import TradingConfig
from data.market_data import market_data_manager

def run_full_pipeline(symbols: List[str] = None, live_mode: bool = False) -> Dict[str, Any]:
    """
    Run the complete Trader-X pipeline with real data
    """
    start_time = datetime.now()
    
    print("🚀 TRADER-X PRODUCTION PIPELINE")
    print("=" * 60)
    print(f"🕐 Started at: {start_time}")
    print(f"📊 Mode: {'LIVE TRADING' if live_mode else 'PAPER TRADING'}")
    print(f"🎯 Symbols: {symbols or TradingConfig.TEST_STOCKS}")
    print("=" * 60)
    
    # Use default test stocks if none provided
    if not symbols:
        symbols = TradingConfig.TEST_STOCKS
    
    try:
        # Run the orchestrator with real data
        results = orchestrator.run_full_pipeline(test_mode=not live_mode)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display comprehensive results
        display_production_results(results, execution_time, live_mode)
        
        return results
        
    except Exception as e:
        logger.error(f"Production pipeline failed: {e}", "PRODUCTION")
        print(f"❌ Pipeline failed: {e}")
        return {"error": str(e)}

def run_market_scan(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Quick market scan for opportunities
    """
    print("🔍 TRADER-X MARKET SCAN")
    print("=" * 40)
    
    if not symbols:
        # Scan a broader universe for opportunities
        symbols = TradingConfig.TEST_STOCKS + [
            "AMZN", "GOOGL", "META", "NFLX", "CRM", "SNOW", "DDOG", "ZM"
        ]
    
    try:
        # Run quick scan for opportunities
        results = orchestrator.run_quick_scan(symbols)
        
        # Display scan results
        print(f"\n📊 QUICK SCAN RESULTS (from {len(symbols)} scanned):")
        print("-" * 60)
        
        candidates = []
        for symbol, data in results.items():
            if isinstance(data, dict) and 'error' not in data:
                fund_score = data.get('fundamental_score', 0)
                recommendation = data.get('recommendation', 'LOW_INTEREST')
                
                candidates.append({
                    'symbol': symbol,
                    'fundamental_score': fund_score,
                    'recommendation': recommendation,
                    'data': data
                })
        
        # Sort by fundamental score
        candidates.sort(key=lambda x: x['fundamental_score'], reverse=True)
        
        for i, candidate in enumerate(candidates[:10], 1):
            symbol = candidate['symbol']
            score = candidate['fundamental_score']
            rec = candidate['recommendation']
            
            print(f"{i:2d}. {symbol:6s} | Score: {score:5.1f} | Rec: {rec}")
        
        return {"candidates": candidates, "scan_results": results}
        
    except Exception as e:
        logger.error(f"Market scan failed: {e}", "PRODUCTION")
        print(f"❌ Market scan failed: {e}")
        return {"error": str(e)}

def run_continuous_monitoring(symbols: List[str] = None, interval_minutes: int = 30):
    """
    Continuous monitoring mode - runs pipeline at regular intervals
    """
    print("🔄 TRADER-X CONTINUOUS MONITORING")
    print("=" * 50)
    print(f"⏰ Interval: {interval_minutes} minutes")
    print(f"🎯 Symbols: {symbols or TradingConfig.TEST_STOCKS}")
    print("📊 Press Ctrl+C to stop")
    print("=" * 50)
    
    if not symbols:
        symbols = TradingConfig.TEST_STOCKS
    
    cycle = 1
    
    try:
        while True:
            print(f"\n🔄 MONITORING CYCLE {cycle}")
            print(f"🕐 {datetime.now()}")
            print("-" * 30)
            
            # Run pipeline
            results = run_full_pipeline(symbols, live_mode=False)
            
            # Check for high-confidence opportunities
            check_alerts(results)
            
            print(f"\n⏳ Waiting {interval_minutes} minutes until next cycle...")
            time.sleep(interval_minutes * 60)
            cycle += 1
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user after {cycle-1} cycles")
    except Exception as e:
        logger.error(f"Continuous monitoring failed: {e}", "PRODUCTION")
        print(f"❌ Monitoring failed: {e}")

def check_alerts(results: Dict[str, Any]):
    """Check for high-confidence trading alerts"""
    if 'ai_decisions' not in results:
        return
    
    alerts = []
    for symbol, decision_data in results['ai_decisions'].items():
        if isinstance(decision_data, dict):
            decision = decision_data.get('decision', 'HOLD')
            confidence = decision_data.get('confidence', 0.0)
            
            if decision in ['BUY', 'SELL'] and confidence >= TradingConfig.AI_CONFIDENCE_THRESHOLD:
                alerts.append({
                    'symbol': symbol,
                    'decision': decision,
                    'confidence': confidence,
                    'reasoning': decision_data.get('reasoning', '')[:100] + '...'
                })
    
    if alerts:
        print(f"\n🚨 HIGH-CONFIDENCE ALERTS ({len(alerts)}):")
        print("-" * 40)
        for alert in alerts:
            print(f"🎯 {alert['symbol']}: {alert['decision']} (confidence: {alert['confidence']:.2f})")
            print(f"   {alert['reasoning']}")

def display_production_results(results: Dict[str, Any], execution_time: float, live_mode: bool):
    """Display comprehensive production results"""
    
    print(f"\n{'=' * 60}")
    print("📊 TRADER-X PRODUCTION RESULTS")
    print(f"{'=' * 60}")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    print(f"🎯 Mode: {'LIVE TRADING' if live_mode else 'PAPER TRADING'}")
    print(f"📅 Completed: {datetime.now()}")
    
    # Phase 1 Results
    if 'phase1_results' in results:
        phase1 = results['phase1_results']
        passed_fundamental = sum(1 for r in phase1.values() if isinstance(r, dict) and r.get('fundamental', {}).get('passed', False))
        passed_sentiment = sum(1 for r in phase1.values() if isinstance(r, dict) and r.get('sentiment', {}).get('passed', False))
        
        print(f"\n📈 PHASE 1 - SIGNAL GENERATION:")
        print(f"   Stocks Analyzed: {len(phase1)}")
        print(f"   Fundamental Passed: {passed_fundamental}")
        print(f"   Sentiment Passed: {passed_sentiment}")
    
    # Phase 2 Results
    if 'phase2_results' in results:
        phase2 = results['phase2_results']
        print(f"\n🔍 PHASE 2 - DEEP ANALYSIS:")
        print(f"   Candidates Analyzed: {len(phase2)}")
    
    # AI Decisions
    if 'ai_decisions' in results:
        decisions = results['ai_decisions']
        buy_signals = sum(1 for d in decisions.values() if isinstance(d, dict) and d.get('decision') == 'BUY')
        sell_signals = sum(1 for d in decisions.values() if isinstance(d, dict) and d.get('decision') == 'SELL')
        hold_signals = sum(1 for d in decisions.values() if isinstance(d, dict) and d.get('decision') == 'HOLD')
        
        print(f"\n🧠 AI DECISIONS:")
        print(f"   BUY Signals: {buy_signals}")
        print(f"   SELL Signals: {sell_signals}")
        print(f"   HOLD Signals: {hold_signals}")
        
        # Show top recommendations
        recommendations = []
        for symbol, decision_data in decisions.items():
            if isinstance(decision_data, dict):
                decision = decision_data.get('decision', 'HOLD')
                confidence = decision_data.get('confidence', 0.0)
                if decision in ['BUY', 'SELL']:
                    recommendations.append((symbol, decision, confidence))
        
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        if recommendations:
            print(f"\n🎯 TOP RECOMMENDATIONS:")
            print("-" * 40)
            for i, (symbol, decision, confidence) in enumerate(recommendations[:5], 1):
                print(f"{i}. {symbol}: {decision} (confidence: {confidence:.2f})")
    
    # Market Context
    if 'market_context' in results:
        market = results['market_context']
        print(f"\n🌍 MARKET CONTEXT:")
        print(f"   Market Score: {market.get('market_score', 'N/A')}")
        print(f"   VIX Level: {market.get('vix', 'N/A')}")
        print(f"   SPY Trend: {market.get('spy_trend', 'N/A')}")
    
    print(f"\n{'=' * 60}")

def main():
    """Main production runner"""
    parser = argparse.ArgumentParser(description='Trader-X Production Runner')
    parser.add_argument('--mode', choices=['full', 'scan', 'monitor'], default='full',
                       help='Execution mode')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze')
    parser.add_argument('--live', action='store_true', help='Enable live trading mode')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Monitoring interval in minutes (for monitor mode)')
    
    args = parser.parse_args()
    
    # Validate live mode
    if args.live:
        print("⚠️  LIVE TRADING MODE ENABLED")
        print("   This will execute real trades with real money!")
        confirm = input("   Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("❌ Live trading cancelled")
            return
    
    try:
        if args.mode == 'full':
            results = run_full_pipeline(args.symbols, args.live)
            return 0 if 'error' not in results else 1
            
        elif args.mode == 'scan':
            results = run_market_scan(args.symbols)
            return 0 if 'error' not in results else 1
            
        elif args.mode == 'monitor':
            run_continuous_monitoring(args.symbols, args.interval)
            return 0
            
    except KeyboardInterrupt:
        print("\n🛑 Execution stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Production runner failed: {e}")
        logger.error(f"Production runner failed: {e}", "PRODUCTION")
        return 1

if __name__ == "__main__":
    sys.exit(main())
