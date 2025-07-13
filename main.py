"""
Trader-X Main Entry Point
AI-Powered Autonomous Trading System
"""
import sys
import argparse
from datetime import datetime
import json
from typing import Dict, Any

from config.api_keys import APIKeys
from config.trading_config import TradingConfig
from config.supabase_config import supabase_manager
from core.logger import logger
from core.orchestrator import orchestrator
from data.market_data import market_data_manager

def setup_system():
    """Initialize and setup the Trader-X system"""
    logger.info("Initializing Trader-X System", "MAIN")
    
    try:
        # Validate API keys
        logger.info("Validating API keys...", "MAIN")
        APIKeys.validate_keys()
        logger.info("API keys validated successfully", "MAIN")
        
        # Initialize database
        logger.info("Setting up database...", "MAIN")
        supabase_manager.create_tables()
        supabase_manager.create_indexes()
        logger.info("Database setup completed", "MAIN")
        
        logger.info("Trader-X system initialized successfully", "MAIN")
        return True
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}", "MAIN")
        return False

def run_full_pipeline(test_mode: bool = True):
    """Run the complete Trader-X trading pipeline"""
    logger.info("=" * 60, "MAIN")
    logger.info("TRADER-X AUTONOMOUS TRADING SYSTEM", "MAIN")
    logger.info("=" * 60, "MAIN")
    
    try:
        # Run the full pipeline
        results = orchestrator.run_full_pipeline(test_mode=test_mode)
        
        # Display results
        display_pipeline_results(results)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", "MAIN")
        return None

def run_quick_scan(symbols: list = None):
    """Run a quick scan of specified symbols"""
    logger.info("Running quick market scan...", "MAIN")
    
    try:
        results = orchestrator.run_quick_scan(symbols)
        
        print("\n" + "=" * 50)
        print("QUICK MARKET SCAN RESULTS")
        print("=" * 50)
        
        for symbol, data in results.items():
            if 'error' in data:
                print(f"\n{symbol}: ERROR - {data['error']}")
            else:
                print(f"\n{symbol}:")
                print(f"  Fundamental Score: {data['fundamental_score']}")
                print(f"  Technical Summary: {data['technical_summary']}")
                print(f"  Recommendation: {data['recommendation']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Quick scan failed: {e}", "MAIN")
        return None

def display_pipeline_results(results: Dict[str, Any]):
    """Display formatted pipeline results"""
    print("\n" + "=" * 60)
    print("TRADER-X PIPELINE EXECUTION RESULTS")
    print("=" * 60)
    
    # Execution summary
    print(f"\nExecution Time: {results.get('total_execution_time', 0):.2f} seconds")
    print(f"Test Mode: {results.get('test_mode', True)}")
    print(f"Trades Executed: {results.get('trades_executed', 0)}")
    
    # Phase 1 Results
    phase1 = results.get('phase1_results', {})
    if phase1:
        print(f"\n--- PHASE 1: SIGNAL GENERATION ---")
        fundamental = phase1.get('fundamental_screening', {})
        print(f"Stocks Screened: {fundamental.get('total_screened', 0)}")
        print(f"Passed Fundamental: {fundamental.get('passed_screening', 0)}")
        print(f"Qualified for Phase 2: {len(phase1.get('qualified_candidates', []))}")
        
        # Show top candidates
        candidates = phase1.get('qualified_candidates', [])
        if candidates:
            print(f"\nTop Candidates:")
            for i, candidate in enumerate(candidates[:3], 1):
                symbol = candidate['symbol']
                score = candidate.get('phase1_score', 0)
                hype = candidate.get('hype_score', 0)
                print(f"  {i}. {symbol} - Score: {score:.1f}, Hype: {hype:.1f}%")
    
    # Phase 2 Results
    phase2 = results.get('phase2_results', {})
    if phase2:
        print(f"\n--- PHASE 2: DEEP ANALYSIS ---")
        print(f"Candidates Analyzed: {phase2.get('candidates_analyzed', 0)}")
        print(f"Execution Time: {phase2.get('execution_time', 0):.2f} seconds")
    
    # AI Decisions
    ai_decisions = results.get('ai_decisions', [])
    if ai_decisions:
        print(f"\n--- AI DECISIONS ---")
        for decision in ai_decisions:
            symbol = decision['symbol']
            action = decision['decision']
            confidence = decision['confidence']
            print(f"  {symbol}: {action} (confidence: {confidence:.2f})")
    
    # Phase 3 Results
    phase3 = results.get('phase3_results', {})
    if phase3:
        print(f"\n--- PHASE 3: EXECUTION ---")
        executed = phase3.get('executed_trades', [])
        rejected = phase3.get('rejected_trades', [])
        
        if executed:
            print(f"Executed Trades ({len(executed)}):")
            for trade in executed:
                print(f"  {trade['symbol']}: {trade['action']} - {trade['status']}")
        
        if rejected:
            print(f"Rejected Trades ({len(rejected)}):")
            for trade in rejected[:3]:  # Show first 3
                print(f"  {trade['symbol']}: {trade['rejection_reason']}")
    
    # Errors
    errors = results.get('errors', [])
    if errors:
        print(f"\n--- ERRORS ---")
        for error in errors:
            print(f"  {error}")
    
    print("\n" + "=" * 60)

def test_individual_modules():
    """Test individual modules for debugging"""
    print("\n" + "=" * 50)
    print("TESTING INDIVIDUAL MODULES")
    print("=" * 50)
    
    # Test market data
    print("\n1. Testing Market Data...")
    try:
        data = market_data_manager.get_stock_data("TSLA", period="5d")
        print(f"   ✓ TSLA data: {len(data)} rows")
    except Exception as e:
        print(f"   ✗ Market data failed: {e}")
    
    # Test fundamental analysis
    print("\n2. Testing Fundamental Analysis...")
    try:
        fundamental_data = market_data_manager.get_fundamental_data("TSLA")
        revenue_growth = fundamental_data.get('revenue_growth_yoy', 0)
        print(f"   ✓ TSLA revenue growth: {revenue_growth:.1f}%")
    except Exception as e:
        print(f"   ✗ Fundamental analysis failed: {e}")
    
    # Test technical analysis
    print("\n3. Testing Technical Analysis...")
    try:
        technical_data = market_data_manager.get_technical_indicators("TSLA")
        rsi = technical_data.get('rsi', 0)
        print(f"   ✓ TSLA RSI: {rsi:.1f}")
    except Exception as e:
        print(f"   ✗ Technical analysis failed: {e}")
    
    print("\n" + "=" * 50)

def show_system_status():
    """Display current system status"""
    status = orchestrator.get_system_status()
    
    print("\n" + "=" * 40)
    print("TRADER-X SYSTEM STATUS")
    print("=" * 40)
    print(f"Status: {status['status']}")
    print(f"Current Candidates: {status['current_candidates']}")
    print(f"Active Positions: {status['active_positions']}")
    print(f"Daily Trade Count: {status['daily_trade_count']}")
    print(f"Last Run: {status['last_run_date'] or 'Never'}")
    print(f"Timestamp: {status['timestamp']}")
    print("=" * 40)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Trader-X Autonomous Trading System")
    parser.add_argument('--mode', choices=['full', 'scan', 'test', 'status'], 
                       default='full', help='Execution mode')
    parser.add_argument('--symbols', nargs='+', help='Symbols for scan mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode (default: test mode)')
    parser.add_argument('--setup', action='store_true', help='Setup system and exit')
    
    args = parser.parse_args()
    
    # Setup system
    if not setup_system():
        logger.error("System setup failed. Exiting.", "MAIN")
        sys.exit(1)
    
    if args.setup:
        logger.info("System setup completed successfully", "MAIN")
        return
    
    # Execute based on mode
    try:
        if args.mode == 'full':
            test_mode = not args.live
            results = run_full_pipeline(test_mode=test_mode)
            
        elif args.mode == 'scan':
            results = run_quick_scan(args.symbols)
            
        elif args.mode == 'test':
            test_individual_modules()
            results = None
            
        elif args.mode == 'status':
            show_system_status()
            results = None
        
        # Save results if available
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trader_x_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filename}", "MAIN")
    
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user", "MAIN")
    except Exception as e:
        logger.error(f"Execution failed: {e}", "MAIN")
        sys.exit(1)

if __name__ == "__main__":
    main()
