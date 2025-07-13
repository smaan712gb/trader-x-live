"""
Production Trading System Validation
Comprehensive test of the live trading system with real data validation
"""
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logger import logger
from data.market_data_live import live_market_data_manager
from modules.phase1_signal_generation.live_fundamental_screener import live_fundamental_screener
from config.trading_config import TradingConfig

def test_system_initialization():
    """Test system initialization and connections"""
    logger.info("=== SYSTEM INITIALIZATION TEST ===", "VALIDATION")
    
    try:
        # Test live data manager initialization
        logger.info("Testing live data manager initialization...", "VALIDATION")
        status = live_market_data_manager.get_connection_status()
        
        logger.info(f"IB Gateway Connected: {status['ib_gateway_connected']}", "VALIDATION")
        logger.info(f"Data Validation Enabled: {status['data_validation_enabled']}", "VALIDATION")
        logger.info(f"Live Trading Ready: {status['live_trading_ready']}", "VALIDATION")
        logger.info(f"Min Data Quality: {status['min_data_quality_threshold']}", "VALIDATION")
        
        if not status['live_trading_ready']:
            logger.error("SYSTEM NOT READY FOR LIVE TRADING", "VALIDATION")
            return False
        
        # Test live screener initialization
        logger.info("Testing live fundamental screener initialization...", "VALIDATION")
        screener_status = live_fundamental_screener.validate_screening_readiness()
        
        logger.info(f"Screening System Ready: {screener_status['ready_for_screening']}", "VALIDATION")
        
        logger.info("✓ System initialization successful", "VALIDATION")
        return True
        
    except Exception as e:
        logger.error(f"✗ System initialization failed: {e}", "VALIDATION")
        return False

def test_live_data_quality():
    """Test live data quality and validation"""
    logger.info("=== LIVE DATA QUALITY TEST ===", "VALIDATION")
    
    test_symbols = ["SPY", "NVDA", "AAPL"]  # Highly liquid stocks for testing
    
    for symbol in test_symbols:
        try:
            logger.info(f"Testing data quality for {symbol}...", "VALIDATION")
            
            # Test real-time quote
            quote = live_market_data_manager.get_real_time_quote(symbol)
            logger.info(f"  Real-time quote: ${quote['last']:.2f} (validated: {quote['validated']})", "VALIDATION")
            
            # Test historical data
            historical = live_market_data_manager.get_stock_data(symbol, period="5d", interval="1d")
            logger.info(f"  Historical data: {len(historical)} rows", "VALIDATION")
            
            # Test technical indicators
            technical = live_market_data_manager.get_technical_indicators(symbol)
            logger.info(f"  Technical indicators: RSI={technical['rsi']:.1f}, validated={technical['validated']}", "VALIDATION")
            
            logger.info(f"✓ {symbol} data quality validated", "VALIDATION")
            
        except Exception as e:
            logger.error(f"✗ {symbol} data quality test failed: {e}", "VALIDATION")
            return False
    
    logger.info("✓ Live data quality test passed", "VALIDATION")
    return True

def test_live_fundamental_screening():
    """Test live fundamental screening with real data"""
    logger.info("=== LIVE FUNDAMENTAL SCREENING TEST ===", "VALIDATION")
    
    try:
        # Test with a subset of high-quality stocks
        test_universe = ["NVDA", "AVGO", "MSFT", "AAPL", "GOOGL"]
        
        logger.info(f"Testing live screening with {len(test_universe)} stocks...", "VALIDATION")
        logger.info("USING REAL DATA ONLY - NO FALLBACK", "VALIDATION")
        
        start_time = time.time()
        qualified_stocks = live_fundamental_screener.screen_universe(test_universe)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"Screening completed in {duration:.2f} seconds", "VALIDATION")
        logger.info(f"Results: {len(qualified_stocks)} stocks qualified", "VALIDATION")
        
        for stock in qualified_stocks:
            logger.info(f"QUALIFIED: {stock['symbol']}", "VALIDATION")
            logger.info(f"  Revenue Growth YoY: {stock['revenue_growth_yoy']:.1f}%", "VALIDATION")
            logger.info(f"  Revenue Growth QoQ: {stock['revenue_growth_qoq']:.1f}%", "VALIDATION")
            logger.info(f"  Composite Score: {stock['composite_score']:.1f}", "VALIDATION")
            logger.info(f"  Data Validated: {stock['data_validated']}", "VALIDATION")
            logger.info(f"  Live Data Source: {stock['live_data_source']}", "VALIDATION")
            
            # Verify all data is validated
            if not stock['data_validated'] or not stock['live_data_source']:
                logger.error(f"✗ {stock['symbol']} using non-validated data", "VALIDATION")
                return False
        
        logger.info("✓ Live fundamental screening test passed", "VALIDATION")
        return True
        
    except Exception as e:
        logger.error(f"✗ Live fundamental screening test failed: {e}", "VALIDATION")
        return False

def test_top_candidates_selection():
    """Test selection of top trading candidates"""
    logger.info("=== TOP CANDIDATES SELECTION TEST ===", "VALIDATION")
    
    try:
        logger.info("Selecting top candidates for live trading...", "VALIDATION")
        
        top_candidates = live_fundamental_screener.get_top_live_candidates(max_candidates=3)
        
        logger.info(f"Selected {len(top_candidates)} top candidates", "VALIDATION")
        
        for i, candidate in enumerate(top_candidates, 1):
            logger.info(f"CANDIDATE #{i}: {candidate['symbol']}", "VALIDATION")
            logger.info(f"  Composite Score: {candidate['composite_score']:.1f}", "VALIDATION")
            logger.info(f"  Growth Score: {candidate['revenue_growth_score']:.1f}", "VALIDATION")
            logger.info(f"  Quality Score: {candidate['quality_score']:.1f}", "VALIDATION")
            logger.info(f"  Technical Score: {candidate['technical_score']:.1f}", "VALIDATION")
            logger.info(f"  Growth Acceleration: {candidate['screening_results']['growth_acceleration']}", "VALIDATION")
            
            # Verify strict criteria
            criteria = candidate['screening_results']['criteria_met']
            logger.info(f"  Criteria Met: {all(criteria.values())}", "VALIDATION")
            
            if not all(criteria.values()):
                logger.error(f"✗ {candidate['symbol']} does not meet all live trading criteria", "VALIDATION")
                return False
        
        logger.info("✓ Top candidates selection test passed", "VALIDATION")
        return True
        
    except Exception as e:
        logger.error(f"✗ Top candidates selection test failed: {e}", "VALIDATION")
        return False

def test_system_safety_mechanisms():
    """Test system safety mechanisms"""
    logger.info("=== SYSTEM SAFETY MECHANISMS TEST ===", "VALIDATION")
    
    try:
        # Test data validation requirements
        logger.info("Testing data validation requirements...", "VALIDATION")
        
        # Verify minimum data quality threshold
        status = live_market_data_manager.get_connection_status()
        if status['min_data_quality_threshold'] < 0.95:
            logger.error("✗ Data quality threshold too low for live trading", "VALIDATION")
            return False
        
        # Test connection dependency
        logger.info("Testing connection dependency...", "VALIDATION")
        if not live_market_data_manager.is_ready_for_live_trading():
            logger.error("✗ System not ready for live trading", "VALIDATION")
            return False
        
        # Test screening readiness
        logger.info("Testing screening readiness...", "VALIDATION")
        readiness = live_fundamental_screener.validate_screening_readiness()
        if not readiness['ready_for_screening']:
            logger.error("✗ Screening system not ready", "VALIDATION")
            return False
        
        logger.info("✓ System safety mechanisms test passed", "VALIDATION")
        return True
        
    except Exception as e:
        logger.error(f"✗ System safety mechanisms test failed: {e}", "VALIDATION")
        return False

def run_comprehensive_validation():
    """Run comprehensive system validation"""
    logger.info("STARTING COMPREHENSIVE PRODUCTION SYSTEM VALIDATION", "VALIDATION")
    logger.info(f"Validation started at: {datetime.now()}", "VALIDATION")
    logger.info("=" * 60, "VALIDATION")
    
    start_time = time.time()
    test_results = {}
    
    # Run all validation tests
    tests = [
        ("System Initialization", test_system_initialization),
        ("Live Data Quality", test_live_data_quality),
        ("Live Fundamental Screening", test_live_fundamental_screening),
        ("Top Candidates Selection", test_top_candidates_selection),
        ("System Safety Mechanisms", test_system_safety_mechanisms)
    ]
    
    all_passed = True
    
    for test_name, test_function in tests:
        logger.info(f"Running {test_name}...", "VALIDATION")
        try:
            result = test_function()
            test_results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}", "VALIDATION")
            test_results[test_name] = False
            all_passed = False
        
        logger.info("-" * 40, "VALIDATION")
    
    # Final summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    logger.info("=" * 60, "VALIDATION")
    logger.info("VALIDATION SUMMARY", "VALIDATION")
    logger.info("=" * 60, "VALIDATION")
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}", "VALIDATION")
    
    logger.info(f"Total Duration: {total_duration:.2f} seconds", "VALIDATION")
    logger.info(f"Tests Passed: {sum(test_results.values())}/{len(test_results)}", "VALIDATION")
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED - SYSTEM READY FOR LIVE TRADING", "VALIDATION")
        system_status = "PRODUCTION_READY"
    else:
        logger.error("❌ SOME TESTS FAILED - SYSTEM NOT READY FOR LIVE TRADING", "VALIDATION")
        system_status = "NOT_READY"
    
    logger.info(f"System Status: {system_status}", "VALIDATION")
    logger.info(f"Validation completed at: {datetime.now()}", "VALIDATION")
    
    return {
        'system_status': system_status,
        'all_tests_passed': all_passed,
        'test_results': test_results,
        'duration': total_duration,
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Main validation function"""
    try:
        results = run_comprehensive_validation()
        
        if results['all_tests_passed']:
            logger.info("Production system validation completed successfully", "VALIDATION")
            logger.info("System is ready for live trading with real money", "VALIDATION")
        else:
            logger.error("Production system validation failed", "VALIDATION")
            logger.error("DO NOT USE FOR LIVE TRADING", "VALIDATION")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation crashed: {e}", "VALIDATION")
        logger.error("SYSTEM NOT SAFE FOR LIVE TRADING", "VALIDATION")
        raise

if __name__ == "__main__":
    main()
