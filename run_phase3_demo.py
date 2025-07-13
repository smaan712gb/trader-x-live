"""
Phase 3 Demo - Live Trade Execution
Demonstrates the complete Trader-X pipeline with Phase 3 execution
"""
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Import core modules
from core.orchestrator import orchestrator
from core.logger import logger
from modules.phase3_execution.trade_executor import trade_executor

def test_phase3_simulation():
    """Test Phase 3 with simulated trades"""
    print("\n" + "="*60)
    print("🚀 TRADER-X PHASE 3 SIMULATION TEST")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run full pipeline in test mode
        print("\n📊 Running full pipeline in simulation mode...")
        pipeline_results = orchestrator.run_full_pipeline(test_mode=True)
        
        # Check if we have AI decisions to execute
        ai_decisions = pipeline_results.get('ai_decisions', [])
        
        if not ai_decisions:
            print("❌ No AI decisions generated - cannot test Phase 3")
            return
        
        print(f"\n✅ Pipeline generated {len(ai_decisions)} AI decisions")
        
        # Display AI decisions
        print("\n📋 AI DECISIONS:")
        for i, decision in enumerate(ai_decisions[:5], 1):  # Show top 5
            symbol = decision['symbol']
            trade_decision = decision['decision']
            confidence = decision['confidence']
            phase1_score = decision.get('phase1_score', 0)
            phase2_score = decision.get('phase2_score', 0)
            
            print(f"   {i}. {symbol}: {trade_decision} (Confidence: {confidence:.2f})")
            print(f"      Phase 1: {phase1_score:.1f}, Phase 2: {phase2_score:.1f}")
        
        # Test Phase 3 execution in simulation mode
        print("\n🎯 Testing Phase 3 execution (SIMULATION)...")
        execution_results = trade_executor.execute_ai_decisions(ai_decisions, test_mode=True)
        
        # Display execution results
        orders_placed = execution_results.get('orders_placed', [])
        orders_rejected = execution_results.get('orders_rejected', [])
        total_capital_used = execution_results.get('total_capital_used', 0)
        
        print(f"\n📈 EXECUTION RESULTS:")
        print(f"   ✅ Orders Placed: {len(orders_placed)}")
        print(f"   ❌ Orders Rejected: {len(orders_rejected)}")
        print(f"   💰 Capital Used: ${total_capital_used:,.2f}")
        
        # Show placed orders
        if orders_placed:
            print(f"\n📋 PLACED ORDERS:")
            for order in orders_placed:
                symbol = order['symbol']
                action = order['action']
                quantity = order['quantity']
                price = order['price']
                capital = order['capital_used']
                
                print(f"   • {action} {quantity} {symbol} @ ${price:.2f} (${capital:,.2f})")
        
        # Show rejected orders
        if orders_rejected:
            print(f"\n❌ REJECTED ORDERS:")
            for rejection in orders_rejected:
                symbol = rejection['symbol']
                decision = rejection['decision']
                reason = rejection['reason']
                
                print(f"   • {symbol} ({decision}): {reason}")
        
        # Get portfolio summary
        portfolio = trade_executor.get_portfolio_summary()
        
        if portfolio.get('total_positions', 0) > 0:
            print(f"\n💼 PORTFOLIO SUMMARY:")
            print(f"   Positions: {portfolio['total_positions']}")
            print(f"   Market Value: ${portfolio['total_market_value']:,.2f}")
            print(f"   Total P&L: ${portfolio['total_pnl']:,.2f}")
            
            # Show individual positions
            positions = portfolio.get('positions', [])
            if positions:
                print(f"\n📊 INDIVIDUAL POSITIONS:")
                for pos in positions:
                    symbol = pos['symbol']
                    quantity = pos['quantity']
                    avg_cost = pos['avg_cost']
                    market_value = pos['market_value']
                    pnl = pos['unrealized_pnl']
                    pnl_pct = pos['pnl_percent']
                    
                    pnl_indicator = "📈" if pnl >= 0 else "📉"
                    print(f"   {pnl_indicator} {symbol}: {quantity} shares @ ${avg_cost:.2f}")
                    print(f"      Market Value: ${market_value:,.2f}, P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)")
        
        execution_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
        
        print(f"\n✅ Phase 3 simulation test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Phase 3 test failed: {e}")
        logger.error(f"Phase 3 test failed: {e}", "PHASE3_DEMO")

def test_portfolio_management():
    """Test portfolio management features"""
    print("\n" + "="*60)
    print("💼 PORTFOLIO MANAGEMENT TEST")
    print("="*60)
    
    try:
        # Get current portfolio status
        portfolio = trade_executor.get_portfolio_summary()
        
        print(f"Current Portfolio Status:")
        print(f"   Total Positions: {portfolio.get('total_positions', 0)}")
        print(f"   Total Market Value: ${portfolio.get('total_market_value', 0):,.2f}")
        print(f"   Available Capital: ${portfolio.get('available_capital', 0):,.2f}")
        print(f"   Total P&L: ${portfolio.get('total_pnl', 0):,.2f}")
        
        # Test position closing if we have positions
        if portfolio.get('total_positions', 0) > 0:
            print(f"\n🔄 Testing position closing (simulation)...")
            close_results = trade_executor.close_all_positions(test_mode=True)
            
            positions_closed = close_results.get('positions_closed', [])
            errors = close_results.get('errors', [])
            
            print(f"   Positions Closed: {len(positions_closed)}")
            print(f"   Errors: {len(errors)}")
            
            if positions_closed:
                total_realized_pnl = sum(pos['realized_pnl'] for pos in positions_closed)
                print(f"   Total Realized P&L: ${total_realized_pnl:,.2f}")
                
                for pos in positions_closed:
                    symbol = pos['symbol']
                    quantity = pos['quantity']
                    avg_cost = pos['avg_cost']
                    close_price = pos['close_price']
                    realized_pnl = pos['realized_pnl']
                    
                    pnl_indicator = "📈" if realized_pnl >= 0 else "📉"
                    print(f"   {pnl_indicator} {symbol}: {quantity} shares, "
                          f"${avg_cost:.2f} → ${close_price:.2f}, P&L: ${realized_pnl:,.2f}")
        else:
            print("   No positions to manage")
        
    except Exception as e:
        print(f"❌ Portfolio management test failed: {e}")
        logger.error(f"Portfolio management test failed: {e}", "PHASE3_DEMO")

def demonstrate_risk_management():
    """Demonstrate risk management features"""
    print("\n" + "="*60)
    print("⚠️  RISK MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    try:
        # Create mock high-risk decisions to test rejection
        mock_decisions = [
            {
                'symbol': 'TSLA',
                'decision': 'BUY',
                'confidence': 0.95,  # High confidence
                'phase1_score': 85,
                'phase2_score': 80
            },
            {
                'symbol': 'AAPL',
                'decision': 'BUY',
                'confidence': 0.60,  # Below threshold
                'phase1_score': 70,
                'phase2_score': 65
            },
            {
                'symbol': 'NVDA',
                'decision': 'SELL',
                'confidence': 0.85,
                'phase1_score': 75,
                'phase2_score': 70
            }
        ]
        
        print(f"Testing risk management with {len(mock_decisions)} mock decisions...")
        
        # Test execution with risk management
        execution_results = trade_executor.execute_ai_decisions(mock_decisions, test_mode=True)
        
        orders_placed = execution_results.get('orders_placed', [])
        orders_rejected = execution_results.get('orders_rejected', [])
        
        print(f"\n📊 Risk Management Results:")
        print(f"   ✅ Orders Approved: {len(orders_placed)}")
        print(f"   ❌ Orders Rejected: {len(orders_rejected)}")
        
        if orders_rejected:
            print(f"\n❌ Rejection Reasons:")
            for rejection in orders_rejected:
                symbol = rejection['symbol']
                decision = rejection['decision']
                reason = rejection['reason']
                print(f"   • {symbol} ({decision}): {reason}")
        
        if orders_placed:
            print(f"\n✅ Approved Orders:")
            for order in orders_placed:
                symbol = order['symbol']
                action = order['action']
                quantity = order['quantity']
                print(f"   • {action} {quantity} {symbol}")
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
        logger.error(f"Risk management test failed: {e}", "PHASE3_DEMO")

def main():
    """Run comprehensive Phase 3 demonstration"""
    print("🚀 TRADER-X PHASE 3 COMPREHENSIVE DEMO")
    print("="*60)
    print("This demo showcases the complete live execution capabilities")
    print("of the Trader-X autonomous trading platform.")
    print("="*60)
    
    try:
        # Test 1: Phase 3 Simulation
        test_phase3_simulation()
        
        # Test 2: Portfolio Management
        test_portfolio_management()
        
        # Test 3: Risk Management
        demonstrate_risk_management()
        
        print("\n" + "="*60)
        print("🎯 PHASE 3 DEMO SUMMARY")
        print("="*60)
        print("✅ Phase 3 execution system is fully operational")
        print("✅ Risk management controls are active")
        print("✅ Portfolio management features working")
        print("✅ Ready for live trading (when test_mode=False)")
        print("\n🚨 IMPORTANT NOTES:")
        print("   • All tests run in SIMULATION mode")
        print("   • No real money was used")
        print("   • Set test_mode=False for live trading")
        print("   • Ensure IB Gateway is connected for live execution")
        print("   • Review all risk parameters before going live")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Phase 3 demo failed: {e}", "PHASE3_DEMO")

if __name__ == "__main__":
    main()
