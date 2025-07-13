"""
Live Paper Trading Test - Simplified Version
Test the complete pipeline without strict IB Gateway requirements
"""
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Import core modules
from core.orchestrator import orchestrator
from core.logger import logger
from modules.phase3_execution.trade_executor import trade_executor

def run_live_paper_test():
    """Run live paper trading test"""
    print("\n" + "="*70)
    print("🚀 TRADER-X LIVE PAPER TRADING TEST")
    print("="*70)
    print("⚠️  LIVE EXECUTION MODE - PAPER ACCOUNT")
    print(f"Started at: {datetime.now()}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        print("✅ Starting live paper trading test...")
        
        # Run full pipeline with live execution
        print("\n📊 Running full pipeline with LIVE EXECUTION...")
        print("🔴 This will place REAL orders on your paper account")
        
        # Confirm execution
        confirm = input("\nProceed with live paper trading? (yes/no): ").lower().strip()
        if confirm != 'yes':
            print("❌ Live trading cancelled by user")
            return
        
        print("\n🎯 EXECUTING LIVE PAPER TRADES...")
        
        # Run pipeline with live execution (test_mode=False)
        pipeline_results = orchestrator.run_full_pipeline(test_mode=False)
        
        # Display results
        print("\n" + "="*50)
        print("📈 LIVE EXECUTION RESULTS")
        print("="*50)
        
        # Market context
        market_context = pipeline_results.get('market_context', {})
        if market_context:
            market_score = market_context.get('market_score', 'N/A')
            trading_rec = market_context.get('trading_recommendation', 'N/A')
            print(f"Market Score: {market_score}")
            print(f"Trading Recommendation: {trading_rec}")
        
        # Phase 1 results
        phase1_results = pipeline_results.get('phase1_results', {})
        qualified_candidates = phase1_results.get('qualified_candidates', [])
        print(f"\nPhase 1: {len(qualified_candidates)} qualified candidates")
        
        # Phase 2 results
        phase2_results = pipeline_results.get('phase2_results', {})
        analyzed_candidates = len(phase2_results.get('analysis_results', {}))
        print(f"Phase 2: {analyzed_candidates} candidates analyzed")
        
        # AI decisions
        ai_decisions = pipeline_results.get('ai_decisions', [])
        print(f"AI Decisions: {len(ai_decisions)} generated")
        
        if ai_decisions:
            print("\n📋 TOP AI DECISIONS:")
            for i, decision in enumerate(ai_decisions[:5], 1):
                symbol = decision['symbol']
                trade_decision = decision['decision']
                confidence = decision['confidence']
                print(f"   {i}. {symbol}: {trade_decision} (Confidence: {confidence:.2f})")
        
        # Phase 3 execution results
        phase3_results = pipeline_results.get('phase3_results', {})
        if phase3_results:
            execution_results = phase3_results.get('execution_results', {})
            orders_placed = execution_results.get('orders_placed', [])
            orders_rejected = execution_results.get('orders_rejected', [])
            total_capital_used = execution_results.get('total_capital_used', 0)
            
            print(f"\n🎯 PHASE 3 EXECUTION:")
            print(f"   ✅ Orders Placed: {len(orders_placed)}")
            print(f"   ❌ Orders Rejected: {len(orders_rejected)}")
            print(f"   💰 Capital Used: ${total_capital_used:,.2f}")
            
            # Show placed orders
            if orders_placed:
                print(f"\n📋 LIVE ORDERS PLACED:")
                for order in orders_placed:
                    symbol = order['symbol']
                    action = order['action']
                    quantity = order.get('quantity', 'N/A')
                    ib_order_id = order.get('ib_order_id', 'N/A')
                    
                    print(f"   🔴 {action} {quantity} {symbol} (IB Order ID: {ib_order_id})")
            
            # Show rejected orders
            if orders_rejected:
                print(f"\n❌ REJECTED ORDERS:")
                for rejection in orders_rejected:
                    symbol = rejection['symbol']
                    decision = rejection['decision']
                    reason = rejection['reason']
                    print(f"   • {symbol} ({decision}): {reason}")
            
            # Portfolio summary
            portfolio_summary = phase3_results.get('portfolio_summary', {})
            if portfolio_summary.get('total_positions', 0) > 0:
                print(f"\n💼 LIVE PORTFOLIO:")
                print(f"   Positions: {portfolio_summary['total_positions']}")
                print(f"   Market Value: ${portfolio_summary['total_market_value']:,.2f}")
                print(f"   Total P&L: ${portfolio_summary['total_pnl']:,.2f}")
        
        else:
            print("\n⚠️  No Phase 3 execution (likely due to market conditions)")
        
        # Execution time
        execution_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {execution_time:.2f} seconds")
        
        # Trading summary
        trades_executed = pipeline_results.get('trades_executed', 0)
        print(f"\n🎯 TRADING SESSION SUMMARY:")
        print(f"   • Live trades executed: {trades_executed}")
        print(f"   • Account type: PAPER TRADING")
        print(f"   • Execution mode: LIVE")
        print(f"   • Risk controls: ACTIVE")
        
        if trades_executed > 0:
            print(f"\n✅ Live paper trading session completed successfully!")
            print(f"   Check your IB paper account for order confirmations")
        else:
            print(f"\n⚠️  No trades executed this session")
            print(f"   This may be due to:")
            print(f"   • Market conditions")
            print(f"   • Risk management filters")
            print(f"   • No qualifying opportunities")
        
    except Exception as e:
        print(f"\n❌ Live paper trading failed: {e}")
        logger.error(f"Live paper trading failed: {e}", "LIVE_PAPER_TRADING")

def monitor_positions():
    """Monitor current positions"""
    print("\n" + "="*50)
    print("📊 POSITION MONITORING")
    print("="*50)
    
    try:
        # Get current portfolio
        portfolio = trade_executor.get_portfolio_summary()
        
        if portfolio.get('total_positions', 0) == 0:
            print("No positions to monitor")
            return
        
        print(f"Monitoring {portfolio['total_positions']} positions...")
        
        positions = portfolio.get('positions', [])
        for pos in positions:
            symbol = pos['symbol']
            quantity = pos['quantity']
            avg_cost = pos['avg_cost']
            market_value = pos['market_value']
            unrealized_pnl = pos['unrealized_pnl']
            pnl_pct = pos['pnl_percent']
            
            pnl_indicator = "📈" if unrealized_pnl >= 0 else "📉"
            print(f"{pnl_indicator} {symbol}: {quantity} shares @ ${avg_cost:.2f}")
            print(f"   Market Value: ${market_value:,.2f}")
            print(f"   Unrealized P&L: ${unrealized_pnl:,.2f} ({pnl_pct:+.1f}%)")
            print()
        
        total_pnl = portfolio.get('total_pnl', 0)
        total_value = portfolio.get('total_market_value', 0)
        
        print(f"📊 PORTFOLIO TOTALS:")
        print(f"   Total Market Value: ${total_value:,.2f}")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        
    except Exception as e:
        print(f"❌ Position monitoring failed: {e}")
        logger.error(f"Position monitoring failed: {e}", "LIVE_PAPER_TRADING")

def main():
    """Main interface"""
    print("🚀 TRADER-X LIVE PAPER TRADING TEST")
    print("="*70)
    print("Simplified version for testing live execution")
    print("="*70)
    
    while True:
        print("\n📋 AVAILABLE ACTIONS:")
        print("1. Run Live Paper Trading Test")
        print("2. Monitor Positions")
        print("3. Exit")
        
        choice = input("\nSelect action (1-3): ").strip()
        
        if choice == '1':
            run_live_paper_test()
        elif choice == '2':
            monitor_positions()
        elif choice == '3':
            print("👋 Exiting test system")
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")

if __name__ == "__main__":
    main()
