#!/usr/bin/env python3
"""
Live Paper Trading System - Continuous Operation
Runs the complete Trader-X system continuously on paper account with real-time data
"""

import sys
import os
import time
import asyncio
from datetime import datetime, timedelta
import signal
import threading
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logger import logger
from core.orchestrator import TradingOrchestrator
from config.trading_config import TradingConfig
from config.api_keys import APIKeys
from data.ib_gateway import IBGatewayConnector
from modules.phase3_execution.trade_executor import TradeExecutor

class LivePaperTradingSystem:
    def __init__(self):
        self.orchestrator = TradingOrchestrator()
        self.trade_executor = TradeExecutor()
        self.ib_gateway = IBGatewayConnector()
        self.running = False
        self.last_scan_time = None
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.positions = {}
        
        # Trading schedule
        self.scan_interval = 300  # 5 minutes between scans
        self.max_daily_trades = TradingConfig.MAX_DAILY_TRADES
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Shutdown signal received, stopping live trading...", "LIVE_TRADING")
        self.stop()
    
    def start(self):
        """Start the live paper trading system"""
        logger.info("🚀 Starting Live Paper Trading System", "LIVE_TRADING")
        
        # Validate system before starting
        if not self._validate_system():
            logger.error("System validation failed, cannot start live trading", "LIVE_TRADING")
            return False
        
        # Connect to IB Gateway
        if not self._connect_to_ib():
            logger.error("Failed to connect to IB Gateway", "LIVE_TRADING")
            return False
        
        self.running = True
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        logger.info("✅ Live Paper Trading System Started Successfully", "LIVE_TRADING")
        logger.info(f"📊 Scan Interval: {self.scan_interval} seconds", "LIVE_TRADING")
        logger.info(f"📈 Max Daily Trades: {self.max_daily_trades}", "LIVE_TRADING")
        logger.info(f"🎯 Test Stocks: {TradingConfig.TEST_STOCKS}", "LIVE_TRADING")
        
        # Start the main trading loop
        self._run_trading_loop()
        
        return True
    
    def stop(self):
        """Stop the live trading system"""
        self.running = False
        logger.info("🛑 Live Paper Trading System Stopped", "LIVE_TRADING")
        
        # Disconnect from IB Gateway
        if self.ib_gateway:
            self.ib_gateway.disconnect()
    
    def _validate_system(self) -> bool:
        """Validate system components before starting"""
        logger.info("Validating system components...", "LIVE_TRADING")
        
        try:
            # Check API keys
            APIKeys.validate_keys()
            logger.info("✅ API keys validated", "LIVE_TRADING")
            
            # Check trading configuration
            if not TradingConfig.TEST_STOCKS:
                logger.error("No test stocks configured", "LIVE_TRADING")
                return False
            
            logger.info("✅ Trading configuration validated", "LIVE_TRADING")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {e}", "LIVE_TRADING")
            return False
    
    def _connect_to_ib(self) -> bool:
        """Connect to Interactive Brokers Gateway"""
        try:
            logger.info("Connecting to IB Gateway for live paper trading...", "LIVE_TRADING")
            
            # Use paper trading port (4002) for safety
            connection_result = self.ib_gateway.connect()
            
            if connection_result:
                # Get account info
                account_info = self.ib_gateway.get_account_info()
                logger.info(f"✅ Connected to IB Paper Account: {account_info.get('account_id', 'Unknown')}", "LIVE_TRADING")
                logger.info(f"💰 Available Buying Power: ${account_info.get('buying_power', 0):,.2f}", "LIVE_TRADING")
                return True
            else:
                logger.error("Failed to connect to IB Gateway", "LIVE_TRADING")
                return False
                
        except Exception as e:
            logger.error(f"IB connection failed: {e}", "LIVE_TRADING")
            return False
    
    def _run_trading_loop(self):
        """Main trading loop"""
        logger.info("🔄 Starting main trading loop", "LIVE_TRADING")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if it's time for a new scan
                if self._should_run_scan(current_time):
                    self._run_trading_cycle()
                    self.last_scan_time = current_time
                
                # Check existing positions
                self._monitor_positions()
                
                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, stopping...", "LIVE_TRADING")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", "LIVE_TRADING")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _should_run_scan(self, current_time: datetime) -> bool:
        """Check if it's time to run a new scan"""
        if self.last_scan_time is None:
            return True
        
        time_since_last_scan = (current_time - self.last_scan_time).total_seconds()
        return time_since_last_scan >= self.scan_interval
    
    def _run_trading_cycle(self):
        """Run a complete trading cycle"""
        logger.info("🔍 Starting new trading cycle", "LIVE_TRADING")
        
        try:
            # Check daily trade limit
            if self.daily_trades >= self.max_daily_trades:
                logger.info(f"Daily trade limit reached ({self.daily_trades}/{self.max_daily_trades})", "LIVE_TRADING")
                return
            
            # Run Phase 1: Signal Generation
            logger.info("📊 Running Phase 1: Signal Generation", "LIVE_TRADING")
            phase1_result = self.orchestrator.run_phase1_screening(TradingConfig.TEST_STOCKS)
            
            if not phase1_result or not phase1_result.get('qualified_stocks'):
                logger.info("No stocks qualified from Phase 1 screening", "LIVE_TRADING")
                return
            
            qualified_stocks = phase1_result['qualified_stocks']
            logger.info(f"✅ Phase 1 Complete: {len(qualified_stocks)} stocks qualified", "LIVE_TRADING")
            
            # Run Phase 2: Deep Analysis
            logger.info("🔬 Running Phase 2: Deep Analysis", "LIVE_TRADING")
            phase2_result = self.orchestrator.run_phase2_analysis(qualified_stocks)
            
            if not phase2_result or not phase2_result.get('trade_candidates'):
                logger.info("No trade candidates from Phase 2 analysis", "LIVE_TRADING")
                return
            
            trade_candidates = phase2_result['trade_candidates']
            logger.info(f"✅ Phase 2 Complete: {len(trade_candidates)} trade candidates", "LIVE_TRADING")
            
            # Run Phase 3: Trade Execution
            logger.info("⚡ Running Phase 3: Trade Execution", "LIVE_TRADING")
            for candidate in trade_candidates:
                if self.daily_trades >= self.max_daily_trades:
                    break
                
                success = self._execute_trade(candidate)
                if success:
                    self.daily_trades += 1
                    logger.info(f"✅ Trade executed for {candidate.get('symbol')} ({self.daily_trades}/{self.max_daily_trades})", "LIVE_TRADING")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", "LIVE_TRADING")
    
    def _execute_trade(self, candidate: Dict[str, Any]) -> bool:
        """Execute a trade for a candidate"""
        try:
            symbol = candidate.get('symbol')
            action = candidate.get('action', 'BUY')
            quantity = candidate.get('quantity', 100)
            
            logger.info(f"🎯 Executing {action} order for {symbol} (qty: {quantity})", "LIVE_TRADING")
            
            # Execute the trade through IB Gateway
            order_result = self.trade_executor.execute_trade(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type='MKT'  # Market order
            )
            
            if order_result and order_result.get('success'):
                # Store position info
                self.positions[symbol] = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'entry_time': datetime.now(),
                    'entry_price': order_result.get('fill_price', 0),
                    'order_id': order_result.get('order_id')
                }
                
                logger.info(f"✅ Trade executed successfully for {symbol}", "LIVE_TRADING")
                return True
            else:
                logger.error(f"❌ Trade execution failed for {symbol}", "LIVE_TRADING")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade for {candidate.get('symbol', 'unknown')}: {e}", "LIVE_TRADING")
            return False
    
    def _monitor_positions(self):
        """Monitor existing positions for stop loss/take profit"""
        if not self.positions:
            return
        
        try:
            for symbol, position in list(self.positions.items()):
                # Get current price
                current_data = self.ib_gateway.get_market_data(symbol)
                if not current_data:
                    continue
                
                current_price = current_data.get('last_price', 0)
                entry_price = position.get('entry_price', 0)
                
                if current_price == 0 or entry_price == 0:
                    continue
                
                # Calculate P&L percentage
                if position['action'] == 'BUY':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # SELL
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check stop loss
                if pnl_pct <= -TradingConfig.STOP_LOSS_PERCENTAGE:
                    logger.info(f"🛑 Stop loss triggered for {symbol} (P&L: {pnl_pct:.2%})", "LIVE_TRADING")
                    self._close_position(symbol, "STOP_LOSS")
                
                # Check take profit
                elif pnl_pct >= TradingConfig.TAKE_PROFIT_PERCENTAGE:
                    logger.info(f"🎯 Take profit triggered for {symbol} (P&L: {pnl_pct:.2%})", "LIVE_TRADING")
                    self._close_position(symbol, "TAKE_PROFIT")
                
        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", "LIVE_TRADING")
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Determine close action (opposite of entry)
            close_action = 'SELL' if position['action'] == 'BUY' else 'BUY'
            
            # Execute close order
            close_result = self.trade_executor.execute_trade(
                symbol=symbol,
                action=close_action,
                quantity=position['quantity'],
                order_type='MKT'
            )
            
            if close_result and close_result.get('success'):
                logger.info(f"✅ Position closed for {symbol} - Reason: {reason}", "LIVE_TRADING")
                del self.positions[symbol]
            else:
                logger.error(f"❌ Failed to close position for {symbol}", "LIVE_TRADING")
                
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}", "LIVE_TRADING")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'open_positions': len(self.positions),
            'positions': list(self.positions.keys()),
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'next_scan_in': self.scan_interval - (datetime.now() - self.last_scan_time).total_seconds() if self.last_scan_time else 0
        }

def main():
    """Main function to start live paper trading"""
    print("=" * 80)
    print("🚀 TRADER-X LIVE PAPER TRADING SYSTEM")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now()}")
    print(f"🎯 Trading Mode: PAPER TRADING (Safe)")
    print(f"📊 Scan Interval: 5 minutes")
    print(f"📈 Max Daily Trades: {TradingConfig.MAX_DAILY_TRADES}")
    print(f"🔍 Stocks to Monitor: {', '.join(TradingConfig.TEST_STOCKS)}")
    print("=" * 80)
    print()
    
    # Create and start the live trading system
    live_system = LivePaperTradingSystem()
    
    try:
        success = live_system.start()
        if not success:
            print("❌ Failed to start live trading system")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1
    finally:
        live_system.stop()
        print("👋 Live Paper Trading System Shutdown Complete")
    
    return 0

if __name__ == "__main__":
    exit(main())
