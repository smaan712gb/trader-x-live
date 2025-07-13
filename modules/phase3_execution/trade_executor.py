"""
Phase 3: Trade Execution Module
Handles live trade execution through Interactive Brokers
"""
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from config.trading_config import TradingConfig
from core.logger import logger
from data.ib_gateway import ib_gateway

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PARTIAL = "PARTIAL"

@dataclass
class TradeOrder:
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[int] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class Position:
    symbol: str
    quantity: int
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradeExecutor:
    def __init__(self):
        self.active_orders: Dict[int, TradeOrder] = {}
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeOrder] = []
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.order_counter = 1000
        
    def execute_ai_decisions(self, ai_decisions: List[Dict[str, Any]], 
                           test_mode: bool = True) -> Dict[str, Any]:
        """
        Execute trades based on AI decisions
        """
        logger.info(f"Executing {len(ai_decisions)} AI decisions (test_mode: {test_mode})", "TRADE_EXECUTOR")
        
        execution_results = {
            'orders_placed': [],
            'orders_rejected': [],
            'positions_opened': [],
            'positions_closed': [],
            'total_capital_used': 0.0,
            'execution_time': time.time()
        }
        
        try:
            for decision in ai_decisions:
                symbol = decision['symbol']
                trade_decision = decision['decision']
                confidence = decision['confidence']
                
                # Check if we should execute this decision
                if not self._should_execute_decision(decision):
                    rejection_reason = self._get_rejection_reason(decision)
                    execution_results['orders_rejected'].append({
                        'symbol': symbol,
                        'decision': trade_decision,
                        'reason': rejection_reason
                    })
                    logger.info(f"Decision rejected for {symbol}: {rejection_reason}", "TRADE_EXECUTOR")
                    continue
                
                # Calculate position size
                position_size = self._calculate_position_size(symbol, confidence, decision)
                
                if position_size == 0:
                    execution_results['orders_rejected'].append({
                        'symbol': symbol,
                        'decision': trade_decision,
                        'reason': 'Position size calculated as 0'
                    })
                    continue
                
                # Execute the trade
                if trade_decision in ['BUY', 'SELL']:
                    order_result = self._execute_trade(
                        symbol, trade_decision, position_size, decision, test_mode
                    )
                    
                    if order_result['success']:
                        execution_results['orders_placed'].append(order_result)
                        execution_results['total_capital_used'] += order_result.get('capital_used', 0)
                        logger.info(f"Order placed for {symbol}: {trade_decision} {position_size} shares", "TRADE_EXECUTOR")
                    else:
                        execution_results['orders_rejected'].append({
                            'symbol': symbol,
                            'decision': trade_decision,
                            'reason': order_result.get('error', 'Unknown error')
                        })
                        logger.error(f"Order failed for {symbol}: {order_result.get('error')}", "TRADE_EXECUTOR")
                
                elif trade_decision == 'HOLD':
                    # Check if we need to adjust existing positions
                    adjustment_result = self._handle_hold_decision(symbol, decision)
                    if adjustment_result:
                        execution_results['positions_opened'].append(adjustment_result)
            
            # Update execution summary
            execution_results['execution_time'] = time.time() - execution_results['execution_time']
            
            logger.info(f"Execution complete: {len(execution_results['orders_placed'])} orders placed, "
                       f"{len(execution_results['orders_rejected'])} rejected", "TRADE_EXECUTOR")
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}", "TRADE_EXECUTOR")
            execution_results['error'] = str(e)
            return execution_results
    
    def _should_execute_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Determine if a trading decision should be executed
        """
        symbol = decision['symbol']
        trade_decision = decision['decision']
        confidence = decision['confidence']
        
        # Check confidence threshold
        if confidence < TradingConfig.AI_CONFIDENCE_THRESHOLD:
            return False
        
        # Check daily trade limits
        if len([o for o in self.trade_history if o.timestamp and 
                o.timestamp.date() == datetime.now().date()]) >= TradingConfig.MAX_DAILY_TRADES:
            return False
        
        # Check if we already have a position in this symbol
        if symbol in self.positions and trade_decision == 'BUY':
            # Don't add to existing long positions unless specifically allowed
            if not TradingConfig.ALLOW_POSITION_SCALING:
                return False
        
        # Check market hours (if required)
        if TradingConfig.MARKET_HOURS_ONLY and not self._is_market_hours():
            return False
        
        # Check available capital
        required_capital = self._estimate_required_capital(symbol, decision)
        if required_capital > self._get_available_capital():
            return False
        
        return True
    
    def _calculate_position_size(self, symbol: str, confidence: float, 
                                decision: Dict[str, Any]) -> int:
        """
        Calculate optimal position size based on risk management rules
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {symbol}", "TRADE_EXECUTOR")
                return 0
            
            # Get available capital
            available_capital = self._get_available_capital()
            
            # Base position size calculation
            base_allocation = TradingConfig.MAX_POSITION_SIZE_PCT / 100.0
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence / TradingConfig.AI_CONFIDENCE_THRESHOLD, 1.5)
            
            # Adjust based on Phase 2 score if available
            phase2_score = decision.get('phase2_score', 50)
            score_multiplier = max(0.5, phase2_score / 100.0)
            
            # Calculate position value
            position_value = available_capital * base_allocation * confidence_multiplier * score_multiplier
            
            # Convert to shares
            shares = int(position_value / current_price)
            
            # Apply minimum and maximum limits
            min_shares = max(1, int(TradingConfig.MIN_ORDER_VALUE / current_price))
            max_shares = int(available_capital * TradingConfig.MAX_POSITION_SIZE_PCT / 100.0 / current_price)
            
            shares = max(min_shares, min(shares, max_shares))
            
            logger.info(f"Position size for {symbol}: {shares} shares "
                       f"(price: ${current_price:.2f}, confidence: {confidence:.2f})", "TRADE_EXECUTOR")
            
            return shares
            
        except Exception as e:
            logger.error(f"Position size calculation failed for {symbol}: {e}", "TRADE_EXECUTOR")
            return 0
    
    def _execute_trade(self, symbol: str, action: str, quantity: int, 
                      decision: Dict[str, Any], test_mode: bool = True) -> Dict[str, Any]:
        """
        Execute a single trade
        """
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                return {'success': False, 'error': 'Could not get current price'}
            
            # Create order
            order = TradeOrder(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type=OrderType.MARKET,  # Start with market orders for simplicity
                order_id=self._get_next_order_id(),
                timestamp=datetime.now()
            )
            
            if test_mode:
                # Simulate order execution
                order.status = OrderStatus.FILLED
                order.filled_price = current_price
                order.filled_quantity = quantity
                order.commission = self._calculate_commission(quantity, current_price)
                
                # Update positions
                self._update_position_simulation(order)
                
                result = {
                    'success': True,
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_price,
                    'capital_used': quantity * current_price,
                    'commission': order.commission,
                    'test_mode': True
                }
                
                logger.info(f"SIMULATED: {action} {quantity} {symbol} @ ${current_price:.2f}", "TRADE_EXECUTOR")
                
            else:
                # Execute real order through IB Gateway
                ib_order_result = self._place_ib_order(order)
                
                if ib_order_result['success']:
                    order.status = OrderStatus.SUBMITTED
                    self.active_orders[order.order_id] = order
                    
                    result = {
                        'success': True,
                        'order_id': order.order_id,
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'estimated_price': current_price,
                        'capital_used': quantity * current_price,
                        'ib_order_id': ib_order_result.get('ib_order_id'),
                        'test_mode': False
                    }
                    
                    logger.info(f"LIVE ORDER: {action} {quantity} {symbol} @ ~${current_price:.2f}", "TRADE_EXECUTOR")
                    
                else:
                    result = {
                        'success': False,
                        'error': ib_order_result.get('error', 'IB order placement failed')
                    }
            
            # Add to trade history
            self.trade_history.append(order)
            
            return result
            
        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}", "TRADE_EXECUTOR")
            return {'success': False, 'error': str(e)}
    
    def _place_ib_order(self, order: TradeOrder) -> Dict[str, Any]:
        """
        Place order through Interactive Brokers
        """
        try:
            if not ib_gateway.is_connected():
                return {'success': False, 'error': 'IB Gateway not connected'}
            
            # Create IB order
            ib_order = {
                'symbol': order.symbol,
                'action': order.action,
                'quantity': order.quantity,
                'orderType': order.order_type.value,
                'timeInForce': order.time_in_force
            }
            
            if order.limit_price:
                ib_order['lmtPrice'] = order.limit_price
            if order.stop_price:
                ib_order['auxPrice'] = order.stop_price
            
            # Submit order to IB
            ib_result = ib_gateway.place_order(ib_order)
            
            if ib_result.get('success'):
                return {
                    'success': True,
                    'ib_order_id': ib_result.get('order_id'),
                    'message': 'Order submitted to IB'
                }
            else:
                return {
                    'success': False,
                    'error': ib_result.get('error', 'IB order submission failed')
                }
                
        except Exception as e:
            logger.error(f"IB order placement failed: {e}", "TRADE_EXECUTOR")
            return {'success': False, 'error': str(e)}
    
    def _update_position_simulation(self, order: TradeOrder):
        """
        Update positions for simulated trades
        """
        symbol = order.symbol
        
        if symbol not in self.positions:
            # New position
            if order.action == 'BUY':
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.filled_quantity,
                    avg_cost=order.filled_price,
                    market_value=order.filled_quantity * order.filled_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_time=order.timestamp
                )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if order.action == 'BUY':
                # Add to position
                total_cost = (position.quantity * position.avg_cost) + (order.filled_quantity * order.filled_price)
                total_quantity = position.quantity + order.filled_quantity
                position.avg_cost = total_cost / total_quantity
                position.quantity = total_quantity
                
            elif order.action == 'SELL':
                # Reduce or close position
                if order.filled_quantity >= position.quantity:
                    # Close position
                    realized_pnl = (order.filled_price - position.avg_cost) * position.quantity
                    position.realized_pnl += realized_pnl
                    self.total_pnl += realized_pnl
                    del self.positions[symbol]
                else:
                    # Partial close
                    realized_pnl = (order.filled_price - position.avg_cost) * order.filled_quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= order.filled_quantity
                    self.total_pnl += realized_pnl
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol
        """
        try:
            if ib_gateway.is_connected():
                price_data = ib_gateway.get_market_data(symbol)
                if price_data and 'last_price' in price_data:
                    return float(price_data['last_price'])
            
            # Fallback to market data manager
            from data.market_data_enhanced import market_data_manager
            market_data = market_data_manager.get_current_price(symbol)
            if market_data:
                return float(market_data)
            
            logger.warning(f"Could not get current price for {symbol}", "TRADE_EXECUTOR")
            return None
            
        except Exception as e:
            logger.error(f"Price lookup failed for {symbol}: {e}", "TRADE_EXECUTOR")
            return None
    
    def _get_available_capital(self) -> float:
        """
        Get available trading capital
        """
        try:
            if ib_gateway.is_connected():
                account_info = ib_gateway.get_account_info()
                if account_info and 'available_funds' in account_info:
                    return float(account_info['available_funds'])
            
            # Fallback to configured capital
            return TradingConfig.INITIAL_CAPITAL
            
        except Exception as e:
            logger.error(f"Could not get available capital: {e}", "TRADE_EXECUTOR")
            return TradingConfig.INITIAL_CAPITAL
    
    def _estimate_required_capital(self, symbol: str, decision: Dict[str, Any]) -> float:
        """
        Estimate capital required for a trade
        """
        try:
            current_price = self._get_current_price(symbol)
            if not current_price:
                return float('inf')  # Can't trade without price
            
            # Estimate position size
            confidence = decision.get('confidence', 0.5)
            estimated_shares = self._calculate_position_size(symbol, confidence, decision)
            
            return estimated_shares * current_price * 1.1  # Add 10% buffer
            
        except Exception as e:
            logger.error(f"Capital estimation failed for {symbol}: {e}", "TRADE_EXECUTOR")
            return float('inf')
    
    def _calculate_commission(self, quantity: int, price: float) -> float:
        """
        Calculate estimated commission for a trade
        """
        # IB commission structure (simplified)
        commission_per_share = 0.005  # $0.005 per share
        min_commission = 1.00  # $1.00 minimum
        
        commission = max(min_commission, quantity * commission_per_share)
        return commission
    
    def _get_next_order_id(self) -> int:
        """
        Get next order ID
        """
        self.order_counter += 1
        return self.order_counter
    
    def _is_market_hours(self) -> bool:
        """
        Check if market is currently open
        """
        now = datetime.now()
        # Simplified market hours check (9:30 AM - 4:00 PM ET, weekdays)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _get_rejection_reason(self, decision: Dict[str, Any]) -> str:
        """
        Get reason for trade rejection
        """
        symbol = decision['symbol']
        confidence = decision['confidence']
        
        if confidence < TradingConfig.AI_CONFIDENCE_THRESHOLD:
            return f"Confidence {confidence:.2f} below threshold {TradingConfig.AI_CONFIDENCE_THRESHOLD}"
        
        if len([o for o in self.trade_history if o.timestamp and 
                o.timestamp.date() == datetime.now().date()]) >= TradingConfig.MAX_DAILY_TRADES:
            return f"Daily trade limit reached ({TradingConfig.MAX_DAILY_TRADES})"
        
        if symbol in self.positions and decision['decision'] == 'BUY':
            return "Already have position in this symbol"
        
        if TradingConfig.MARKET_HOURS_ONLY and not self._is_market_hours():
            return "Outside market hours"
        
        required_capital = self._estimate_required_capital(symbol, decision)
        available_capital = self._get_available_capital()
        if required_capital > available_capital:
            return f"Insufficient capital (need ${required_capital:.2f}, have ${available_capital:.2f})"
        
        return "Unknown rejection reason"
    
    def _handle_hold_decision(self, symbol: str, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle HOLD decisions - may involve position adjustments
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        current_price = self._get_current_price(symbol)
        
        if not current_price:
            return None
        
        # Update unrealized P&L
        position.market_value = position.quantity * current_price
        position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
        
        # Check if we need to set/update stop loss or take profit
        phase2_data = decision.get('phase2_data', {})
        technical_analysis = phase2_data.get('technical_analysis', {})
        
        # Set stop loss based on technical support levels
        support_level = technical_analysis.get('support_resistance', {}).get('support_level')
        if support_level and not position.stop_loss:
            stop_loss_price = support_level * 0.98  # 2% below support
            position.stop_loss = stop_loss_price
            
            logger.info(f"Set stop loss for {symbol} at ${stop_loss_price:.2f}", "TRADE_EXECUTOR")
            
            return {
                'symbol': symbol,
                'action': 'STOP_LOSS_SET',
                'price': stop_loss_price,
                'current_price': current_price
            }
        
        return None
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary
        """
        total_value = 0.0
        total_unrealized_pnl = 0.0
        
        for position in self.positions.values():
            current_price = self._get_current_price(position.symbol)
            if current_price:
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                total_value += position.market_value
                total_unrealized_pnl += position.unrealized_pnl
        
        return {
            'total_positions': len(self.positions),
            'total_market_value': total_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_realized_pnl': self.total_pnl,
            'total_pnl': total_unrealized_pnl + self.total_pnl,
            'available_capital': self._get_available_capital(),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'avg_cost': pos.avg_cost,
                    'market_value': pos.market_value,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'pnl_percent': (pos.unrealized_pnl / (pos.avg_cost * pos.quantity)) * 100
                }
                for pos in self.positions.values()
            ]
        }
    
    def close_all_positions(self, test_mode: bool = True) -> Dict[str, Any]:
        """
        Close all open positions
        """
        logger.info(f"Closing all positions (test_mode: {test_mode})", "TRADE_EXECUTOR")
        
        results = {
            'positions_closed': [],
            'errors': []
        }
        
        for symbol, position in list(self.positions.items()):
            try:
                close_order = TradeOrder(
                    symbol=symbol,
                    action='SELL',
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    order_id=self._get_next_order_id(),
                    timestamp=datetime.now()
                )
                
                if test_mode:
                    current_price = self._get_current_price(symbol)
                    if current_price:
                        close_order.status = OrderStatus.FILLED
                        close_order.filled_price = current_price
                        close_order.filled_quantity = position.quantity
                        
                        # Calculate realized P&L
                        realized_pnl = (current_price - position.avg_cost) * position.quantity
                        
                        results['positions_closed'].append({
                            'symbol': symbol,
                            'quantity': position.quantity,
                            'avg_cost': position.avg_cost,
                            'close_price': current_price,
                            'realized_pnl': realized_pnl
                        })
                        
                        # Remove position
                        del self.positions[symbol]
                        self.total_pnl += realized_pnl
                        
                        logger.info(f"SIMULATED CLOSE: {symbol} - P&L: ${realized_pnl:.2f}", "TRADE_EXECUTOR")
                
                else:
                    # Execute real close order
                    ib_result = self._place_ib_order(close_order)
                    if ib_result['success']:
                        results['positions_closed'].append({
                            'symbol': symbol,
                            'quantity': position.quantity,
                            'order_id': close_order.order_id,
                            'ib_order_id': ib_result.get('ib_order_id')
                        })
                    else:
                        results['errors'].append({
                            'symbol': symbol,
                            'error': ib_result.get('error')
                        })
                
            except Exception as e:
                logger.error(f"Failed to close position for {symbol}: {e}", "TRADE_EXECUTOR")
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return results

# Global trade executor instance
trade_executor = TradeExecutor()
