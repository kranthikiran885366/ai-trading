#!/usr/bin/env python3
"""
Advanced Execution Engine with Smart Order Routing and Real-time Execution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    OCO = "OCO"  # One Cancels Other

@dataclass
class ExecutionResult:
    """Execution result with comprehensive details"""
    success: bool
    order_id: str
    symbol: str
    action: str
    quantity: float
    executed_price: float
    executed_quantity: float
    commission: float
    slippage: float
    execution_time: datetime
    broker: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class ExecutionEngine:
    """Advanced execution engine with smart order routing"""
    
    def __init__(self, broker_manager, risk_manager):
        self.broker_manager = broker_manager
        self.risk_manager = risk_manager
        
        # Execution tracking
        self.active_orders = {}
        self.execution_history = []
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'avg_slippage': 0.0,
            'avg_execution_time': 0.0
        }
        
        # Smart routing parameters
        self.routing_preferences = {
            'prefer_low_latency': True,
            'prefer_low_cost': True,
            'prefer_high_liquidity': True,
            'max_slippage_tolerance': 0.001  # 0.1%
        }
        
        # Order management
        self.order_timeout = 300  # 5 minutes
        self.retry_attempts = 3
        self.partial_fill_threshold = 0.9  # 90% fill considered complete
        
        logger.info("Execution Engine initialized")
    
    async def execute_signal(self, signal: Any) -> ExecutionResult:
        """Execute a trading signal with smart order routing"""
        try:
            # Pre-execution validation
            if not await self._pre_execution_validation(signal):
                return ExecutionResult(
                    success=False,
                    order_id="",
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=0,
                    executed_price=0,
                    executed_quantity=0,
                    commission=0,
                    slippage=0,
                    execution_time=datetime.now(),
                    broker="",
                    error_message="Pre-execution validation failed"
                )
            
            # Select optimal broker
            broker = await self._select_optimal_broker(signal)
            if not broker:
                return ExecutionResult(
                    success=False,
                    order_id="",
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=0,
                    executed_price=0,
                    executed_quantity=0,
                    commission=0,
                    slippage=0,
                    execution_time=datetime.now(),
                    broker="",
                    error_message="No suitable broker available"
                )
            
            # Calculate optimal order size
            order_quantity = await self._calculate_order_quantity(signal)
            
            # Determine order type and parameters
            order_type, order_params = await self._determine_order_strategy(signal, broker)
            
            # Execute the order
            execution_result = await self._execute_order(
                broker=broker,
                symbol=signal.symbol,
                action=signal.action,
                quantity=order_quantity,
                order_type=order_type,
                order_params=order_params,
                signal=signal
            )
            
            # Post-execution processing
            await self._post_execution_processing(execution_result, signal)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=signal.symbol,
                action=signal.action,
                quantity=0,
                executed_price=0,
                executed_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                broker="",
                error_message=str(e)
            )
    
    async def _pre_execution_validation(self, signal: Any) -> bool:
        """Validate signal before execution"""
        try:
            # Check if signal is still valid
            signal_age = (datetime.now() - signal.timestamp).total_seconds()
            if signal_age > 300:  # 5 minutes
                logger.warning(f"Signal too old: {signal_age} seconds")
                return False
            
            # Check market hours
            if not await self._is_market_open(signal.symbol):
                logger.warning(f"Market closed for {signal.symbol}")
                return False
            
            # Risk validation
            if not await self.risk_manager.validate_signal(signal):
                logger.warning(f"Signal failed risk validation")
                return False
            
            # Check for duplicate orders
            if await self._has_duplicate_order(signal):
                logger.warning(f"Duplicate order detected for {signal.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in pre-execution validation: {e}")
            return False
    
    async def _select_optimal_broker(self, signal: Any) -> Optional[str]:
        """Select the optimal broker for execution"""
        try:
            available_brokers = await self.broker_manager.get_available_brokers(signal.symbol)
            
            if not available_brokers:
                return None
            
            # Score brokers based on various factors
            broker_scores = {}
            
            for broker_name in available_brokers:
                score = 0.0
                
                # Latency score
                latency = await self.broker_manager.get_broker_latency(broker_name)
                latency_score = max(0, 1 - latency / 100)  # Normalize to 0-1
                score += latency_score * 0.3
                
                # Cost score
                commission = await self.broker_manager.get_commission_rate(broker_name, signal.symbol)
                cost_score = max(0, 1 - commission / 0.01)  # Normalize to 0-1
                score += cost_score * 0.3
                
                # Liquidity score
                liquidity = await self.broker_manager.get_liquidity_score(broker_name, signal.symbol)
                score += liquidity * 0.2
                
                # Reliability score
                reliability = await self.broker_manager.get_reliability_score(broker_name)
                score += reliability * 0.2
                
                broker_scores[broker_name] = score
            
            # Select broker with highest score
            best_broker = max(broker_scores, key=broker_scores.get)
            logger.info(f"Selected broker {best_broker} with score {broker_scores[best_broker]:.2f}")
            
            return best_broker
            
        except Exception as e:
            logger.error(f"Error selecting optimal broker: {e}")
            return list(available_brokers)[0] if available_brokers else None
    
    async def _calculate_order_quantity(self, signal: Any) -> float:
        """Calculate optimal order quantity"""
        try:
            # Base quantity from signal
            base_quantity = signal.position_size
            
            # Adjust for current portfolio state
            current_position = await self._get_current_position(signal.symbol)
            
            # If we already have a position, adjust the order size
            if current_position:
                if signal.action == 'BUY' and current_position > 0:
                    # Adding to long position
                    base_quantity *= 0.8  # Reduce size
                elif signal.action == 'SELL' and current_position < 0:
                    # Adding to short position
                    base_quantity *= 0.8  # Reduce size
                elif (signal.action == 'SELL' and current_position > 0) or \
                     (signal.action == 'BUY' and current_position < 0):
                    # Closing or reversing position
                    base_quantity = min(base_quantity, abs(current_position))
            
            # Ensure minimum order size
            min_order_size = await self._get_min_order_size(signal.symbol)
            base_quantity = max(base_quantity, min_order_size)
            
            # Ensure maximum order size
            max_order_size = await self._get_max_order_size(signal.symbol)
            base_quantity = min(base_quantity, max_order_size)
            
            return base_quantity
            
        except Exception as e:
            logger.error(f"Error calculating order quantity: {e}")
            return signal.position_size
    
    async def _determine_order_strategy(self, signal: Any, broker: str) -> Tuple[OrderType, Dict[str, Any]]:
        """Determine optimal order type and parameters"""
        try:
            # Get current market conditions
            market_data = await self.broker_manager.get_market_data(broker, signal.symbol)
            
            if not market_data:
                return OrderType.MARKET, {}
            
            current_price = market_data.get('price', signal.price)
            bid_ask_spread = market_data.get('spread', 0.001)
            volatility = market_data.get('volatility', 0.02)
            
            # Determine order type based on market conditions and signal characteristics
            order_params = {}
            
            # High confidence signals use market orders for immediate execution
            if signal.confidence > 0.9:
                return OrderType.MARKET, order_params
            
            # For volatile markets, use limit orders to control slippage
            if volatility > 0.05 or bid_ask_spread > 0.002:
                # Set limit price with small buffer
                if signal.action == 'BUY':
                    limit_price = current_price * (1 + bid_ask_spread / 2)
                else:
                    limit_price = current_price * (1 - bid_ask_spread / 2)
                
                order_params['limit_price'] = limit_price
                order_params['time_in_force'] = 'IOC'  # Immediate or Cancel
                
                return OrderType.LIMIT, order_params
            
            # For normal conditions, use market orders
            return OrderType.MARKET, order_params
            
        except Exception as e:
            logger.error(f"Error determining order strategy: {e}")
            return OrderType.MARKET, {}
    
    async def _execute_order(self, broker: str, symbol: str, action: str, 
                           quantity: float, order_type: OrderType, 
                           order_params: Dict[str, Any], signal: Any) -> ExecutionResult:
        """Execute the actual order"""
        try:
            order_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            # Create order request
            order_request = {
                'order_id': order_id,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'order_type': order_type.value,
                'params': order_params,
                'signal_id': getattr(signal, 'id', ''),
                'timestamp': start_time
            }
            
            # Store active order
            self.active_orders[order_id] = {
                'request': order_request,
                'status': OrderStatus.PENDING,
                'start_time': start_time
            }
            
            # Execute through broker
            broker_result = await self.broker_manager.execute_order(broker, order_request)
            
            if not broker_result.get('success', False):
                # Order failed
                self.active_orders[order_id]['status'] = OrderStatus.REJECTED
                
                return ExecutionResult(
                    success=False,
                    order_id=order_id,
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    executed_price=0,
                    executed_quantity=0,
                    commission=0,
                    slippage=0,
                    execution_time=datetime.now(),
                    broker=broker,
                    error_message=broker_result.get('error', 'Unknown error')
                )
            
            # Order submitted successfully
            self.active_orders[order_id]['status'] = OrderStatus.SUBMITTED
            self.active_orders[order_id]['broker_order_id'] = broker_result.get('broker_order_id')
            
            # Wait for execution
            execution_result = await self._wait_for_execution(order_id, broker, signal.price)
            
            # Update statistics
            self._update_execution_stats(execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return ExecutionResult(
                success=False,
                order_id=order_id if 'order_id' in locals() else "",
                symbol=symbol,
                action=action,
                quantity=quantity,
                executed_price=0,
                executed_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                broker=broker,
                error_message=str(e)
            )
    
    async def _wait_for_execution(self, order_id: str, broker: str, expected_price: float) -> ExecutionResult:
        """Wait for order execution with timeout"""
        try:
            timeout_time = datetime.now() + timedelta(seconds=self.order_timeout)
            
            while datetime.now() < timeout_time:
                # Check order status
                order_status = await self.broker_manager.get_order_status(broker, order_id)
                
                if order_status.get('status') == 'FILLED':
                    # Order fully executed
                    executed_price = order_status.get('executed_price', expected_price)
                    executed_quantity = order_status.get('executed_quantity', 0)
                    commission = order_status.get('commission', 0)
                    
                    # Calculate slippage
                    slippage = abs(executed_price - expected_price) / expected_price
                    
                    self.active_orders[order_id]['status'] = OrderStatus.FILLED
                    
                    return ExecutionResult(
                        success=True,
                        order_id=order_id,
                        symbol=order_status.get('symbol', ''),
                        action=order_status.get('action', ''),
                        quantity=order_status.get('quantity', 0),
                        executed_price=executed_price,
                        executed_quantity=executed_quantity,
                        commission=commission,
                        slippage=slippage,
                        execution_time=datetime.now(),
                        broker=broker
                    )
                
                elif order_status.get('status') == 'PARTIALLY_FILLED':
                    # Check if partial fill is acceptable
                    executed_quantity = order_status.get('executed_quantity', 0)
                    total_quantity = order_status.get('quantity', 0)
                    
                    if total_quantity > 0 and executed_quantity / total_quantity >= self.partial_fill_threshold:
                        # Accept partial fill
                        executed_price = order_status.get('executed_price', expected_price)
                        commission = order_status.get('commission', 0)
                        slippage = abs(executed_price - expected_price) / expected_price
                        
                        self.active_orders[order_id]['status'] = OrderStatus.PARTIALLY_FILLED
                        
                        return ExecutionResult(
                            success=True,
                            order_id=order_id,
                            symbol=order_status.get('symbol', ''),
                            action=order_status.get('action', ''),
                            quantity=total_quantity,
                            executed_price=executed_price,
                            executed_quantity=executed_quantity,
                            commission=commission,
                            slippage=slippage,
                            execution_time=datetime.now(),
                            broker=broker
                        )
                
                elif order_status.get('status') in ['CANCELLED', 'REJECTED']:
                    # Order failed
                    self.active_orders[order_id]['status'] = OrderStatus.REJECTED
                    
                    return ExecutionResult(
                        success=False,
                        order_id=order_id,
                        symbol=order_status.get('symbol', ''),
                        action=order_status.get('action', ''),
                        quantity=order_status.get('quantity', 0),
                        executed_price=0,
                        executed_quantity=0,
                        commission=0,
                        slippage=0,
                        execution_time=datetime.now(),
                        broker=broker,
                        error_message=order_status.get('error', 'Order cancelled or rejected')
                    )
                
                # Wait before checking again
                await asyncio.sleep(1)
            
            # Timeout reached
            self.active_orders[order_id]['status'] = OrderStatus.EXPIRED
            
            # Try to cancel the order
            await self.broker_manager.cancel_order(broker, order_id)
            
            return ExecutionResult(
                success=False,
                order_id=order_id,
                symbol='',
                action='',
                quantity=0,
                executed_price=0,
                executed_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                broker=broker,
                error_message="Order execution timeout"
            )
            
        except Exception as e:
            logger.error(f"Error waiting for execution: {e}")
            return ExecutionResult(
                success=False,
                order_id=order_id,
                symbol='',
                action='',
                quantity=0,
                executed_price=0,
                executed_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                broker=broker,
                error_message=str(e)
            )
    
    async def _post_execution_processing(self, execution_result: ExecutionResult, signal: Any):
        """Post-execution processing and logging"""
        try:
            # Store execution result
            self.execution_history.append(execution_result)
            
            # Keep only recent history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]
            
            # Log execution
            if execution_result.success:
                logger.info(f"Order executed successfully: {execution_result.symbol} {execution_result.action} "
                          f"{execution_result.executed_quantity} @ {execution_result.executed_price:.4f}")
            else:
                logger.error(f"Order execution failed: {execution_result.error_message}")
            
            # Update position tracking
            await self._update_position_tracking(execution_result)
            
            # Send execution notification
            await self._send_execution_notification(execution_result, signal)
            
        except Exception as e:
            logger.error(f"Error in post-execution processing: {e}")
    
    async def _update_position_tracking(self, execution_result: ExecutionResult):
        """Update position tracking after execution"""
        try:
            if not execution_result.success:
                return
            
            # This would update the portfolio manager with the new position
            # Implementation depends on the portfolio manager interface
            pass
            
        except Exception as e:
            logger.error(f"Error updating position tracking: {e}")
    
    async def _send_execution_notification(self, execution_result: ExecutionResult, signal: Any):
        """Send execution notification"""
        try:
            notification = {
                'type': 'execution',
                'success': execution_result.success,
                'symbol': execution_result.symbol,
                'action': execution_result.action,
                'quantity': execution_result.executed_quantity,
                'price': execution_result.executed_price,
                'slippage': execution_result.slippage,
                'commission': execution_result.commission,
                'timestamp': execution_result.execution_time.isoformat(),
                'signal_confidence': signal.confidence,
                'error': execution_result.error_message
            }
            
            # This would send to notification system
            # Implementation depends on notification manager
            
        except Exception as e:
            logger.error(f"Error sending execution notification: {e}")
    
    def _update_execution_stats(self, execution_result: ExecutionResult):
        """Update execution statistics"""
        try:
            self.execution_stats['total_orders'] += 1
            
            if execution_result.success:
                self.execution_stats['successful_orders'] += 1
                
                # Update average slippage
                old_avg_slippage = self.execution_stats['avg_slippage']
                successful_orders = self.execution_stats['successful_orders']
                new_avg_slippage = (old_avg_slippage * (successful_orders - 1) + execution_result.slippage) / successful_orders
                self.execution_stats['avg_slippage'] = new_avg_slippage
                
            else:
                self.execution_stats['failed_orders'] += 1
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    async def execute_manual_trade(self, trade_data: Dict[str, Any]) -> ExecutionResult:
        """Execute a manual trade"""
        try:
            # Validate manual trade data
            required_fields = ['symbol', 'action', 'quantity']
            for field in required_fields:
                if field not in trade_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Create a mock signal for manual trade
            class ManualSignal:
                def __init__(self, data):
                    self.symbol = data['symbol']
                    self.action = data['action']
                    self.position_size = data['quantity']
                    self.price = data.get('price', 0)
                    self.confidence = 1.0  # Manual trades have full confidence
                    self.timestamp = datetime.now()
                    self.id = str(uuid.uuid4())
            
            manual_signal = ManualSignal(trade_data)
            
            # Execute the manual trade
            return await self.execute_signal(manual_signal)
            
        except Exception as e:
            logger.error(f"Error executing manual trade: {e}")
            return ExecutionResult(
                success=False,
                order_id="",
                symbol=trade_data.get('symbol', ''),
                action=trade_data.get('action', ''),
                quantity=trade_data.get('quantity', 0),
                executed_price=0,
                executed_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                broker="",
                error_message=str(e)
            )
    
    async def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.info("Closing all open positions...")
            
            # Get current positions from all brokers
            all_positions = await self.broker_manager.get_all_positions()
            
            close_orders = []
            
            for broker, positions in all_positions.items():
                for position in positions:
                    if position.get('quantity', 0) != 0:
                        # Create closing order
                        close_action = 'SELL' if position['quantity'] > 0 else 'BUY'
                        close_quantity = abs(position['quantity'])
                        
                        close_order = {
                            'symbol': position['symbol'],
                            'action': close_action,
                            'quantity': close_quantity,
                            'price': position.get('current_price', 0)
                        }
                        
                        # Execute closing order
                        result = await self.execute_manual_trade(close_order)
                        close_orders.append(result)
            
            successful_closes = sum(1 for order in close_orders if order.success)
            logger.info(f"Closed {successful_closes}/{len(close_orders)} positions")
            
            return close_orders
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return []
    
    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            logger.info("Cancelling all pending orders...")
            
            cancelled_count = 0
            
            for order_id, order_info in self.active_orders.items():
                if order_info['status'] in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    try:
                        broker = order_info.get('broker', '')
                        await self.broker_manager.cancel_order(broker, order_id)
                        order_info['status'] = OrderStatus.CANCELLED
                        cancelled_count += 1
                    except Exception as e:
                        logger.error(f"Error cancelling order {order_id}: {e}")
            
            logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def _is_market_open(self, symbol: str) -> bool:
        """Check if market is open for trading"""
        try:
            now = datetime.now()
            
            # Crypto markets are always open
            if symbol.endswith('USDT'):
                return True
            
            # Stock market hours (simplified)
            if now.weekday() >= 5:  # Weekend
                return False
            
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return True  # Default to open
    
    async def _has_duplicate_order(self, signal: Any) -> bool:
        """Check for duplicate orders"""
        try:
            # Check recent orders for same symbol and action
            recent_cutoff = datetime.now() - timedelta(minutes=5)
            
            for order_info in self.active_orders.values():
                order_request = order_info.get('request', {})
                
                if (order_request.get('symbol') == signal.symbol and
                    order_request.get('action') == signal.action and
                    order_request.get('timestamp', datetime.min) > recent_cutoff):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate orders: {e}")
            return False
    
    async def _get_current_position(self, symbol: str) -> float:
        """Get current position for symbol"""
        try:
            # This would query the portfolio manager
            # For now, return 0 (no position)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting current position: {e}")
            return 0.0
    
    async def _get_min_order_size(self, symbol: str) -> float:
        """Get minimum order size for symbol"""
        try:
            # Default minimum order sizes
            if symbol.endswith('USDT'):
                return 0.001  # Crypto
            else:
                return 1.0  # Stocks (1 share)
                
        except Exception as e:
            logger.error(f"Error getting min order size: {e}")
            return 1.0
    
    async def _get_max_order_size(self, symbol: str) -> float:
        """Get maximum order size for symbol"""
        try:
            # Default maximum order sizes (as percentage of portfolio)
            return 0.1  # 10% of portfolio
            
        except Exception as e:
            logger.error(f"Error getting max order size: {e}")
            return 0.1
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            total_orders = self.execution_stats['total_orders']
            
            stats = {
                'total_orders': total_orders,
                'successful_orders': self.execution_stats['successful_orders'],
                'failed_orders': self.execution_stats['failed_orders'],
                'success_rate': self.execution_stats['successful_orders'] / total_orders if total_orders > 0 else 0,
                'avg_slippage': self.execution_stats['avg_slippage'],
                'avg_execution_time': self.execution_stats['avg_execution_time'],
                'active_orders_count': len([o for o in self.active_orders.values() 
                                          if o['status'] in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]),
                'recent_executions': len(self.execution_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {e}")
            return {}
    
    def get_recent_executions(self, limit: int = 50) -> List[ExecutionResult]:
        """Get recent execution results"""
        try:
            return self.execution_history[-limit:] if limit else self.execution_history
            
        except Exception as e:
            logger.error(f"Error getting recent executions: {e}")
            return []
