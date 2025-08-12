#!/usr/bin/env python3
"""
Trade Executor - Handles order execution across multiple brokers
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: datetime = None
    broker: str = None
    time_in_force: str = "DAY"
    commission: float = 0.0
    created_at: datetime = None
    updated_at: datetime = None
    strategy: str = ""
    parent_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    broker: str
    market_value: float = 0.0
    last_price: float = 0.0
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Trade:
    """Trade execution data structure"""
    id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    broker: str

class BrokerInterface:
    """Base broker interface"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def connect(self):
        """Connect to broker"""
        pass
    
    async def disconnect(self):
        """Disconnect from broker"""
        pass
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place order with broker"""
        pass
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        pass
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        pass
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

class SimulatedBroker(BrokerInterface):
    """Simulated broker for testing and backtesting"""
    
    def __init__(self, initial_balance: float = 100000, commission_rate: float = 0.001):
        super().__init__("SimulatedBroker")
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.commission_rate = commission_rate
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.market_data = {}
        self.slippage_rate = 0.001  # 0.1% slippage
        
    async def connect(self):
        """Connect to simulated broker"""
        self.is_connected = True
        self.logger.info("Connected to simulated broker")
    
    async def disconnect(self):
        """Disconnect from simulated broker"""
        self.is_connected = False
        self.logger.info("Disconnected from simulated broker")
    
    def set_market_data(self, symbol: str, price: float):
        """Set current market price for symbol"""
        self.market_data[symbol] = price
    
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place order with simulated broker"""
        try:
            if not self.is_connected:
                return {"success": False, "error": "Not connected to broker"}
            
            # Store order
            self.orders[order.order_id] = order
            order.status = OrderStatus.SUBMITTED
            order.broker = self.name
            
            # Simulate order execution
            if order.order_type == OrderType.MARKET:
                await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                await self._handle_limit_order(order)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "status": order.status.value
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_market_order(self, order: Order):
        """Execute market order immediately"""
        if order.symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_price = self.market_data[order.symbol]
        
        # Apply slippage
        if order.action == 'BUY':
            execution_price = market_price * (1 + self.slippage_rate)
        else:
            execution_price = market_price * (1 - self.slippage_rate)
        
        # Check if we have enough balance/shares
        if not self._can_execute_order(order, execution_price):
            order.status = OrderStatus.REJECTED
            return
        
        # Execute the order
        commission = order.quantity * execution_price * self.commission_rate
        
        # Create trade
        trade = Trade(
            id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=OrderSide(order.action),
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            timestamp=datetime.now(),
            broker=self.name
        )
        
        self.trades.append(trade)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = execution_price
        order.commission = commission
        order.updated_at = datetime.now()
        
        # Update positions and balance
        self._update_position(order.symbol, OrderSide(order.action), order.quantity, execution_price)
        
        if order.action == 'BUY':
            self.current_balance -= (order.quantity * execution_price + commission)
        else:
            self.current_balance += (order.quantity * execution_price - commission)
        
        self.logger.info(f"Executed {order.action} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
    
    async def _handle_limit_order(self, order: Order):
        """Handle limit order (simplified - would need continuous monitoring)"""
        # For simulation, execute immediately if price is favorable
        if order.symbol not in self.market_data:
            order.status = OrderStatus.REJECTED
            return
        
        market_price = self.market_data[order.symbol]
        
        should_execute = False
        if order.action == 'BUY' and market_price <= order.price:
            should_execute = True
        elif order.action == 'SELL' and market_price >= order.price:
            should_execute = True
        
        if should_execute:
            # Execute at limit price
            if self._can_execute_order(order, order.price):
                commission = order.quantity * order.price * self.commission_rate
                
                trade = Trade(
                    id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=OrderSide(order.action),
                    quantity=order.quantity,
                    price=order.price,
                    commission=commission,
                    timestamp=datetime.now(),
                    broker=self.name
                )
                
                self.trades.append(trade)
                
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = order.price
                order.commission = commission
                order.updated_at = datetime.now()
                
                self._update_position(order.symbol, OrderSide(order.action), order.quantity, order.price)
                
                if order.action == 'BUY':
                    self.current_balance -= (order.quantity * order.price + commission)
                else:
                    self.current_balance += (order.quantity * order.price - commission)
    
    def _can_execute_order(self, order: Order, price: float) -> bool:
        """Check if order can be executed"""
        if order.action == 'BUY':
            required_balance = order.quantity * price * (1 + self.commission_rate)
            return self.current_balance >= required_balance
        else:  # SELL
            current_position = self.positions.get(order.symbol, Position(
                symbol=order.symbol, quantity=0, average_price=0, market_value=0,
                unrealized_pnl=0, realized_pnl=0, last_price=price, broker=self.name
            ))
            return current_position.quantity >= order.quantity
    
    def _update_position(self, symbol: str, side: OrderSide, quantity: float, price: float):
        """Update position after trade"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=0, average_price=0, market_value=0,
                unrealized_pnl=0, realized_pnl=0, last_price=price, broker=self.name
            )
        
        position = self.positions[symbol]
        
        if side == OrderSide.BUY:
            # Calculate new average price
            total_cost = position.quantity * position.average_price + quantity * price
            total_quantity = position.quantity + quantity
            
            if total_quantity > 0:
                position.average_price = total_cost / total_quantity
            position.quantity = total_quantity
        else:  # SELL
            # Calculate realized P&L
            if position.quantity > 0:
                realized_pnl = quantity * (price - position.average_price)
                position.realized_pnl += realized_pnl
            
            position.quantity -= quantity
        
        # Update market value and unrealized P&L
        position.last_price = price
        position.market_value = position.quantity * price
        
        if position.quantity > 0:
            position.unrealized_pnl = position.quantity * (price - position.average_price)
        else:
            position.unrealized_pnl = 0
        
        position.updated_at = datetime.now()
        
        # Remove position if quantity is zero
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order"""
        if order_id not in self.orders:
            return {"success": False, "error": "Order not found"}
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return {"success": False, "error": f"Cannot cancel order with status {order.status.value}"}
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        return {"success": True, "order_id": order_id}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if order_id not in self.orders:
            return {"success": False, "error": "Order not found"}
        
        order = self.orders[order_id]
        return {
            "success": True,
            "order": order
        }
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        total_value = self.current_balance
        total_pnl = 0
        
        for position in self.positions.values():
            total_value += position.market_value
            total_pnl += position.unrealized_pnl + position.realized_pnl
        
        return {
            "balance": self.current_balance,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "positions_count": len(self.positions),
            "orders_count": len([o for o in self.orders.values() if o.status == OrderStatus.SUBMITTED])
        }

class TradeExecutor:
    """Handles trade execution across multiple brokers"""
    
    def __init__(self, brokers: Dict[str, Any]):
        self.brokers = brokers
        self.orders = {}
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        
        # Execution settings
        self.max_slippage = 0.005  # 0.5%
        self.order_timeout = 300  # 5 minutes
        self.retry_attempts = 3
        
        logger.info("Trade Executor initialized")
    
    async def execute_order(self, order_data: Dict) -> Dict[str, Any]:
        """Execute a trading order"""
        try:
            # Create order object
            order = Order(
                id=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order_data['symbol']}",
                symbol=order_data['symbol'],
                side=order_data['action'].upper(),
                quantity=order_data.get('quantity', 1),
                order_type=OrderType(order_data.get('order_type', 'MARKET')),
                price=order_data.get('price'),
                stop_price=order_data.get('stop_loss'),
                broker=self._select_broker(order_data['symbol'])
            )
            
            # Store order
            self.orders[order.id] = order
            
            # Execute based on order type
            if order.order_type == OrderType.MARKET:
                result = await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(order)
            elif order.order_type == OrderType.STOP_LOSS:
                result = await self._execute_stop_order(order)
            else:
                result = {'success': False, 'error': 'Unsupported order type'}
            
            # Update order status
            if result['success']:
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = result.get('fill_price', order.price or 0)
                
                # Update positions
                await self._update_position(order)
                
                # Log trade
                self._log_trade(order, result)
            else:
                order.status = OrderStatus.REJECTED
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_market_order(self, order: Order) -> Dict[str, Any]:
        """Execute market order"""
        try:
            broker = self.brokers.get(order.broker)
            if not broker:
                return {'success': False, 'error': 'Broker not available'}
            
            # Get current price for validation
            current_price = await self._get_current_price(order.symbol, order.broker)
            if current_price == 0:
                return {'success': False, 'error': 'Unable to get current price'}
            
            # Execute order through broker
            if hasattr(broker, 'place_market_order'):
                result = await broker.place_market_order(
                    order.symbol, order.side, order.quantity
                )
            else:
                # Fallback to generic order placement
                result = await broker.place_order(
                    order.symbol, order.side, 'MARKET', order.quantity
                )
            
            if result and 'orderId' in result:
                # Wait for fill confirmation
                fill_result = await self._wait_for_fill(order, result['orderId'])
                return {
                    'success': True,
                    'order_id': result['orderId'],
                    'fill_price': fill_result.get('price', current_price),
                    'filled_quantity': fill_result.get('quantity', order.quantity)
                }
            else:
                return {'success': False, 'error': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing market order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_limit_order(self, order: Order) -> Dict[str, Any]:
        """Execute limit order"""
        try:
            broker = self.brokers.get(order.broker)
            if not broker:
                return {'success': False, 'error': 'Broker not available'}
            
            if not order.price:
                return {'success': False, 'error': 'Limit price required'}
            
            # Execute order through broker
            if hasattr(broker, 'place_limit_order'):
                result = await broker.place_limit_order(
                    order.symbol, order.side, order.quantity, order.price
                )
            else:
                result = await broker.place_order(
                    order.symbol, order.side, 'LIMIT', order.quantity, price=order.price
                )
            
            if result and 'orderId' in result:
                return {
                    'success': True,
                    'order_id': result['orderId'],
                    'status': 'PENDING',
                    'limit_price': order.price
                }
            else:
                return {'success': False, 'error': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing limit order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_stop_order(self, order: Order) -> Dict[str, Any]:
        """Execute stop loss order"""
        try:
            broker = self.brokers.get(order.broker)
            if not broker:
                return {'success': False, 'error': 'Broker not available'}
            
            if not order.stop_price:
                return {'success': False, 'error': 'Stop price required'}
            
            # Execute stop order through broker
            if hasattr(broker, 'place_stop_loss_order'):
                result = await broker.place_stop_loss_order(
                    order.symbol, order.side, order.quantity, order.stop_price
                )
            else:
                result = await broker.place_order(
                    order.symbol, order.side, 'STOP_LOSS', order.quantity, 
                    stop_price=order.stop_price
                )
            
            if result and 'orderId' in result:
                return {
                    'success': True,
                    'order_id': result['orderId'],
                    'status': 'PENDING',
                    'stop_price': order.stop_price
                }
            else:
                return {'success': False, 'error': 'Order placement failed'}
                
        except Exception as e:
            logger.error(f"Error executing stop order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _wait_for_fill(self, order: Order, order_id: str) -> Dict[str, Any]:
        """Wait for order to be filled"""
        try:
            broker = self.brokers.get(order.broker)
            timeout = datetime.now() + timedelta(seconds=self.order_timeout)
            
            while datetime.now() < timeout:
                # Check order status
                if hasattr(broker, 'get_order'):
                    order_status = await broker.get_order(order.symbol, order_id)
                    
                    if order_status.get('status') == 'FILLED':
                        return {
                            'price': float(order_status.get('price', 0)),
                            'quantity': float(order_status.get('executedQty', order.quantity))
                        }
                    elif order_status.get('status') in ['CANCELLED', 'REJECTED']:
                        break
                
                await asyncio.sleep(1)  # Wait 1 second before checking again
            
            # Timeout or order not filled
            return {'price': 0, 'quantity': 0}
            
        except Exception as e:
            logger.error(f"Error waiting for fill: {e}")
            return {'price': 0, 'quantity': 0}
    
    async def _get_current_price(self, symbol: str, broker_name: str) -> float:
        """Get current price for symbol"""
        try:
            broker = self.brokers.get(broker_name)
            if not broker:
                return 0.0
            
            if hasattr(broker, 'get_current_price'):
                return await broker.get_current_price(symbol)
            elif hasattr(broker, 'get_ticker_price'):
                ticker = await broker.get_ticker_price(symbol)
                return float(ticker.get('price', 0))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def _select_broker(self, symbol: str) -> str:
        """Select best broker for symbol"""
        # Simple broker selection logic
        for broker_name, broker in self.brokers.items():
            if hasattr(broker, 'supports_symbol'):
                if broker.supports_symbol(symbol):
                    return broker_name
            else:
                # Assume broker supports symbol if method not available
                return broker_name
        
        # Return first available broker as fallback
        return list(self.brokers.keys())[0] if self.brokers else None
    
    async def _update_position(self, order: Order):
        """Update position after order execution"""
        try:
            position_key = f"{order.symbol}_{order.broker}"
            
            if position_key in self.positions:
                position = self.positions[position_key]
                
                if order.side == 'BUY':
                    # Add to position
                    total_cost = (position.quantity * position.avg_price + 
                                 order.filled_quantity * order.avg_fill_price)
                    total_quantity = position.quantity + order.filled_quantity
                    position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                    position.quantity = total_quantity
                else:  # SELL
                    # Reduce position
                    position.quantity -= order.filled_quantity
                    
                    # Calculate realized PnL
                    realized_pnl = (order.avg_fill_price - position.avg_price) * order.filled_quantity
                    position.realized_pnl += realized_pnl
                
                # Update current price and unrealized PnL
                current_price = await self._get_current_price(order.symbol, order.broker)
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                position.timestamp = datetime.now()
                
                # Remove position if quantity is zero
                if abs(position.quantity) < 0.001:
                    del self.positions[position_key]
            else:
                # Create new position
                if order.side == 'BUY':
                    current_price = await self._get_current_price(order.symbol, order.broker)
                    self.positions[position_key] = Position(
                        symbol=order.symbol,
                        quantity=order.filled_quantity,
                        avg_price=order.avg_fill_price,
                        current_price=current_price,
                        unrealized_pnl=(current_price - order.avg_fill_price) * order.filled_quantity,
                        realized_pnl=0.0,
                        timestamp=datetime.now(),
                        broker=order.broker
                    )
                    
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    def _log_trade(self, order: Order, result: Dict):
        """Log completed trade"""
        trade_log = {
            'timestamp': datetime.now(),
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.avg_fill_price,
            'order_type': order.order_type.value,
            'broker': order.broker,
            'status': 'FILLED',
            'result': result
        }
        
        self.trade_history.append(trade_log)
        logger.info(f"Trade logged: {order.side} {order.filled_quantity} {order.symbol} @ {order.avg_fill_price}")
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            if order_id not in self.orders:
                return {'success': False, 'error': 'Order not found'}
            
            order = self.orders[order_id]
            broker = self.brokers.get(order.broker)
            
            if not broker:
                return {'success': False, 'error': 'Broker not available'}
            
            # Cancel order through broker
            if hasattr(broker, 'cancel_order'):
                result = await broker.cancel_order(order.symbol, order_id)
                
                if result:
                    order.status = OrderStatus.CANCELLED
                    return {'success': True, 'message': 'Order cancelled'}
                else:
                    return {'success': False, 'error': 'Cancellation failed'}
            else:
                return {'success': False, 'error': 'Broker does not support cancellation'}
                
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_position(self, symbol: str, broker: str = None) -> Dict[str, Any]:
        """Close a position"""
        try:
            # Find position
            position_key = f"{symbol}_{broker}" if broker else None
            position = None
            
            if position_key and position_key in self.positions:
                position = self.positions[position_key]
            else:
                # Find position across all brokers
                for key, pos in self.positions.items():
                    if pos.symbol == symbol:
                        position = pos
                        break
            
            if not position:
                return {'success': False, 'error': 'Position not found'}
            
            # Create closing order
            close_side = 'SELL' if position.quantity > 0 else 'BUY'
            close_quantity = abs(position.quantity)
            
            order_data = {
                'symbol': symbol,
                'action': close_side,
                'quantity': close_quantity,
                'order_type': 'MARKET'
            }
            
            result = await self.execute_order(order_data)
            
            if result['success']:
                return {
                    'success': True,
                    'message': f'Position closed: {close_side} {close_quantity} {symbol}',
                    'realized_pnl': position.realized_pnl + position.unrealized_pnl
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions"""
        try:
            results = []
            
            for position_key, position in list(self.positions.items()):
                result = await self.close_position(position.symbol, position.broker)
                results.append({
                    'symbol': position.symbol,
                    'result': result
                })
            
            return {
                'success': True,
                'message': f'Attempted to close {len(results)} positions',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        try:
            total_value = 0.0
            total_pnl = 0.0
            total_unrealized_pnl = 0.0
            total_realized_pnl = 0.0
            
            positions_list = []
            
            for position in self.positions.values():
                # Update current price and unrealized PnL
                current_price = await self._get_current_price(position.symbol, position.broker)
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                
                position_value = abs(position.quantity * current_price)
                total_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
                
                positions_list.append({
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'avg_price': position.avg_price,
                    'current_price': position.current_price,
                    'market_value': position_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'broker': position.broker
                })
            
            total_pnl = total_realized_pnl + total_unrealized_pnl
            
            return {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'positions': positions_list,
                'open_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")
            return {
                'total_value': 0.0,
                'total_pnl': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'positions': [],
                'open_orders': 0,
                'total_trades': 0
            }
    
    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history"""
        return self.trade_history[-limit:] if limit else self.trade_history
    
    def get_open_orders(self) -> List[Dict]:
        """Get open orders"""
        open_orders = []
        
        for order in self.orders.values():
            if order.status == OrderStatus.PENDING:
                open_orders.append({
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'order_type': order.order_type.value,
                    'price': order.price,
                    'stop_price': order.stop_price,
                    'timestamp': order.timestamp,
                    'broker': order.broker
                })
        
        return open_orders
    
    async def update_positions(self):
        """Update all positions with current prices"""
        try:
                    'timestamp': order.timestamp,
                    'broker': order.broker
                })
        
        return open_orders
    
    async def update_positions(self):
        """Update all positions with current prices"""
        try:
            for position in self.positions.values():
                current_price = await self._get_current_price(position.symbol, position.broker)
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
                position.timestamp = datetime.now()
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    def start(self):
        """Start trade executor"""
        self.is_running = True
        logger.info("Trade Executor started")
    
    def stop(self):
        """Stop trade executor"""
        self.is_running = False
        logger.info("Trade Executor stopped")

async def main():
    """Test the trade executor"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Mock broker APIs
    class MockBrokerAPI:
        def __init__(self, name):
            self.name = name

        async def supports_symbol(self, symbol: str) -> bool:
            """Mock implementation to check if broker supports a symbol"""
            return True

        async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
            """Mock implementation to place a market order"""
            logger.info(f"{self.name}: Placing market order for {quantity} {symbol} {side}")
            return {
                'success': True,
                'orderId': f"{self.name}_market_{int(time.time())}",
                'filled': True,
                'executedQty': quantity,
                'price': 150.0 if symbol == 'AAPL' else 2500.0
            }

        async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
            """Mock implementation to place a limit order"""
            logger.info(f"{self.name}: Placing limit order for {quantity} {symbol} {side} at {price}")
            return {
                'success': True,
                'orderId': f"{self.name}_limit_{int(time.time())}",
                'filled': False,
                'executedQty': 0,
                'price': price
            }

        async def place_stop_loss_order(self, symbol: str, side: str, quantity: float, stop_price: float) -> Dict:
            """Mock implementation to place a stop loss order"""
            logger.info(f"{self.name}: Placing stop loss order for {quantity} {symbol} {side} at {stop_price}")
            return {
                'success': True,
                'orderId': f"{self.name}_stop_{int(time.time())}",
                'filled': False,
                'executedQty': 0,
                'stop_price': stop_price
            }

        async def cancel_order(self, symbol: str, order_id: str) -> Dict:
            """Mock implementation to cancel an order"""
            logger.info(f"{self.name}: Cancelling order {order_id} for {symbol}")
            return {'success': True}

        async def get_current_price(self, symbol: str) -> float:
            """Mock implementation to get current price"""
            if symbol == 'AAPL':
                return 152.0
            elif symbol == 'GOOGL':
                return 2550.0
            else:
                return 100.0

        async def get_order(self, symbol: str, order_id: str) -> Dict:
            """Mock implementation to get order status"""
            if "market" in order_id:
                return {'status': 'FILLED', 'price': 151.0, 'executedQty': 10}
            else:
                return {'status': 'PENDING', 'price': 149.0, 'executedQty': 0}

    # Initialize brokers
    brokers = {
        'broker1': MockBrokerAPI('Broker1'),
        'broker2': MockBrokerAPI('Broker2')
    }

    # Create trade executor
    executor = TradeExecutor(brokers=brokers)
    executor.start()

    # Test order
    order_data = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 10,
        'order_type': 'MARKET'
    }

    # Execute order
    result = await executor.execute_order(order_data)
    logger.info(f"Order execution result: {result}")

    # Get portfolio status
    portfolio_status = await executor.get_portfolio_status()
    logger.info(f"Portfolio status: {portfolio_status}")

    # Close position
    close_result = await executor.close_position('AAPL')
    logger.info(f"Close position result: {close_result}")

    # Get portfolio status after closing
    portfolio_status = await executor.get_portfolio_status()
    logger.info(f"Portfolio status after closing: {portfolio_status}")

    executor.stop()

if __name__ == "__main__":
    asyncio.run(main())
