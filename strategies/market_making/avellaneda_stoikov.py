"""
Avellaneda-Stoikov Market Making Strategy

Implementation of the optimal market making model from:
"High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)

This strategy optimally sets bid/ask prices to maximize expected utility
while managing inventory risk.
"""

from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class MarketData:
    """Market data snapshot"""
    timestamp: int
    symbol: str
    best_bid: float
    best_ask: float
    bid_volume: float
    ask_volume: float
    mid_price: float
    spread: float

@dataclass
class Position:
    """Position information"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class AvellanedaStoikovParams:
    """Strategy parameters for Avellaneda-Stoikov model"""
    # Risk aversion parameter
    gamma: float = 0.1
    
    # Inventory penalty parameter  
    k: float = 1.5
    
    # Market impact parameter
    A: float = 140.0
    
    # Time to maturity (seconds)
    T: float = 86400.0  # 1 day
    
    # Volatility estimate
    sigma: float = 0.02
    
    # Tick size
    tick_size: float = 0.01
    
    # Maximum spread
    max_spread: float = 0.05
    
    # Order size
    order_size: float = 100.0
    
    # Maximum inventory
    max_inventory: float = 1000.0

class AvellanedaStoikovStrategy:
    """
    Avellaneda-Stoikov optimal market making strategy
    
    The strategy computes optimal bid/ask quotes based on:
    - Current inventory level
    - Market volatility
    - Risk aversion
    - Time to horizon
    """
    
    def __init__(self, params: AvellanedaStoikovParams):
        self.params = params
        self.position: Optional[Position] = None
        self.last_market_data: Optional[MarketData] = None
        self.start_time: Optional[int] = None
        
    def on_market_data(self, data: MarketData) -> Dict[str, Any]:
        """
        Process market data and generate quotes
        
        Returns:
            Dictionary containing bid/ask prices and quantities
        """
        self.last_market_data = data
        
        if self.start_time is None:
            self.start_time = data.timestamp
            
        # Calculate time remaining until horizon
        elapsed_time = (data.timestamp - self.start_time) / 1_000_000_000  # ns to seconds
        time_remaining = max(self.params.T - elapsed_time, 1.0)
        
        # Get current inventory
        inventory = self.position.quantity if self.position else 0.0
        
        # Calculate reservation price
        reservation_price = self._calculate_reservation_price(
            data.mid_price, inventory, time_remaining
        )
        
        # Calculate optimal spread
        optimal_spread = self._calculate_optimal_spread(time_remaining)
        
        # Generate bid/ask quotes
        bid_price = reservation_price - optimal_spread / 2
        ask_price = reservation_price + optimal_spread / 2
        
        # Apply tick size rounding
        bid_price = self._round_to_tick(bid_price)
        ask_price = self._round_to_tick(ask_price)
        
        # Ensure minimum spread
        spread = ask_price - bid_price
        if spread < self.params.tick_size:
            bid_price -= self.params.tick_size / 2
            ask_price += self.params.tick_size / 2
            
        # Apply maximum spread constraint
        if spread > self.params.max_spread:
            mid = (bid_price + ask_price) / 2
            bid_price = mid - self.params.max_spread / 2
            ask_price = mid + self.params.max_spread / 2
            
        # Check inventory limits
        quotes = {}
        
        if abs(inventory) < self.params.max_inventory:
            if inventory <= 0:  # Can buy more
                quotes['bid'] = {
                    'price': bid_price,
                    'quantity': self.params.order_size,
                    'side': 'BUY'
                }
                
            if inventory >= 0:  # Can sell more  
                quotes['ask'] = {
                    'price': ask_price,
                    'quantity': self.params.order_size,
                    'side': 'SELL'
                }
        
        return quotes
        
    def _calculate_reservation_price(self, mid_price: float, inventory: float, 
                                   time_remaining: float) -> float:
        """
        Calculate the reservation price based on current inventory and time
        
        r = S - q * gamma * sigma^2 * (T - t)
        
        Where:
        - S is the mid price
        - q is the inventory
        - gamma is risk aversion
        - sigma is volatility
        - T-t is time remaining
        """
        inventory_adjustment = (inventory * self.params.gamma * 
                              self.params.sigma**2 * time_remaining)
        
        return mid_price - inventory_adjustment
        
    def _calculate_optimal_spread(self, time_remaining: float) -> float:
        """
        Calculate optimal spread based on market parameters
        
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        """
        risk_term = self.params.gamma * self.params.sigma**2 * time_remaining
        
        market_impact_term = (2.0 / self.params.gamma * 
                             np.log(1 + self.params.gamma / self.params.k))
        
        optimal_spread = risk_term + market_impact_term
        
        # Ensure minimum spread
        return max(optimal_spread, self.params.tick_size)
        
    def _round_to_tick(self, price: float) -> float:
        """Round price to nearest tick size"""
        return round(price / self.params.tick_size) * self.params.tick_size
        
    def on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order updates (fills, cancellations, etc.)"""
        if order_update.get('status') == 'FILLED':
            # Update position
            fill_qty = order_update['filled_quantity']
            fill_price = order_update['fill_price']
            side = order_update['side']
            
            if side == 'BUY':
                self._update_position(fill_qty, fill_price)
            else:  # SELL
                self._update_position(-fill_qty, fill_price)
                
    def _update_position(self, quantity_delta: float, price: float) -> None:
        """Update position with new fill"""
        if self.position is None:
            self.position = Position(
                symbol=self.last_market_data.symbol,
                quantity=quantity_delta,
                avg_price=price,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )
        else:
            # Update position with FIFO accounting
            old_qty = self.position.quantity
            new_qty = old_qty + quantity_delta
            
            if old_qty * new_qty >= 0:  # Same side or crossing zero
                # Weighted average price calculation
                total_cost = (old_qty * self.position.avg_price + 
                             quantity_delta * price)
                self.position.avg_price = total_cost / new_qty if new_qty != 0 else 0
                self.position.quantity = new_qty
            else:  # Position reduction
                # Realize PnL on the reduced portion
                reduced_qty = min(abs(quantity_delta), abs(old_qty))
                if old_qty > 0:  # Reducing long position
                    pnl = reduced_qty * (price - self.position.avg_price)
                else:  # Reducing short position
                    pnl = reduced_qty * (self.position.avg_price - price)
                    
                self.position.realized_pnl += pnl
                self.position.quantity = new_qty
                
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics"""
        if self.position is None:
            return {}
            
        unrealized_pnl = 0.0
        if self.last_market_data and self.position.quantity != 0:
            current_price = self.last_market_data.mid_price
            unrealized_pnl = (self.position.quantity * 
                             (current_price - self.position.avg_price))
                             
        total_pnl = self.position.realized_pnl + unrealized_pnl
        
        return {
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'position': self.position.quantity,
            'avg_price': self.position.avg_price
        }
