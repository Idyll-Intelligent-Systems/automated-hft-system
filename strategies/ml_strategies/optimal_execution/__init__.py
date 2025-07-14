"""
Optimal Execution Module
======================

Implements Step 7 of the ML modeling workflow from Modeling.md:
Optimal Order Placement Logic (Avellaneda-Stoikov, Almgren-Chriss, RL-based Execution)

Decides precise entry and exit levels based on model's forecast,
risk management criteria, and position inventory constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
import math

from .base_ml_strategy import MarketMicrostructureData, MLPrediction, RiskConstraints

logger = logging.getLogger(__name__)

@dataclass
class ExecutionParams:
    """Parameters for optimal execution"""
    # Avellaneda-Stoikov parameters
    risk_aversion: float = 0.1
    inventory_penalty: float = 1.5
    market_impact: float = 140.0
    time_to_maturity: float = 86400.0  # 1 day in seconds
    
    # Almgren-Chriss parameters
    volatility: float = 0.02
    temporary_impact_coeff: float = 0.1
    permanent_impact_coeff: float = 0.01
    
    # General execution parameters
    max_participation_rate: float = 0.1  # Max 10% of volume
    min_order_size: float = 100.0
    max_order_size: float = 10000.0
    tick_size: float = 0.01

@dataclass
class OrderRecommendation:
    """Optimal order recommendation"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str
    time_in_force: str
    urgency: float  # 0-1, how urgent the execution is
    expected_cost: float
    confidence: float
    reasoning: str

class OptimalExecutionEngine(ABC):
    """Abstract base class for optimal execution algorithms"""
    
    @abstractmethod
    def calculate_optimal_orders(self,
                               prediction: MLPrediction,
                               current_position: float,
                               target_position: float,
                               market_data: MarketMicrostructureData,
                               params: ExecutionParams) -> List[OrderRecommendation]:
        """Calculate optimal orders based on execution algorithm"""
        pass

class AvellanedaStoikovExecutor(OptimalExecutionEngine):
    """
    Avellaneda-Stoikov optimal market making execution
    
    Uses stochastic optimal control for inventory-based market making.
    Optimally sets bid/ask prices to maximize expected utility while managing inventory risk.
    """
    
    def __init__(self):
        self.name = "Avellaneda-Stoikov"
        
    def calculate_optimal_orders(self,
                               prediction: MLPrediction,
                               current_position: float,
                               target_position: float,
                               market_data: MarketMicrostructureData,
                               params: ExecutionParams) -> List[OrderRecommendation]:
        """Calculate optimal bid/ask quotes for market making"""
        
        orders = []
        
        try:
            # Calculate reservation price
            reservation_price = self._calculate_reservation_price(
                market_data.mid_price,
                current_position,
                params
            )
            
            # Calculate optimal spreads
            bid_spread, ask_spread = self._calculate_optimal_spreads(
                market_data,
                current_position,
                params
            )
            
            # Calculate order quantities
            bid_quantity, ask_quantity = self._calculate_order_quantities(
                current_position,
                target_position,
                market_data,
                params
            )
            
            # Create bid order
            if bid_quantity > params.min_order_size:
                bid_price = reservation_price - bid_spread
                bid_price = self._round_to_tick(bid_price, params.tick_size)
                
                orders.append(OrderRecommendation(
                    symbol=market_data.symbol,
                    side='BUY',
                    quantity=bid_quantity,
                    price=bid_price,
                    order_type='LIMIT',
                    time_in_force='GTC',
                    urgency=0.3,
                    expected_cost=self._calculate_expected_cost(bid_quantity, bid_spread, params),
                    confidence=prediction.confidence,
                    reasoning=f"Avellaneda-Stoikov bid: reservation={reservation_price:.4f}, spread={bid_spread:.4f}"
                ))
            
            # Create ask order
            if ask_quantity > params.min_order_size:
                ask_price = reservation_price + ask_spread
                ask_price = self._round_to_tick(ask_price, params.tick_size)
                
                orders.append(OrderRecommendation(
                    symbol=market_data.symbol,
                    side='SELL',
                    quantity=ask_quantity,
                    price=ask_price,
                    order_type='LIMIT',
                    time_in_force='GTC',
                    urgency=0.3,
                    expected_cost=self._calculate_expected_cost(ask_quantity, ask_spread, params),
                    confidence=prediction.confidence,
                    reasoning=f"Avellaneda-Stoikov ask: reservation={reservation_price:.4f}, spread={ask_spread:.4f}"
                ))
            
        except Exception as e:
            logger.error(f"Error in Avellaneda-Stoikov execution: {e}")
        
        return orders
    
    def _calculate_reservation_price(self,
                                   mid_price: float,
                                   inventory: float,
                                   params: ExecutionParams) -> float:
        """Calculate reservation price based on inventory and risk aversion"""
        
        # Risk adjustment based on inventory
        inventory_adjustment = params.risk_aversion * params.volatility**2 * inventory * params.time_to_maturity
        
        reservation_price = mid_price - inventory_adjustment
        
        return reservation_price
    
    def _calculate_optimal_spreads(self,
                                 market_data: MarketMicrostructureData,
                                 inventory: float,
                                 params: ExecutionParams) -> Tuple[float, float]:
        """Calculate optimal bid and ask spreads"""
        
        # Base spread from Avellaneda-Stoikov formula
        gamma = params.risk_aversion
        sigma = params.volatility
        T = params.time_to_maturity
        A = params.market_impact
        
        # Optimal spread
        spread = gamma * sigma**2 * T + (2/gamma) * np.log(1 + gamma/A)
        
        # Adjust for current inventory (skew the spreads)
        inventory_skew = gamma * sigma**2 * T * inventory
        
        bid_spread = spread/2 + inventory_skew
        ask_spread = spread/2 - inventory_skew
        
        # Ensure spreads are positive and reasonable
        min_spread = market_data.spread * 0.5  # At least half of current spread
        max_spread = market_data.spread * 3.0  # At most 3x current spread
        
        bid_spread = max(min_spread, min(bid_spread, max_spread))
        ask_spread = max(min_spread, min(ask_spread, max_spread))
        
        return bid_spread, ask_spread
    
    def _calculate_order_quantities(self,
                                  current_position: float,
                                  target_position: float,
                                  market_data: MarketMicrostructureData,
                                  params: ExecutionParams) -> Tuple[float, float]:
        """Calculate optimal order quantities"""
        
        # Base quantity from available volume
        base_quantity = min(
            market_data.bid_volume * params.max_participation_rate,
            params.max_order_size
        )
        
        # Adjust based on inventory needs
        position_diff = target_position - current_position
        
        if position_diff > 0:
            # Need to buy more
            bid_quantity = min(base_quantity, abs(position_diff))
            ask_quantity = base_quantity * 0.5  # Reduce ask quantity
        elif position_diff < 0:
            # Need to sell
            ask_quantity = min(base_quantity, abs(position_diff))
            bid_quantity = base_quantity * 0.5  # Reduce bid quantity
        else:
            # Balanced market making
            bid_quantity = base_quantity
            ask_quantity = base_quantity
        
        return max(bid_quantity, params.min_order_size), max(ask_quantity, params.min_order_size)
    
    def _calculate_expected_cost(self, quantity: float, spread: float, params: ExecutionParams) -> float:
        """Calculate expected execution cost"""
        # Simple cost model: half-spread + impact
        impact_cost = params.permanent_impact_coeff * quantity
        spread_cost = spread / 2
        
        return quantity * (spread_cost + impact_cost)
    
    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """Round price to nearest tick"""
        return round(price / tick_size) * tick_size

class AlmgrenChrissExecutor(OptimalExecutionEngine):
    """
    Almgren-Chriss optimal execution algorithm
    
    Minimizes expected market-impact cost with volatility constraints.
    Optimal for executing large orders over time.
    """
    
    def __init__(self):
        self.name = "Almgren-Chriss"
    
    def calculate_optimal_orders(self,
                               prediction: MLPrediction,
                               current_position: float,
                               target_position: float,
                               market_data: MarketMicrostructureData,
                               params: ExecutionParams) -> List[OrderRecommendation]:
        """Calculate optimal execution schedule using Almgren-Chriss"""
        
        orders = []
        
        try:
            # Calculate total shares to trade
            total_shares = target_position - current_position
            
            if abs(total_shares) < params.min_order_size:
                return orders
            
            # Get optimal execution strategy
            execution_schedule = self._solve_almgren_chriss(
                total_shares,
                market_data,
                params
            )
            
            # Create immediate order from first step
            if execution_schedule and len(execution_schedule) > 0:
                first_trade = execution_schedule[0]
                
                if abs(first_trade) >= params.min_order_size:
                    side = 'BUY' if first_trade > 0 else 'SELL'
                    quantity = abs(first_trade)
                    
                    # Calculate aggressive price based on urgency
                    if side == 'BUY':
                        price = market_data.best_ask + params.tick_size  # Slightly aggressive
                    else:
                        price = market_data.best_bid - params.tick_size  # Slightly aggressive
                    
                    urgency = self._calculate_urgency(prediction, params)
                    
                    orders.append(OrderRecommendation(
                        symbol=market_data.symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        order_type='LIMIT',
                        time_in_force='IOC',  # Immediate or Cancel for aggressive execution
                        urgency=urgency,
                        expected_cost=self._calculate_execution_cost(quantity, market_data, params),
                        confidence=prediction.confidence,
                        reasoning=f"Almgren-Chriss optimal execution: step 1 of {len(execution_schedule)}"
                    ))
            
        except Exception as e:
            logger.error(f"Error in Almgren-Chriss execution: {e}")
        
        return orders
    
    def _solve_almgren_chriss(self,
                            total_shares: float,
                            market_data: MarketMicrostructureData,
                            params: ExecutionParams) -> List[float]:
        """Solve Almgren-Chriss optimal execution problem"""
        
        # Problem parameters
        T = params.time_to_maturity  # Total execution time
        sigma = params.volatility
        alpha = params.permanent_impact_coeff
        eta = params.temporary_impact_coeff
        gamma = params.risk_aversion
        
        # Discretize time
        n_steps = min(100, max(10, int(T / 60)))  # 1 minute steps, capped at 100
        dt = T / n_steps
        
        # Calculate optimal trajectory
        kappa = np.sqrt(gamma * sigma**2 / eta)
        
        # Analytical solution for Almgren-Chriss
        trajectory = []
        
        for i in range(n_steps + 1):
            t = i * dt
            remaining_time = T - t
            
            if remaining_time <= 0:
                x = 0
            else:
                # Optimal holdings at time t
                sinh_term = np.sinh(kappa * remaining_time)
                sinh_T = np.sinh(kappa * T)
                
                if sinh_T > 0:
                    x = total_shares * (sinh_term / sinh_T)
                else:
                    x = total_shares * (remaining_time / T)  # Linear fallback
            
            trajectory.append(x)
        
        # Convert to trading schedule (differences)
        execution_schedule = []
        for i in range(len(trajectory) - 1):
            trade_size = trajectory[i] - trajectory[i + 1]
            execution_schedule.append(trade_size)
        
        return execution_schedule
    
    def _calculate_urgency(self, prediction: MLPrediction, params: ExecutionParams) -> float:
        """Calculate execution urgency based on prediction confidence and market conditions"""
        
        # Base urgency from prediction confidence
        urgency = prediction.confidence
        
        # Adjust based on prediction magnitude
        if hasattr(prediction.prediction_value, '__abs__'):
            urgency *= abs(prediction.prediction_value)
        
        # Ensure urgency is between 0 and 1
        return max(0.0, min(1.0, urgency))
    
    def _calculate_execution_cost(self,
                                quantity: float,
                                market_data: MarketMicrostructureData,
                                params: ExecutionParams) -> float:
        """Calculate expected execution cost using Almgren-Chriss cost model"""
        
        # Market impact costs
        permanent_cost = params.permanent_impact_coeff * quantity
        temporary_cost = params.temporary_impact_coeff * quantity
        
        # Add spread cost
        spread_cost = market_data.spread / 2
        
        total_cost = quantity * (permanent_cost + temporary_cost + spread_cost)
        
        return total_cost

class ReinforcementLearningExecutor(OptimalExecutionEngine):
    """
    Reinforcement Learning-based optimal execution
    
    Uses trained RL agent to make execution decisions based on current market state.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.name = "Reinforcement Learning"
        self.model_path = model_path
        self.agent = None
        
        # Load pre-trained agent if available
        if model_path:
            self._load_agent(model_path)
    
    def calculate_optimal_orders(self,
                               prediction: MLPrediction,
                               current_position: float,
                               target_position: float,
                               market_data: MarketMicrostructureData,
                               params: ExecutionParams) -> List[OrderRecommendation]:
        """Use RL agent to determine optimal execution"""
        
        orders = []
        
        if self.agent is None:
            logger.warning("RL agent not loaded, falling back to simple execution")
            return self._fallback_execution(prediction, current_position, target_position, market_data, params)
        
        try:
            # Prepare state vector for RL agent
            state = self._prepare_state(prediction, current_position, target_position, market_data, params)
            
            # Get action from RL agent
            action = self.agent.predict(state)[0]
            
            # Interpret action and create orders
            orders = self._interpret_rl_action(action, market_data, params)
            
        except Exception as e:
            logger.error(f"Error in RL execution: {e}")
            # Fallback to simple execution
            orders = self._fallback_execution(prediction, current_position, target_position, market_data, params)
        
        return orders
    
    def _load_agent(self, model_path: str) -> None:
        """Load pre-trained RL agent"""
        try:
            # This would load a trained RL model
            # For now, we'll use a placeholder
            logger.info(f"Loading RL agent from {model_path}")
            # self.agent = load_model(model_path)
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            self.agent = None
    
    def _prepare_state(self,
                      prediction: MLPrediction,
                      current_position: float,
                      target_position: float,
                      market_data: MarketMicrostructureData,
                      params: ExecutionParams) -> np.ndarray:
        """Prepare state vector for RL agent"""
        
        state = [
            # Market features
            market_data.order_book_imbalance,
            market_data.spread / market_data.mid_price,
            market_data.bid_volume,
            market_data.ask_volume,
            
            # Position features
            current_position / params.max_order_size,
            target_position / params.max_order_size,
            (target_position - current_position) / params.max_order_size,
            
            # Prediction features
            prediction.prediction_value,
            prediction.confidence,
            
            # Normalized time features
            datetime.now().hour / 24,
            datetime.now().minute / 60,
        ]
        
        return np.array(state, dtype=np.float32)
    
    def _interpret_rl_action(self,
                           action: np.ndarray,
                           market_data: MarketMicrostructureData,
                           params: ExecutionParams) -> List[OrderRecommendation]:
        """Interpret RL agent action and create orders"""
        
        orders = []
        
        # Assume action is [position_change, price_aggressiveness, urgency]
        if len(action) >= 3:
            position_change = action[0] * params.max_order_size
            price_aggressiveness = action[1]  # -1 to 1
            urgency = max(0, min(1, action[2]))
            
            if abs(position_change) >= params.min_order_size:
                side = 'BUY' if position_change > 0 else 'SELL'
                quantity = abs(position_change)
                
                # Calculate price based on aggressiveness
                if side == 'BUY':
                    if price_aggressiveness > 0:
                        # Aggressive - above mid
                        price = market_data.mid_price + (price_aggressiveness * market_data.spread)
                    else:
                        # Passive - at or below mid
                        price = market_data.mid_price + (price_aggressiveness * market_data.spread / 2)
                else:
                    if price_aggressiveness > 0:
                        # Aggressive - below mid
                        price = market_data.mid_price - (price_aggressiveness * market_data.spread)
                    else:
                        # Passive - at or above mid
                        price = market_data.mid_price - (price_aggressiveness * market_data.spread / 2)
                
                price = max(price, params.tick_size)  # Ensure positive price
                
                orders.append(OrderRecommendation(
                    symbol=market_data.symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type='LIMIT',
                    time_in_force='GTC' if urgency < 0.7 else 'IOC',
                    urgency=urgency,
                    expected_cost=quantity * market_data.spread / 2,  # Simple estimate
                    confidence=0.6,  # RL model confidence
                    reasoning=f"RL execution: aggressiveness={price_aggressiveness:.2f}, urgency={urgency:.2f}"
                ))
        
        return orders
    
    def _fallback_execution(self,
                          prediction: MLPrediction,
                          current_position: float,
                          target_position: float,
                          market_data: MarketMicrostructureData,
                          params: ExecutionParams) -> List[OrderRecommendation]:
        """Simple fallback execution when RL agent is not available"""
        
        orders = []
        position_diff = target_position - current_position
        
        if abs(position_diff) >= params.min_order_size:
            side = 'BUY' if position_diff > 0 else 'SELL'
            quantity = min(abs(position_diff), params.max_order_size)
            
            # Use mid price with small offset
            if side == 'BUY':
                price = market_data.mid_price + params.tick_size
            else:
                price = market_data.mid_price - params.tick_size
            
            orders.append(OrderRecommendation(
                symbol=market_data.symbol,
                side=side,
                quantity=quantity,
                price=price,
                order_type='LIMIT',
                time_in_force='GTC',
                urgency=0.5,
                expected_cost=quantity * market_data.spread / 2,
                confidence=prediction.confidence,
                reasoning="Fallback execution due to RL agent unavailability"
            ))
        
        return orders

class OptimalExecutionManager:
    """
    Manager class that coordinates different execution algorithms
    
    Selects appropriate execution algorithm based on market conditions,
    prediction type, and trading objectives.
    """
    
    def __init__(self):
        self.executors = {
            'avellaneda_stoikov': AvellanedaStoikovExecutor(),
            'almgren_chriss': AlmgrenChrissExecutor(),
            'reinforcement_learning': ReinforcementLearningExecutor()
        }
        
        self.default_params = ExecutionParams()
        
    def calculate_optimal_orders(self,
                               prediction: MLPrediction,
                               current_position: float,
                               target_position: float,
                               market_data: MarketMicrostructureData,
                               execution_params: Optional[ExecutionParams] = None,
                               algorithm: str = 'auto') -> List[OrderRecommendation]:
        """
        Calculate optimal orders using specified or automatically selected algorithm
        """
        
        if execution_params is None:
            execution_params = self.default_params
        
        # Auto-select algorithm if not specified
        if algorithm == 'auto':
            algorithm = self._select_algorithm(prediction, current_position, target_position, market_data)
        
        # Get executor
        executor = self.executors.get(algorithm)
        if executor is None:
            logger.warning(f"Unknown execution algorithm: {algorithm}, using Almgren-Chriss")
            executor = self.executors['almgren_chriss']
        
        # Calculate orders
        orders = executor.calculate_optimal_orders(
            prediction, current_position, target_position, market_data, execution_params
        )
        
        # Post-process orders
        validated_orders = self._validate_orders(orders, market_data, execution_params)
        
        return validated_orders
    
    def _select_algorithm(self,
                         prediction: MLPrediction,
                         current_position: float,
                         target_position: float,
                         market_data: MarketMicrostructureData) -> str:
        """Automatically select best execution algorithm based on conditions"""
        
        position_change = abs(target_position - current_position)
        spread_bps = (market_data.spread / market_data.mid_price) * 10000
        
        # Market making conditions - use Avellaneda-Stoikov
        if (position_change < 1000 and  # Small position change
            spread_bps > 5 and           # Wide spread
            prediction.confidence > 0.6): # Good prediction confidence
            return 'avellaneda_stoikov'
        
        # Large order execution - use Almgren-Chriss
        elif position_change > 5000:
            return 'almgren_chriss'
        
        # Medium orders with high confidence - try RL
        elif (position_change > 1000 and
              prediction.confidence > 0.7):
            return 'reinforcement_learning'
        
        # Default to Almgren-Chriss
        else:
            return 'almgren_chriss'
    
    def _validate_orders(self,
                        orders: List[OrderRecommendation],
                        market_data: MarketMicrostructureData,
                        params: ExecutionParams) -> List[OrderRecommendation]:
        """Validate and filter orders"""
        
        validated_orders = []
        
        for order in orders:
            # Check quantity limits
            if order.quantity < params.min_order_size:
                logger.debug(f"Order quantity {order.quantity} below minimum {params.min_order_size}")
                continue
            
            if order.quantity > params.max_order_size:
                logger.warning(f"Order quantity {order.quantity} above maximum {params.max_order_size}, capping")
                order.quantity = params.max_order_size
            
            # Check price sanity
            if order.side == 'BUY':
                if order.price > market_data.best_ask * 1.1:  # More than 10% above ask
                    logger.warning(f"Buy price {order.price} too high, adjusting")
                    order.price = market_data.best_ask * 1.05
            else:
                if order.price < market_data.best_bid * 0.9:  # More than 10% below bid
                    logger.warning(f"Sell price {order.price} too low, adjusting")
                    order.price = market_data.best_bid * 0.95
            
            # Round price to tick size
            order.price = round(order.price / params.tick_size) * params.tick_size
            
            validated_orders.append(order)
        
        return validated_orders
    
    def get_execution_analytics(self, orders: List[OrderRecommendation]) -> Dict[str, Any]:
        """Get analytics on execution recommendations"""
        
        if not orders:
            return {}
        
        total_quantity = sum(order.quantity for order in orders)
        total_cost = sum(order.expected_cost for order in orders)
        avg_urgency = np.mean([order.urgency for order in orders])
        avg_confidence = np.mean([order.confidence for order in orders])
        
        buy_orders = [o for o in orders if o.side == 'BUY']
        sell_orders = [o for o in orders if o.side == 'SELL']
        
        return {
            'total_orders': len(orders),
            'total_quantity': total_quantity,
            'total_expected_cost': total_cost,
            'average_urgency': avg_urgency,
            'average_confidence': avg_confidence,
            'buy_orders': len(buy_orders),
            'sell_orders': len(sell_orders),
            'buy_quantity': sum(o.quantity for o in buy_orders),
            'sell_quantity': sum(o.quantity for o in sell_orders),
            'algorithms_used': list(set(o.reasoning.split(':')[0] for o in orders))
        }
