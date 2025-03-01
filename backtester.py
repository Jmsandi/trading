"""
Backtesting and Optimization Module for Trading Bot

This module provides functions to:
1. Backtest trading strategies on historical data
2. Optimize strategy parameters
3. Visualize performance results
4. Simulate paper trading
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import ccxt
from tqdm import tqdm

# Import strategies from bot.py
from bot import (
    calculate_rsi, calculate_atr, calculate_bollinger_bands,
    trend_trading_strategy, breakout_strategy, range_trading_strategy,
    price_action_strategy, day_trading_strategy, scalping_strategy,
    swing_trading_strategy, carry_trade_strategy, position_trading_strategy,
    news_trading_strategy
)

class BacktestEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {symbol: {'entry_price': X, 'quantity': Y, 'side': 'buy'/'sell'}}
        self.trades = []
        self.equity_curve = []

    def reset(self):
        """Reset the backtester to initial state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def _calculate_position_size(self, price, risk_per_trade, stop_loss):
        """Calculate position size based on risk"""
        if stop_loss is None:
            # Default to 2% below/above entry price if no stop loss defined
            stop_loss_distance = price * 0.02
        else:
            stop_loss_distance = abs(price - stop_loss)
            
        risk_amount = self.current_capital * (risk_per_trade / 100)
        position_size = risk_amount / stop_loss_distance
        
        # Ensure position size doesn't exceed available capital
        max_size = self.current_capital / price
        position_size = min(position_size, max_size)
        
        return position_size
        
    def execute_signal(self, signal, timestamp, risk_per_trade=1):
        """Execute a trading signal"""
        symbol = signal['symbol']
        price = signal['price']
        signal_type = signal['signal']
        stop_loss = signal.get('stop_loss', None)
        take_profit = signal.get('take_profit', None)
        
        # For short positions, swap stop loss and take profit
        if signal_type == 'sell':
            if stop_loss is not None and take_profit is not None:
                stop_loss, take_profit = take_profit, stop_loss
        
        # If we have an open position for this symbol, check if this is a closing signal
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Close position if signal is opposite to our position
            if (position['side'] == 'buy' and signal_type == 'sell') or \
               (position['side'] == 'sell' and signal_type == 'buy'):
                
                # Calculate profit/loss
                quantity = position['quantity']
                entry_price = position['entry_price']
                
                if position['side'] == 'buy':
                    pnl = (price - entry_price) * quantity
                else:  # sell/short position
                    pnl = (entry_price - price) * quantity
                
                # Update capital
                self.current_capital += pnl + (quantity * price)
                
                # Record the trade
                trade = {
                    'symbol': symbol,
                    'entry_time': position['timestamp'],
                    'exit_time': timestamp,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'quantity': quantity,
                    'side': position['side'],
                    'pnl': pnl,
                    'pnl_percent': (pnl / (entry_price * quantity)) * 100,
                    'reason': signal.get('reasons', ['Strategy signal'])[0]
                }
                
                self.trades.append(trade)
                del self.positions[symbol]
                
                return True, trade
            
            # Ignore repeated signals for the same side
            else:
                return False, None
        
        # Open a new position
        else:
            # Calculate position size based on risk
            position_size = self._calculate_position_size(price, risk_per_trade, stop_loss)
            
            # Check if we have enough capital
            if position_size * price > self.current_capital:
                return False, None
                
            # Create position
            position = {
                'symbol': symbol,
                'entry_price': price,
                'quantity': position_size,
                'side': signal_type,
                'timestamp': timestamp,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Allocate capital
            self.current_capital -= position_size * price
            
            # Add position
            self.positions[symbol] = position
            
            return True, position
            
    def check_stops(self, symbol, current_price, timestamp):
        """Check if current price hits any stop loss or take profit levels"""
        if symbol not in self.positions:
            return False, None
            
        position = self.positions[symbol]
        stop_loss = position.get('stop_loss', None)
        take_profit = position.get('take_profit', None)
        
        triggered = False
        exit_type = None
        
        # Check stop loss (for buy positions, price below stop; for sell positions, price above stop)
        if stop_loss is not None:
            if (position['side'] == 'buy' and current_price <= stop_loss) or \
               (position['side'] == 'sell' and current_price >= stop_loss):
                triggered = True
                exit_type = 'stop_loss'
        
        # Check take profit (for buy positions, price above target; for sell positions, price below target)
        if take_profit is not None:
            if (position['side'] == 'buy' and current_price >= take_profit) or \
               (position['side'] == 'sell' and current_price <= take_profit):
                triggered = True
                exit_type = 'take_profit'
                
        if not triggered:
            return False, None
            
        # Execute the exit
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        if position['side'] == 'buy':
            pnl = (current_price - entry_price) * quantity
        else:  # sell/short position
            pnl = (entry_price - current_price) * quantity
        
        # Update capital
        self.current_capital += pnl + (quantity * current_price)
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': current_price,
            'quantity': quantity,
            'side': position['side'],
            'pnl': pnl,
            'pnl_percent': (pnl / (entry_price * quantity)) * 100,
            'reason': f'Hit {exit_type}'
        }
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        return True, trade
        
    def update_equity(self, timestamp, prices):
        """Update equity curve with current positions value"""
        equity = self.current_capital
        
        # Add value of open positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                position_value = position['quantity'] * current_price
                
                # For short positions, the value behaves inversely
                if position['side'] == 'sell':
                    entry_value = position['quantity'] * position['entry_price']
                    current_value = position['quantity'] * current_price
                    position_value = entry_value + (entry_value - current_value)
                    
                equity += position_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
        return equity
        
    def get_metrics(self):
        """Calculate and return performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in self.trades if trade['pnl'] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit/loss metrics
        gross_profit = sum(trade['pnl'] for trade in self.trades if trade['pnl'] > 0)
        gross_loss = sum(trade['pnl'] for trade in self.trades if trade['pnl'] < 0)
        
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Return and drawdown
        equity_values = [point['equity'] for point in self.equity_curve]
        if not equity_values:
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
        total_return = (equity_values[-1] / self.initial_capital - 1) * 100
        
        # Calculate max drawdown
        running_max = 0
        max_drawdown = 0
        
        for equity in equity_values:
            if equity > running_max:
                running_max = equity
            drawdown = (running_max - equity) / running_max * 100
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe ratio (simplified, assuming risk-free rate of 0)
        if len(equity_values) > 1:
            returns = [(equity_values[i] / equity_values[i-1]) - 1 for i in range(1, len(equity_values))]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Average trade metrics
        avg_profit = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
        avg_trade = (gross_profit + gross_loss) / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade
        }

class DataProcessor:
    """Class to download and prepare data for backtesting"""
    
    def __init__(self, exchange_id='binance'):
        """Initialize with exchange"""
        self.exchange_id = exchange_id
        self.exchange = self._initialize_exchange()
        
    def _initialize_exchange(self):
        """Initialize exchange instance"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            return exchange_class({
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"Error initializing exchange: {e}")
            return None
    
    def download_historical_data(self, symbol, timeframe='1h', since=None, limit=None):
        """Download historical OHLCV data"""
        try:
            if since is None:
                # Default to 1 year ago
                since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
                
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Rename to match our strategy functions
            df = df.rename(columns={
                'timestamp': 'Date', 
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close', 
                'volume': 'Volume'
            })
            
            return df
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return pd.DataFrame()
    
    def prepare_data_for_backtest(self, df):
        """Prepare data for backtesting by calculating indicators"""
        try:
            if len(df) < 50:
                return df
                
            # Calculate basic technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            df['RSI'] = calculate_rsi(df['Close'])
            df['ATR'] = calculate_atr(df)
            
            df = calculate_bollinger_bands(df)
            
            # MACD
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            return df
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return df

class StrategyOptimizer:
    """Class to optimize strategy parameters"""
    
    def __init__(self, data, strategy_func, param_grid):
        """
        Initialize optimizer with data and strategy
        
        Args:
            data (pd.DataFrame): Historical price data
            strategy_func (function): Strategy function to optimize
            param_grid (dict): Parameter grid to search
        """
        self.data = data
        self.strategy_func = strategy_func
        self.param_grid = param_grid
        self.results = []
        
    def _generate_param_combinations(self):
        """Generate all parameter combinations from param_grid"""
        import itertools
        
        # Extract parameter names and values
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = [
            {param_names[i]: comb[i] for i in range(len(param_names))}
            for comb in combinations
        ]
        
        return param_combinations
        
    def run_optimization(self, initial_capital=10000, risk_per_trade=1):
        """Run parameter optimization"""
        param_combinations = self._generate_param_combinations()
        
        print(f"Running optimization with {len(param_combinations)} parameter combinations")
        
        for params in tqdm(param_combinations):
            try:
                # Create strategy name from parameters
                strategy_name = self.strategy_func.__name__
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                full_strategy_name = f"{strategy_name}({param_str})"
                
                # Run backtest with these parameters
                backtest = BacktestEngine(initial_capital=initial_capital)
                result = self.backtest_strategy(backtest, params, risk_per_trade)
                
                self.results.append({
                    'strategy_name': full_strategy_name,
                    'parameters': params,
                    'metrics': result
                })
                
            except Exception as e:
                print(f"Error optimizing parameters {params}: {e}")
                
        # Sort results by total_return
        self.results.sort(key=lambda x: x['metrics']['total_return'], reverse=True)
        
        return self.results
    
    def backtest_strategy(self, backtest, params, risk_per_trade=1):
        """Backtest a strategy with specific parameters"""
        try:
            # Apply strategy with parameters
            result_df, signals_df = self.strategy_func(self.data.copy(), **params)
            
            if signals_df.empty:
                return backtest.get_metrics()
                
            # Convert price data to dictionary for equity curve calculation
            prices = {}
            
            # Process each signal
            for _, signal_row in signals_df.iterrows():
                # Get data row for this signal date
                date = signal_row['Date']
                data_row = result_df[result_df['Date'] == date]
                
                if len(data_row) == 0:
                    continue
                    
                # Create signal object
                signal = {
                    'symbol': 'BACKTEST',
                    'signal': signal_row['signal'],  # 'buy' or 'sell'
                    'price': signal_row['price'],
                    'reasons': [signal_row['reason']]
                }
                
                # Calculate stop loss and take profit
                if 'ATR' in data_row.columns:
                    atr = data_row['ATR'].iloc[0]
                    if signal['signal'] == 'buy':
                        signal['stop_loss'] = signal['price'] - (atr * 2)  # 2x ATR
                        signal['take_profit'] = signal['price'] + (atr * 4)  # 4x ATR (2:1 reward/risk)
                    else:  # sell signal
                        signal['stop_loss'] = signal['price'] + (atr * 2)
                        signal['take_profit'] = signal['price'] - (atr * 4)
                
                # Execute signal
                backtest.execute_signal(signal, date, risk_per_trade)
                
                # Update prices for equity calculation
                prices['BACKTEST'] = signal['price']
                backtest.update_equity(date, prices)
                
            # Calculate metrics
            return backtest.get_metrics()
            
        except Exception as e:
            print(f"Error in backtest: {e}")
            return backtest.get_metrics()  # Return empty metrics
    
    def get_best_parameters(self, top_n=5):
        """Get top N parameter sets"""
        return self.results[:top_n]
        
    def plot_optimization_results(self, metric='total_return'):
        """Plot optimization results for a specific metric"""
        import matplotlib.pyplot as plt
        
        if not self.results:
            print("No optimization results to plot")
            return
            
        # Extract data for plotting
        strategy_names = [r['strategy_name'] for r in self.results]
        metric_values = [r['metrics'][metric] for r in self.results]
        
        # Sort by metric
        sorted_indices = np.argsort(metric_values)[::-1]  # Descending
        strategy_names = [strategy_names[i] for i in sorted_indices]
        metric_values = [metric_values[i] for i in sorted_indices]
        
        # Take top 20 for readability
        strategy_names = strategy_names[:20]
        metric_values = metric_values[:20]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(strategy_names)), metric_values)
        plt.xticks(range(len(strategy_names)), strategy_names, rotation=90)
        plt.title(f'Strategy Optimization Results - {metric}')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()

def run_backtest(symbol='BTC/USDT', timeframe='1h', strategy='trend', period=365, initial_capital=10000):
    """Run a full backtest on historical data"""
    print(f"Running backtest for {symbol} on {timeframe} timeframe using {strategy} strategy")
    
    # Download historical data
    data_processor = DataProcessor()
    since = int((datetime.now() - timedelta(days=period)).timestamp() * 1000)
    
    df = data_processor.download_historical_data(symbol, timeframe, since)
    if df.empty:
        print("Failed to download historical data")
        return None
        
    print(f"Downloaded {len(df)} historical data points")
    
    # Prepare data
    df = data_processor.prepare_data_for_backtest(df)
    
    # Select strategy
    if strategy == 'trend':
        strategy_func = trend_trading_strategy
    elif strategy == 'breakout':
        strategy_func = breakout_strategy
    elif strategy == 'range':
        strategy_func = range_trading_strategy
    elif strategy == 'price_action':
        strategy_func = price_action_strategy
    elif strategy == 'day':
        strategy_func = day_trading_strategy
    elif strategy == 'scalping':
        strategy_func = scalping_strategy
    elif strategy == 'swing':
        strategy_func = swing_trading_strategy
    else:
        print(f"Unknown strategy: {strategy}")
        return None
    
    # Run strategy
    result_df, signals_df = strategy_func(df)
    
    if signals_df.empty:
        print("No signals generated")
        return None
    
    print(f"Generated {len(signals_df)} signals")
    
    # Set up backtester
    backtest = BacktestEngine(initial_capital=initial_capital)
    
    # Process signals
    prices = {}
    for _, signal_row in signals_df.iterrows():
        date = signal_row['Date']
        data_row = result_df[result_df['Date'] == date]
        
        if len(data_row) == 0:
            continue
            
        signal = {
            'symbol': symbol,
            'signal': signal_row['signal'], 
            'price': signal_row['price'],
            'reasons': [signal_row['reason']]
        }
        
        # Calculate stop loss and take profit
        if 'ATR' in data_row.columns:
            atr = data_row['ATR'].iloc[0]
            if signal['signal'] == 'buy':
                signal['stop_loss'] = signal['price'] - (atr * 2)
                signal['take_profit'] = signal['price'] + (atr * 4)
            else:
                signal['stop_loss'] = signal['price'] + (atr * 2)
                signal['take_profit'] = signal['price'] - (atr * 4)
        
        # Execute signal
        executed, trade_or_position = backtest.execute_signal(signal, date)
        
        # Update prices for equity
        prices[symbol] = signal['price']
        backtest.update_equity(date, prices)
        
        # Check for stop/target hits after signal
        if executed and trade_or_position and isinstance(trade_or_position, dict) and 'side' in trade_or_position:
            signal_idx = df[df['Date'] == date].index[0] if len(df[df['Date'] == date].index) > 0 else -1
            
            if signal_idx >= 0 and signal_idx < len(df) - 1:
                # Look at future prices to check for stop/target hits
                for future_idx in range(signal_idx + 1, len(df)):
                    future_row = df.iloc[future_idx]
                    future_date = future_row['Date']
                    
                    # Check if price hit stop or target
                    triggered, _ = backtest.check_stops(symbol, future_row['Close'], future_date)
                    
                    if triggered:
                        # Update prices and equity
                        prices[symbol] = future_row['Close']
                        backtest.update_equity(future_date, prices)
                        break
    
    # Calculate metrics
    metrics = backtest.get_metrics()
    
    # Print results
    print("\n--- Backtest Results ---")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot equity curve
    if backtest.equity_curve:
        timestamps = [point['timestamp'] for point in backtest.equity_curve]
        equity = [point['equity'] for point in backtest.equity_curve]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, equity)
        plt.title(f'Equity Curve - {strategy} Strategy on {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Return backtest object for further analysis
    return {
        'backtest': backtest,
        'metrics': metrics,
        'data': df,
        'signals': signals_df
    }

def optimize_strategy_params(symbol='BTC/USDT', timeframe='1h', strategy='trend', period=365):
    """Optimize strategy parameters"""
    print(f"Optimizing parameters for {strategy} strategy on {symbol}")
    
    # Download historical data
    data_processor = DataProcessor()
    since = int((datetime.now() - timedelta(days=period)).timestamp() * 1000)
    
    df = data_processor.download_historical_data(symbol, timeframe, since)
    if df.empty:
        print("Failed to download historical data")
        return None
    
    # Prepare data
    df = data_processor.prepare_data_for_backtest(df)
    
    # Define parameter grid based on strategy
    param_grid = {}
    strategy_func = None
    
    if strategy == 'trend':
        strategy_func = trend_trading_strategy
        param_grid = {
            'lookback': [20, 50, 100],
            'ma_short': [5, 10, 20],
            'ma_long': [20, 50, 100]
        }
    elif strategy == 'breakout':
        strategy_func = breakout_strategy
        param_grid = {
            'lookback': [10, 20, 30, 50]
        }
    elif strategy == 'range':
        strategy_func = range_trading_strategy
        param_grid = {
            'lookback': [10, 20, 30],
            'num_std': [1.5, 2.0, 2.5]
        }
    else:
        print(f"Optimization not supported for strategy: {strategy}")
        return None
    
    # Run optimization
    optimizer = StrategyOptimizer(df, strategy_func, param_grid)
    results = optimizer.run_optimization()
    
    # Print results
    print("\n--- Optimization Results ---")
    for i, result in enumerate(optimizer.get_best_parameters()):
        print(f"\n{i+1}. {result['strategy_name']}")
        metrics = result['metrics']
        print(f"   Total Return: {metrics['total_return']:.2f}%")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot results
    optimizer.plot_optimization_results()
    
    return optimizer

def run_multi_pair_backtest(symbols, timeframe='1h', strategy='trend', period=365):
    """Run backtest on multiple pairs to evaluate strategy"""
    results = {}
    
    for symbol in symbols:
        print(f"\nBacktesting {symbol}...")
        result = run_backtest(symbol, timeframe, strategy, period)
        if result:
            results[symbol] = result
    
    # Compare performance across pairs
    if results:
        print("\n--- Multi-Pair Comparison ---")
        comparison = []
        
        for symbol, result in results.items():
            metrics = result['metrics']
            comparison.append({
                'symbol': symbol,
                'return': metrics['total_return'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown': metrics['max_drawdown']
            })
        
        # Convert to DataFrame for easier display
        df_comparison = pd.DataFrame(comparison)
        print(df_comparison)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        plt.bar(df_comparison['symbol'], df_comparison['return'])
        plt.title(f'Return Comparison - {strategy} Strategy')
        plt.ylabel('Return (%)')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
    
    return results

if __name__ == "__main__":
    # Example usage
    backtest_result = run_backtest(symbol='BTC/USDT', strategy='trend')
    
    # To run parameter optimization:
    # optimize_strategy_params(symbol='BTC/USDT', strategy='trend')
    
    # To run multi-pair backtest:
    # run_multi_pair_backtest(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], strategy='trend')