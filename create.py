"""
Trading Signal Bot for Telegram

This bot monitors crypto markets in real-time, analyzes trading pairs using technical analysis,
and sends precise trading signals to Telegram when conditions are met.
"""

import os
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# Data handling imports
import pandas as pd
import numpy as np
import ccxt  # For exchange API integration
import talib  # Technical indicators

# Telegram integration
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    ContextTypes, MessageHandler, filters
)

# Import strategies from bot.py
from bot import (
    calculate_rsi, calculate_atr, calculate_bollinger_bands,
    trend_trading_strategy, breakout_strategy, range_trading_strategy,
    price_action_strategy, day_trading_strategy, scalping_strategy,
    swing_trading_strategy, carry_trade_strategy, position_trading_strategy,
    news_trading_strategy
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'telegram_token': '',  # Fill in your Telegram bot token
    'admin_user_id': 0,    # Fill in your Telegram user ID 
    'exchange': 'binance',
    'timeframe': '1h',     # Default timeframe
    'default_pairs': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    'max_history': 500,    # Number of candles to fetch
    'risk_percent': 1,     # Default risk per trade (1%)
    'update_interval': 300, # Update interval in seconds (5 minutes)
    'strategies': {
        'enabled': ['trend', 'breakout', 'range'], # Default strategies
        'confirmations_required': 2,  # Number of strategies that must agree
    },
    'indicators': {
        'rsi_period': 14,
        'bb_period': 20,
        'bb_std': 2,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
    },
    'risk_management': {
        'stop_loss_atr_multiplier': 2,  # Stop loss at 2x ATR
        'take_profit_risk_multiple': 2, # TP at 2x risk
        'max_trades_per_day': 5,
    }
}

# Cache for storing market data
MARKET_DATA = {}
ACTIVE_SIGNALS = {}
BOT_RUNNING = False
USER_SETTINGS = {}

class TradingBot:
    def __init__(self, config: Dict):
        """Initialize the trading bot with configuration"""
        self.config = config
        self.exchange = self._initialize_exchange()
        self.signals = []
        self.running = False
        
    def _initialize_exchange(self):
        """Initialize the exchange API connection"""
        exchange_id = self.config['exchange']
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,  # Required by CCXT
            })
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            return None
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            # Fetch data from exchange
            ohlcv = await asyncio.to_thread(
                self.exchange.fetch_ohlcv, 
                symbol, 
                timeframe, 
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Rename columns to match our strategy functions
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
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if len(df) < 50:  # Minimum data required
            return df
            
        try:
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # RSI
            df['RSI'] = calculate_rsi(df['Close'], period=self.config['indicators']['rsi_period'])
            
            # Bollinger Bands
            df = calculate_bollinger_bands(
                df, 
                window=self.config['indicators']['bb_period'],
                num_std=self.config['indicators']['bb_std']
            )
            
            # MACD
            df['EMA_12'] = df['Close'].ewm(span=self.config['indicators']['macd_fast']).mean()
            df['EMA_26'] = df['Close'].ewm(span=self.config['indicators']['macd_slow']).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=self.config['indicators']['macd_signal']).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # ATR for stop loss calculation
            df['ATR'] = calculate_atr(df, period=14)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def analyze_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze chart patterns"""
        patterns = {}
        
        if len(df) < 20:  # Need enough data
            return patterns
        
        try:
            # Calculate candlestick body and shadows
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
            df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
            df['candle_range'] = df['High'] - df['Low']
            
            # Doji pattern (small body relative to range)
            last_row = df.iloc[-1]
            if last_row['body_size'] < 0.1 * last_row['candle_range']:
                patterns['doji'] = True
                
            # Hammer pattern
            if (last_row['body_size'] < 0.3 * last_row['candle_range'] and
                last_row['lower_shadow'] > 2 * last_row['body_size'] and
                last_row['upper_shadow'] < 0.5 * last_row['body_size']):
                patterns['hammer'] = True
            
            # Shooting star pattern
            if (last_row['body_size'] < 0.3 * last_row['candle_range'] and
                last_row['upper_shadow'] > 2 * last_row['body_size'] and
                last_row['lower_shadow'] < 0.5 * last_row['body_size']):
                patterns['shooting_star'] = True
                
            # Engulfing patterns
            prev_row = df.iloc[-2]
            # Bullish engulfing
            if (prev_row['Close'] < prev_row['Open'] and  # Previous red candle
                last_row['Close'] > last_row['Open'] and  # Current green candle
                last_row['Open'] < prev_row['Close'] and  # Current open below prev close
                last_row['Close'] > prev_row['Open']):    # Current close above prev open
                patterns['bullish_engulfing'] = True
                
            # Bearish engulfing
            if (prev_row['Close'] > prev_row['Open'] and  # Previous green candle
                last_row['Close'] < last_row['Open'] and  # Current red candle
                last_row['Open'] > prev_row['Close'] and  # Current open above prev close
                last_row['Close'] < prev_row['Open']):    # Current close below prev open
                patterns['bearish_engulfing'] = True
                
            return patterns
        
        except Exception as e:
            logger.error(f"Error analyzing chart patterns: {e}")
            return {}
    
    def apply_strategies(self, df: pd.DataFrame) -> List[Dict]:
        """Apply all enabled strategies to the data"""
        strategy_results = []
        
        if len(df) < 50:  # Need enough data
            return strategy_results
            
        enabled_strategies = self.config['strategies']['enabled']
        
        try:
            # Run each enabled strategy
            for strategy in enabled_strategies:
                result = None
                
                if strategy == 'trend':
                    result, signals_df = trend_trading_strategy(df)
                elif strategy == 'breakout':
                    result, signals_df = breakout_strategy(df)
                elif strategy == 'range':
                    result, signals_df = range_trading_strategy(df)
                elif strategy == 'price_action':
                    result, signals_df = price_action_strategy(df)
                elif strategy == 'day':
                    result, signals_df = day_trading_strategy(df)
                elif strategy == 'scalping':
                    result, signals_df = scalping_strategy(df)
                elif strategy == 'swing':
                    result, signals_df = swing_trading_strategy(df)
                
                # If we have a signal from the latest data point
                if not signals_df.empty and signals_df.iloc[-1]['Date'] >= df.iloc[-5]['Date']:
                    latest_signal = signals_df.iloc[-1]
                    strategy_results.append({
                        'strategy': strategy,
                        'signal': latest_signal['signal'],
                        'price': latest_signal['price'],
                        'reason': latest_signal['reason']
                    })
            
            # Check for chart patterns
            patterns = self.analyze_chart_patterns(df)
            if patterns:
                signal = None
                reason = []
                
                # Determine signal based on patterns
                if 'hammer' in patterns or 'bullish_engulfing' in patterns:
                    signal = 'buy'
                    if 'hammer' in patterns:
                        reason.append('Hammer pattern')
                    if 'bullish_engulfing' in patterns:
                        reason.append('Bullish engulfing')
                
                elif 'shooting_star' in patterns or 'bearish_engulfing' in patterns:
                    signal = 'sell'
                    if 'shooting_star' in patterns:
                        reason.append('Shooting star pattern')
                    if 'bearish_engulfing' in patterns:
                        reason.append('Bearish engulfing')
                
                if signal:
                    strategy_results.append({
                        'strategy': 'patterns',
                        'signal': signal,
                        'price': df.iloc[-1]['Close'],
                        'reason': ', '.join(reason)
                    })
            
            return strategy_results
        
        except Exception as e:
            logger.error(f"Error applying strategies: {e}")
            return []
    
    def make_decision(self, strategy_results: List[Dict]) -> Optional[Dict]:
        """Make trading decision based on strategy results"""
        if not strategy_results:
            return None
        
        # Count buy and sell signals
        buy_count = sum(1 for r in strategy_results if r['signal'] == 'buy')
        sell_count = sum(1 for r in strategy_results if r['signal'] == 'sell')
        
        # Get confirmations required
        required_confirmations = self.config['strategies']['confirmations_required']
        
        # Check if we have enough confirmations
        if buy_count >= required_confirmations:
            # Find which strategies confirmed buy
            confirming_strategies = [r['strategy'] for r in strategy_results if r['signal'] == 'buy']
            reasons = [r['reason'] for r in strategy_results if r['signal'] == 'buy']
            
            return {
                'signal': 'buy',
                'price': strategy_results[0]['price'],  # Use the price from first strategy
                'strategies': confirming_strategies,
                'reasons': reasons
            }
        
        elif sell_count >= required_confirmations:
            # Find which strategies confirmed sell
            confirming_strategies = [r['strategy'] for r in strategy_results if r['signal'] == 'sell']
            reasons = [r['reason'] for r in strategy_results if r['signal'] == 'sell']
            
            return {
                'signal': 'sell',
                'price': strategy_results[0]['price'],  # Use the price from first strategy
                'strategies': confirming_strategies,
                'reasons': reasons
            }
        
        return None
    
    def calculate_risk_management(self, df: pd.DataFrame, signal: Dict) -> Dict:
        """Calculate stop loss and take profit levels"""
        if len(df) < 20:
            return signal
            
        try:
            # Get ATR for stop loss calculation
            atr = df['ATR'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            # Calculate stop loss and take profit based on ATR
            atr_multiplier = self.config['risk_management']['stop_loss_atr_multiplier']
            risk_multiplier = self.config['risk_management']['take_profit_risk_multiple']
            
            if signal['signal'] == 'buy':
                stop_loss = current_price - (atr * atr_multiplier)
                risk = current_price - stop_loss
                take_profit = current_price + (risk * risk_multiplier)
            else:  # sell
                stop_loss = current_price + (atr * atr_multiplier)
                risk = stop_loss - current_price
                take_profit = current_price - (risk * risk_multiplier)
                
            # Add risk management to signal
            signal['stop_loss'] = round(stop_loss, 8)
            signal['take_profit'] = round(take_profit, 8)
            signal['risk_reward'] = round(risk_multiplier, 2)
            
            return signal
        
        except Exception as e:
            logger.error(f"Error calculating risk management: {e}")
            return signal
    
    async def analyze_pair(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze a trading pair and return a signal if found"""
        try:
            # Fetch data
            df = await self.fetch_ohlcv(symbol, timeframe, self.config['max_history'])
            
            if df.empty:
                return None
                
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Apply strategies
            strategy_results = self.apply_strategies(df)
            
            # Make decision
            decision = self.make_decision(strategy_results)
            
            if decision:
                # Add symbol and timestamp
                decision['symbol'] = symbol
                decision['timeframe'] = timeframe
                decision['timestamp'] = datetime.now().isoformat()
                
                # Add risk management
                decision = self.calculate_risk_management(df, decision)
                
                return decision
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def monitor_markets(self):
        """Monitor markets for trading signals"""
        self.running = True
        
        logger.info("Starting market monitoring...")
        
        while self.running:
            try:
                for symbol in self.config['default_pairs']:
                    logger.info(f"Analyzing {symbol}")
                    
                    # Analyze pair
                    signal = await self.analyze_pair(symbol, self.config['timeframe'])
                    
                    if signal:
                        # Add to signals list
                        self.signals.append(signal)
                        
                        # Store in global active signals
                        global ACTIVE_SIGNALS
                        signal_key = f"{symbol}_{signal['timestamp']}"
                        ACTIVE_SIGNALS[signal_key] = signal
                        
                        logger.info(f"Signal generated: {signal}")
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                
                # Wait for next update interval
                await asyncio.sleep(self.config['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    def stop(self):
        """Stop the bot"""
        self.running = False

#------------------------------------------------------------------------------
# Telegram Bot Functions
#------------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user_id = update.effective_user.id
    user = update.effective_user
    
    welcome_message = (
        f"Welcome {user.first_name}! 🤖\n\n"
        f"I am your Trading Signal Bot. I monitor crypto markets and provide trading signals.\n\n"
        f"Use /help to see available commands."
    )
    
    # Set up user settings if not exists
    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = {
            'pairs': CONFIG['default_pairs'].copy(),
            'timeframes': [CONFIG['timeframe']],
            'notifications': True,
        }
    
    keyboard = [
        [
            InlineKeyboardButton("Active Signals", callback_data="signals"),
            InlineKeyboardButton("Settings", callback_data="settings"),
        ],
        [
            InlineKeyboardButton("Start Bot", callback_data="start_bot"),
            InlineKeyboardButton("Stop Bot", callback_data="stop_bot"),
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "Available commands:\n\n"
        "/start - Start the bot\n"
        "/status - Show bot status\n"
        "/signals - Show active signals\n"
        "/settings - Modify your settings\n"
        "/pairs - Show/edit watched trading pairs\n"
        "/timeframes - Show/edit chart timeframes\n"
        "/start_bot - Start monitoring markets\n"
        "/stop_bot - Stop monitoring markets\n"
    )
    await update.message.reply_text(help_text)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot status"""
    global BOT_RUNNING
    
    status_message = (
        f"Bot Status: {'Running ✅' if BOT_RUNNING else 'Stopped ❌'}\n\n"
        f"Monitoring {len(CONFIG['default_pairs'])} trading pairs\n"
        f"Timeframe: {CONFIG['timeframe']}\n"
        f"Update interval: {CONFIG['update_interval']} seconds\n"
        f"Active signals: {len(ACTIVE_SIGNALS)}\n"
    )
    
    await update.message.reply_text(status_message)

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show active signals"""
    if not ACTIVE_SIGNALS:
        await update.message.reply_text("No active signals at the moment.")
        return
    
    signals_text = "Active Signals:\n\n"
    
    for key, signal in ACTIVE_SIGNALS.items():
        time_str = signal['timestamp'].split('T')[1][:5]  # Extract HH:MM
        signals_text += (
            f"📊 {signal['symbol']} ({signal['timeframe']}) at {time_str}\n"
            f"Signal: {'🟢 BUY' if signal['signal'] == 'buy' else '🔴 SELL'}\n"
            f"Price: {signal['price']}\n"
            f"Stop Loss: {signal['stop_loss']}\n"
            f"Take Profit: {signal['take_profit']}\n"
            f"R:R: 1:{signal['risk_reward']}\n"
            f"Reason: {', '.join(signal['reasons'][:1])}\n\n"
        )
    
    await update.message.reply_text(signals_text)

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show and edit settings"""
    user_id = update.effective_user.id
    
    # Ensure user settings exist
    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = {
            'pairs': CONFIG['default_pairs'].copy(),
            'timeframes': [CONFIG['timeframe']],
            'notifications': True,
        }
    
    settings = USER_SETTINGS[user_id]
    
    settings_text = (
        f"Your Current Settings:\n\n"
        f"Trading Pairs: {', '.join(settings['pairs'])}\n"
        f"Timeframes: {', '.join(settings['timeframes'])}\n"
        f"Notifications: {'Enabled ✅' if settings['notifications'] else 'Disabled ❌'}\n"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("Edit Pairs", callback_data="edit_pairs"),
            InlineKeyboardButton("Edit Timeframes", callback_data="edit_timeframes"),
        ],
        [
            InlineKeyboardButton(
                "Disable Notifications" if settings['notifications'] else "Enable Notifications", 
                callback_data="toggle_notifications"
            ),
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(settings_text, reply_markup=reply_markup)

async def start_bot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start the trading bot"""
    global BOT_RUNNING
    
    if BOT_RUNNING:
        await update.message.reply_text("Bot is already running!")
        return
    
    BOT_RUNNING = True
    
    # Start the bot in the background
    bot = TradingBot(CONFIG)
    context.application.bot_instance = bot
    
    # Run the monitoring task
    context.application.create_task(bot.monitor_markets())
    
    await update.message.reply_text(
        "Trading bot started! I'll monitor markets and send you signals."
    )

async def stop_bot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop the trading bot"""
    global BOT_RUNNING
    
    if not BOT_RUNNING:
        await update.message.reply_text("Bot is not running!")
        return
    
    BOT_RUNNING = False
    
    # Stop the bot if it exists
    if hasattr(context.application, 'bot_instance'):
        context.application.bot_instance.stop()
        
    await update.message.reply_text("Trading bot stopped!")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks"""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    callback_data = query.data
    
    if callback_data == "signals":
        if not ACTIVE_SIGNALS:
            await query.edit_message_text("No active signals at the moment.")
        else:
            signals_text = "Active Signals:\n\n"
            
            for key, signal in list(ACTIVE_SIGNALS.items())[:5]:  # Show at most 5
                time_str = signal['timestamp'].split('T')[1][:5]  # Extract HH:MM
                signals_text += (
                    f"📊 {signal['symbol']} ({signal['timeframe']}) at {time_str}\n"
                    f"Signal: {'🟢 BUY' if signal['signal'] == 'buy' else '🔴 SELL'}\n"
                    f"Price: {signal['price']}\n"
                    f"SL: {signal['stop_loss']} | TP: {signal['take_profit']}\n\n"
                )
            
            if len(ACTIVE_SIGNALS) > 5:
                signals_text += f"...and {len(ACTIVE_SIGNALS) - 5} more signals."
                
            await query.edit_message_text(signals_text)
    
    elif callback_data == "settings":
        await settings_command(update, context)
    
    elif callback_data == "start_bot":
        await start_bot_command(update, context)
    
    elif callback_data == "stop_bot":
        await stop_bot_command(update, context)
    
    elif callback_data == "toggle_notifications":
        if user_id in USER_SETTINGS:
            USER_SETTINGS[user_id]['notifications'] = not USER_SETTINGS[user_id]['notifications']
            status = "enabled" if USER_SETTINGS[user_id]['notifications'] else "disabled"
            await query.edit_message_text(f"Notifications {status}!")
    
    # Other callbacks can be handled here

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates."""
    logger.error("Exception while handling an update:", exc_info=context.error)

def main() -> None:
    """Start the bot."""
    # Create the Application 
    if not CONFIG['telegram_token']:
        logger.error("Telegram token not configured!")
        return
        
    application = Application.builder().token(CONFIG['telegram_token']).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("start_bot", start_bot_command))
    application.add_handler(CommandHandler("stop_bot", stop_bot_command))
    
    # Callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Error handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    logger.info("Starting Telegram Bot")
    application.run_polling()

if __name__ == "__main__":
    main()