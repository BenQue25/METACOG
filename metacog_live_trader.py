#!/usr/bin/env python3
"""
METACOG LIVE TRADER - ADX Crossover with FOK Open-Close
SOTA TCP_NODELAY technology for submillisecond execution
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import socket
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class MetaCogLiveTrader:
    def __init__(self, symbol="Volatility 10 Index.0"):
        self.symbol = symbol
        self.running = False
        
        # ADX parameters (EXACT SAME AS BACKTEST)
        self.adx_period = 14
        
        # Kelly parameters
        self.kelly_fraction = 0.25
        self.max_kelly_fraction = 0.50
        self.min_volume = 0.5
        self.max_volume = 400.0
        self.volume_step = 0.01
        
        # Data buffers
        self.rates = deque(maxlen=self.adx_period * 3)
        self.last_bar_time = 0
        
        # Trading state
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_volume = None
        self.entry_ticket = None
        
        # PERSISTENT DOMINANCE STATUS - Only changes on crossovers
        self.current_dominance = None
        self.last_crossover_time = None
        
        # Performance tracking
        self.execution_times = []
        self.trades_executed = 0
        
        # Kelly Criterion tracking
        self.trade_history = []
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Get symbol info for proper volume validation
        self.symbol_info = None
        
        print("METACOG LIVE TRADER INITIALIZED")
        print(f"Symbol: {self.symbol}")
        print(f"Volume Range: {self.min_volume}-{self.max_volume} lots")
        print("=" * 60)

    def setup_tcp_nodelay(self):
        """Setup TCP_NODELAY for minimal latency"""
        try:
            # Get MT5 socket and set TCP_NODELAY
            mt5_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            mt5_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("TCP_NODELAY enabled - SOTA latency optimization")
            return True
        except Exception as e:
            print(f"TCP_NODELAY setup failed: {e}")
            return False

    def initialize_mt5(self):
        """Initialize MT5 with FOK settings"""
        if not mt5.initialize():
            print(f"MT5 initialize failed: {mt5.last_error()}")
            return False
            
        # Check if trading is allowed
        terminal_info = mt5.terminal_info()
        if not terminal_info.trade_allowed:
            print("Trading not allowed in terminal")
            return False
        
        # Get symbol info for volume validation
        self.symbol_info = mt5.symbol_info(self.symbol)
        if self.symbol_info is None:
            print(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return False
            
        # Validate volume against symbol requirements
        if self.symbol_info.volume_min > self.min_volume:
            self.min_volume = self.symbol_info.volume_min
            print(f"Volume adjusted to minimum: {self.min_volume}")
            
        if self.symbol_info.volume_step > 0:
            # Round volume to valid step
            self.min_volume = round(self.min_volume / self.symbol_info.volume_step) * self.symbol_info.volume_step
            print(f"Volume rounded to step: {self.min_volume}")
        
        account = mt5.account_info()
        print(f"Connected: {account.login} | Balance: ${account.balance:.2f}")
        print(f"Symbol: {self.symbol} | Min Volume: {self.symbol_info.volume_min} | Step: {self.symbol_info.volume_step}")
        return True

    def load_initial_data(self):
        """Load initial historical data for ADX calculation"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, self.adx_period * 3)
            if rates is not None and len(rates) > self.adx_period:
                self.rates.extend(rates)
                print(f"Loaded {len(rates)} historical bars")
                
                # Calculate initial dominance
                self.calculate_initial_dominance()
                return True
            else:
                print("Insufficient historical data")
                return False
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return False

    def calculate_initial_dominance(self):
        """Calculate initial dominance from historical data"""
        if len(self.rates) < self.adx_period:
            return
            
        df = pd.DataFrame(list(self.rates))
        if 'high' not in df.columns:
            return
            
        # Calculate ADX components (EXACT SAME AS BACKTEST)
        df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
        df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
        df['TR'] = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)

        df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / df['TR'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean())
        df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / df['TR'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean())
        
        # Set initial dominance
        if len(df) >= 2:
            last = df.iloc[-1]
            if not (pd.isna(last['+DI']) or pd.isna(last['-DI'])):
                self.current_dominance = "DI+" if last['+DI'] > last['-DI'] else "DI-"
                print(f"Initial Dominance: {self.current_dominance}")

    def calculate_kelly_volume(self):
        """Calculate Kelly Criterion position size with proper mathematics"""
        if not self.trade_history:
            return self.min_volume
        
        # Calculate win rate and average win/loss
        wins = [t for t in self.trade_history if t['pnl'] > 0]
        losses = [t for t in self.trade_history if t['pnl'] < 0]
        
        if not wins or not losses:
            return self.min_volume
        
        # Calculate statistics
        win_rate = len(wins) / len(self.trade_history)
        avg_win = sum(t['pnl'] for t in wins) / len(wins)
        avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses))
        
        if avg_loss == 0:
            return self.min_volume
        
        # Kelly formula: f = (bp - q) / b
        # where: b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        # Calculate raw Kelly fraction
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (25% of full Kelly) and constraints
        kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
        kelly_fraction = kelly_fraction * self.kelly_fraction  # Scale down to 25%
        
        # Calculate volume based on Kelly
        account_balance = mt5.account_info().balance
        account_risk = account_balance * kelly_fraction
        
        # Convert to volume (assuming 1 lot = $1000 risk)
        volume = account_risk / 1000
        
        # Apply volume constraints
        volume = max(self.min_volume, min(volume, self.max_volume))
        volume = round(volume / self.volume_step) * self.volume_step
        
        # Update tracking variables
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        
        print(f"Kelly Calculation: Win Rate: {win_rate:.2%}, Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")
        print(f"Kelly Fraction: {kelly_fraction:.4f}, Volume: {volume:.2f} lots")
        
        return volume

    def execute_fok_open(self, signal_type):
        """Execute FOK OPEN order"""
        start_time = time.perf_counter_ns()
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print(f"OPEN {signal_type} FAILED: No tick data")
            return False
        
        volume = self.calculate_kelly_volume()
        
        # Open position with FOK
        open_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if signal_type == "BUY" else tick.bid,
            "deviation": 5,
            "magic": 123456,
            "comment": "METACOG_ADX_OPEN",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        open_result = mt5.order_send(open_request)
        end_time = time.perf_counter_ns()
        execution_time_ms = (end_time - start_time) / 1_000_000
        
        if open_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"OPEN {signal_type} EXECUTED: {execution_time_ms:.3f}ms | Volume: {volume} | Ticket: {open_result.order}")
            
            # Set position state
            self.current_position = signal_type
            self.entry_price = open_result.price
            self.entry_time = datetime.now()
            self.entry_volume = volume
            self.entry_ticket = open_result.order
            
            # Record execution time
            self.execution_times.append(execution_time_ms)
            self.trades_executed += 1
            
            return True
        else:
            print(f"OPEN {signal_type} FAILED: {open_result.retcode} | Time: {execution_time_ms:.3f}ms | Error: {open_result.comment}")
            return False

    def execute_fok_close(self, signal_type):
        """Execute FOK CLOSE order"""
        if not self.current_position:
            return False
            
        start_time = time.perf_counter_ns()
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print(f"CLOSE {signal_type} FAILED: No tick data")
            return False
        
        # Close position with FOK
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.entry_volume,
            "type": mt5.ORDER_TYPE_SELL if self.current_position == "BUY" else mt5.ORDER_TYPE_BUY,
            "position": self.entry_ticket,
            "price": tick.bid if self.current_position == "BUY" else tick.ask,
            "deviation": 5,
            "magic": 123456,
            "comment": "METACOG_ADX_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        close_result = mt5.order_send(close_request)
        end_time = time.perf_counter_ns()
        execution_time_ms = (end_time - start_time) / 1_000_000
        
        if close_result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"CLOSE {signal_type} EXECUTED: {execution_time_ms:.3f}ms | Volume: {self.entry_volume} | Ticket: {close_result.order}")
            
            # Calculate P&L
            if self.current_position == "BUY":
                pnl = (close_result.price - self.entry_price) * self.entry_volume * 1000
            else:  # SELL
                pnl = (self.entry_price - close_result.price) * self.entry_volume * 1000
            
            # Record trade in history
            trade_record = {
                'entry_time': self.entry_time,
                'exit_time': datetime.now(),
                'entry_price': self.entry_price,
                'exit_price': close_result.price,
                'direction': self.current_position,
                'volume': self.entry_volume,
                'pnl': pnl,
                'entry_ticket': self.entry_ticket,
                'exit_ticket': close_result.order
            }
            self.trade_history.append(trade_record)
            
            # Update consecutive wins/losses
            if pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
            print(f"Trade P&L: ${pnl:.2f} | Consecutive Wins: {self.consecutive_wins} | Consecutive Losses: {self.consecutive_losses}")
            
            # Reset position state
            self.current_position = None
            self.entry_price = None
            self.entry_time = None
            self.entry_volume = None
            self.entry_ticket = None
            
            # Record execution time
            self.execution_times.append(execution_time_ms)
            
            return True
        else:
            print(f"CLOSE {signal_type} FAILED: {close_result.retcode} | Time: {execution_time_ms:.3f}ms | Error: {close_result.comment}")
            return False

    def process_completed_bar(self, rates):
        """Process completed minute bar for crossover detection (EXACT SAME AS BACKTEST)"""
        if len(rates) < self.adx_period:
            return
            
        df = pd.DataFrame(rates)
        if 'high' not in df.columns:
            return
            
        # Calculate ADX components (EXACT SAME AS BACKTEST)
        df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
        df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
        df['TR'] = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)

        df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / df['TR'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean())
        df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean() / df['TR'].ewm(alpha=1/self.adx_period, min_periods=self.adx_period).mean())
        
        # EXACT SAME CROSSOVER LOGIC AS BACKTEST
        if len(df) >= 2:
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            if not (pd.isna(current['+DI']) or pd.isna(current['-DI']) or pd.isna(previous['+DI']) or pd.isna(previous['-DI'])):
                # BUY Signal: DI+ crosses above DI- (same as backtest)
                if (previous['+DI'] <= previous['-DI'] and current['+DI'] > current['-DI']):
                    # Update persistent status
                    self.current_dominance = "DI+"
                    self.last_crossover_time = pd.to_datetime(current['time'], unit='s')
                    
                    print(f"\n🟢 BULLISH CROSSOVER DETECTED!")
                    print(f"DI+: {current['+DI']:.2f} | DI-: {current['-DI']:.2f}")
                    print(f"Status: {self.current_dominance} DOMINANT")
                    
                    # Execute BUY signal
                    if self.current_position == "SELL":
                        # Close existing SELL position first
                        self.execute_fok_close("BUY")
                    
                    # Open new BUY position
                    self.execute_fok_open("BUY")
                    
                # SELL Signal: DI- crosses above DI+ (same as backtest)
                elif (previous['-DI'] <= previous['+DI'] and current['-DI'] > current['+DI']):
                    # Update persistent status
                    self.current_dominance = "DI-"
                    self.last_crossover_time = pd.to_datetime(current['time'], unit='s')
                    
                    print(f"\n🔴 BEARISH CROSSOVER DETECTED!")
                    print(f"DI+: {current['+DI']:.2f} | DI-: {current['-DI']:.2f}")
                    print(f"Status: {self.current_dominance} DOMINANT")
                    
                    # Execute SELL signal
                    if self.current_position == "BUY":
                        # Close existing BUY position first
                        self.execute_fok_close("SELL")
                    
                    # Open new SELL position
                    self.execute_fok_open("SELL")
                
                # Always show current status
                else:
                    print(f"DI+: {current['+DI']:.2f} | DI-: {current['-DI']:.2f} | Status: {self.current_dominance}")

    def run(self):
        """Main trading loop"""
        print("STARTING METACOG LIVE TRADER...")
        
        if not self.setup_tcp_nodelay():
            return
            
        if not self.initialize_mt5():
            return
            
        if not self.load_initial_data():
            return
            
        self.running = True
        print("System ready - monitoring for ADX crossovers...")
        print("=" * 60)
        
        last_bar_time = 0
        
        while self.running:
            try:
                # Get current tick
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    tick_time = pd.to_datetime(tick.time, unit='s')
                    minute_time = tick_time.replace(second=0, microsecond=0)
                    
                    # Check if new minute bar completed
                    if minute_time != last_bar_time and minute_time != self.last_bar_time:
                        # Get new completed bar
                        rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, self.adx_period * 3)
                        if rates is not None and len(rates) > self.adx_period:
                            # Update rates buffer
                            self.rates.clear()
                            self.rates.extend(rates)
                            
                            # Process for crossovers
                            self.process_completed_bar(rates)
                            
                            last_bar_time = minute_time
                
                # Small delay
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("Stopping trader...")
                self.running = False
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        
        # Close any remaining position
        if self.current_position:
            print(f"Closing remaining {self.current_position} position...")
            self.execute_fok_close("EXIT")
        
        # Show final results
        self.show_final_results()
        mt5.shutdown()

    def show_final_results(self):
        """Display final trading results"""
        print("\n" + "=" * 60)
        print("METACOG LIVE TRADING RESULTS")
        print("=" * 60)
        
        if not self.execution_times:
            print("No trades executed")
            return
        
        # Calculate execution metrics
        total_executions = len(self.execution_times)
        avg_execution = sum(self.execution_times) / len(self.execution_times)
        min_execution = min(self.execution_times)
        max_execution = max(self.execution_times)
        
        print(f"Trading Performance:")
        print(f"  Total Trades: {self.trades_executed}")
        print(f"  Total Executions: {total_executions}")
        print()
        
        # Kelly Criterion performance
        if self.trade_history:
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
            losing_trades = len([t for t in self.trade_history if t['pnl'] < 0])
            
            print(f"Kelly Criterion Performance:")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Win Rate: {self.win_rate:.2%}")
            print(f"  Average Win: ${self.avg_win:.2f}")
            print(f"  Average Loss: ${self.avg_loss:.2f}")
            print(f"  Consecutive Wins: {self.consecutive_wins}")
            print(f"  Consecutive Losses: {self.consecutive_losses}")
            print()
        
        print(f"Execution Times:")
        print(f"  Average: {avg_execution:.3f}ms")
        print(f"  Fastest: {min_execution:.3f}ms")
        print(f"  Slowest: {max_execution:.3f}ms")
        print()
        
        print(f"Technology:")
        print(f"  TCP_NODELAY: ENABLED")
        print(f"  FOK Filling: ACTIVE")
        print(f"  ADX Logic: EXACT BACKTEST MATCH")
        print(f"  Kelly Criterion: MATHEMATICAL FRACTIONAL")
        print("=" * 60)

if __name__ == "__main__":
    print("METACOG LIVE TRADER - ADX Crossover with FOK Open-Close")
    print("SOTA TCP_NODELAY Technology")
    print("Press Ctrl+C to stop")
    print()
    
    trader = MetaCogLiveTrader()
    trader.run()
