#!/usr/bin/env python3
"""
ADX Live Trader - Fixed Version
Fixed ADX calculation issues and proper signal detection
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PositionModerator:
    """Lightweight moderator to prevent missed signals and duplicate orders"""
    def __init__(self):
        self.last_processed_bar_time = None
        self.last_executed_signal = None  # 'BUY' or 'SELL'

    def approve(self, signal: str, bar_time, current_position) -> (bool, str):
        if self.last_processed_bar_time == bar_time and self.last_executed_signal == signal:
            return False, "duplicate-signal-same-bar"
        if current_position and current_position.get('signal') == signal:
            return False, "duplicate-consecutive-order"
        if not current_position and self.last_executed_signal == signal:
            return False, "duplicate-when-flat"
        return True, "ok"

    def mark_processed(self, bar_time):
        self.last_processed_bar_time = bar_time

    def on_order_executed(self, signal: str):
        self.last_executed_signal = signal

class ADXLiveTraderFixed:
    """Fixed ADX Live Trading System"""
    
    def __init__(self):
        # MT5 Connection Settings
        self.account = 9956902
        self.password = "@Ripper25"
        self.server = "DerivBVI-Server"
        
        # Trading Settings
        self.symbol = "Volatility 100 (1s) Index.0"
        # Lot size will be calculated dynamically using Kelly from balance/trade stats
        self.lot_size = None
        self.magic_number = 987654321
        self.deviation = 5
        
        # Position Sizing Parameters for V100 (1s)
        self.min_lot = 0.2      # Minimum lot size for V100 (1s)
        self.max_lot = 330.0    # Maximum lot size for V100 (1s)
        self.lot_step = 0.01    # Volume step
        
        # Kelly Criterion Settings
        self.kelly_fraction = 0.75  # Match backtest scaling
        self.min_trades_for_kelly = 10  # Minimum trades needed before using Kelly
        self.max_risk_per_trade = 0.75  # Align with backtest cap
        self.use_kelly = True  # Enable/disable Kelly sizing
        self.use_session_kelly = True  # If True, size from realized session P&L only
        
        # ADX Settings
        self.analysis_length = 14
        self.confirm_on_bar_close = True
        
        # Trading State
        self.is_connected = False
        self.current_position = None
        self.running = False
        self.last_bar_time = None
        
        # Data Storage
        self.max_bars = 500
        
        # Performance Tracking
        self.trades_today = 0
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        # realized P&L list for session-based Kelly (store dicts with pnl and volume)
        self.session_pnls = []

        # Safety: Position moderation
        self.moderator = PositionModerator()
        self.pause_on_fault = True
        
        print("METACOG Live Trader initialized")
        print(f"Symbol: {self.symbol}")
        print(f"Kelly Fraction: {self.kelly_fraction * 100}% (Fractional Kelly)")
        print(f"Min Lot: {self.min_lot}, Max Lot: {self.max_lot}")
        print(f"METACOG Length: {self.analysis_length}")
        print("=" * 60)

    def connect_mt5(self) -> bool:
        try:
            print("Connecting to MT5...")
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"MT5 initialize failed: {error}")
                return False
            if not mt5.login(self.account, self.password, self.server):
                error = mt5.last_error()
                print(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
            account_info = mt5.account_info()
            if account_info is None:
                print("Failed to get account info")
                mt5.shutdown()
                return False
            print(f"Connected to MT5 - Account: {account_info.login}, Balance: ${account_info.balance:.2f}")
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                print(f"Symbol {self.symbol} not found")
                return False
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    print(f"Failed to enable symbol {self.symbol}")
                    return False
                print(f"Enabled symbol {self.symbol}")
            self.is_connected = True
            print(f"Symbol {self.symbol} ready for trading")
            return True
        except Exception as e:
            print(f"MT5 connection error: {e}")
            return False

    def get_current_price(self):
        try:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return None
            return {'bid': tick.bid, 'ask': tick.ask, 'time': datetime.fromtimestamp(tick.time)}
        except Exception as e:
            print(f"Error getting current price: {e}")
            return None

    def get_m1_from_ticks(self, minutes: int = 500):
        """Build 1-minute OHLC from ticks using mid price, matching backtest pipeline."""
        try:
            to_time = datetime.now()
            from_time = to_time - timedelta(minutes=minutes + 2)
            ticks = mt5.copy_ticks_range(self.symbol, from_time, to_time, mt5.COPY_TICKS_ALL)
            if ticks is None or len(ticks) == 0:
                return None
            df = pd.DataFrame(ticks)
            # Convert time
            if 'time_msc' in df.columns and df['time_msc'].notna().any():
                df['time'] = pd.to_datetime(df['time'], unit='s')
            else:
                df['time'] = pd.to_datetime(df['time'], unit='s')
            # Ensure bid/ask exist
            if 'bid' not in df.columns or 'ask' not in df.columns:
                return None
            # Mid price as in backtest
            df['price'] = (df['bid'] + df['ask']) / 2.0
            df.set_index('time', inplace=True)
            ohlc = df['price'].resample('1min').ohlc()
            ohlc = ohlc.dropna()
            # Exclude the last potentially incomplete bar
            if len(ohlc) > 0:
                ohlc = ohlc.iloc[:-1]
            return ohlc[['open', 'high', 'low', 'close']]
        except Exception as e:
            print(f"Error building M1 from ticks: {e}")
            return None
    
    def get_trade_history(self):
        """Fetch actual trade history from MT5 terminal for Kelly calculation"""
        try:
            # Get deals from the last 30 days
            from_date = datetime.now() - timedelta(days=30)
            to_date = datetime.now()
            
            # Fetch deals history
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals is None or len(deals) == 0:
                print("No trade history found in MT5")
                return None
            
            # Convert to DataFrame
            df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
            
            # Filter for our magic number and symbol
            df_deals = df_deals[
                (df_deals['magic'] == self.magic_number) & 
                (df_deals['symbol'] == self.symbol)
            ]
            
            if len(df_deals) == 0:
                print("No trades found for this strategy")
                return None
            
            # Calculate P&L for each closed trade
            trades = []
            for _, deal in df_deals.iterrows():
                if deal['entry'] == mt5.DEAL_ENTRY_OUT:  # Exit deal
                    # Use deal volume to compute per-lot P&L statistics later
                    trades.append({
                        'profit': deal['profit'],
                        'commission': deal['commission'],
                        'swap': deal['swap'],
                        'net_profit': deal['profit'] + deal['commission'] + deal['swap'],
                        'volume': deal.get('volume', None)
                    })
            
            return trades
        except Exception as e:
            print(f"Error fetching trade history: {e}")
            return None
    
    def calculate_kelly_lot_size(self):
        """Calculate optimal lot size using Fractional Kelly Criterion from actual trades"""
        try:
            # Get current account balance
            account_info = mt5.account_info()
            if account_info is None:
                print("Cannot get account info for Kelly calculation")
                return self.min_lot
            
            current_balance = account_info.balance
            
            # If Kelly is disabled or balance too low, use minimum lot
            if not self.use_kelly or current_balance < 10:
                return self.min_lot

            # Session-based Kelly: use realized P&L from this session only
            if self.use_session_kelly:
                vol = self._kelly_volume_from_session(self.session_pnls, current_balance)
                return vol
            
            # Get trade history from MT5
            trades = self.get_trade_history()
            
            # If not enough trades, defer to broker minimum until stats mature (no global heuristics)
            if trades is None or len(trades) < self.min_trades_for_kelly:
                print(f"Not enough trades for Kelly ({len(trades) if trades else 0} < {self.min_trades_for_kelly}), using broker min lot: {self.min_lot}")
                return self.min_lot
            
            # Calculate win rate and average win/loss
            # Compute per-lot net P&L where volume available; fallback to raw if not
            wins = []
            losses = []
            for t in trades:
                vol = t.get('volume') or 0
                npnl = t['net_profit']
                # If volume present and > 0, normalize per lot
                if vol and vol > 0:
                    per_lot_pnl = npnl / vol
                else:
                    per_lot_pnl = npnl
                if per_lot_pnl > 0:
                    wins.append(per_lot_pnl)
                elif per_lot_pnl < 0:
                    losses.append(abs(per_lot_pnl))
            
            if len(wins) == 0 or len(losses) == 0:
                print("Insufficient win/loss data for Kelly")
                return self.min_lot
            
            win_rate = len(wins) / len(trades)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            
            # Kelly formula: f = (p * b - q) / b
            # where p = win probability, q = loss probability, b = win/loss ratio
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate  # Win probability
            q = 1 - p  # Loss probability
            
            # Calculate Kelly percentage
            kelly_percentage = (p * b - q) / b if b > 0 else 0
            
            # Apply fractional Kelly (e.g., 75% of full Kelly per kelly_fraction)
            kelly_percentage = kelly_percentage * self.kelly_fraction
            
            # Apply maximum risk per trade limit
            kelly_percentage = min(kelly_percentage, self.max_risk_per_trade)
            
            # If Kelly is negative, use minimum lot
            if kelly_percentage <= 0:
                print(f"Kelly negative ({kelly_percentage:.2%}), using minimum lot")
                return self.min_lot
            
            # Calculate lot size based on Kelly risk dollars and per-lot average loss
            # Risk dollars per trade:
            risk_dollars = current_balance * kelly_percentage
            # Convert risk dollars into lots using expected loss per lot
            kelly_lot = risk_dollars / avg_loss  # avg_loss is per-lot dollars
            
            # Round to lot step
            kelly_lot = round(kelly_lot / self.lot_step) * self.lot_step
            
            # Apply min/max constraints
            kelly_lot = max(self.min_lot, min(kelly_lot, self.max_lot))
            
            print(f"Kelly Calculation: Win Rate={win_rate:.2%}, B={b:.2f}, Kelly%={kelly_percentage:.2%}, Lot={kelly_lot:.2f}")
            print(f"Based on {len(trades)} trades, Balance=${current_balance:.2f}")
            
            return kelly_lot
            
        except Exception as e:
            print(f"Error in Kelly calculation: {e}")
            return self.min_lot

    def calculate_adx_components(self, df: pd.DataFrame):
        """Calculate DI+ and DI- using Pine-style Wilder smoothing, matching backtest."""
        try:
            high, low, close = df['high'], df['low'], df['close']
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            high_diff = high - prev_high
            low_diff = prev_low - low
            dm_plus = pd.Series(np.where(high_diff > low_diff, np.maximum(high_diff, 0), 0), index=high.index)
            dm_minus = pd.Series(np.where(low_diff > high_diff, np.maximum(low_diff, 0), 0), index=high.index)
            smoothed_tr = self._pine_smooth(true_range)
            smoothed_dm_plus = self._pine_smooth(dm_plus)
            smoothed_dm_minus = self._pine_smooth(dm_minus)
            di_plus = (smoothed_dm_plus / smoothed_tr).replace([np.inf, -np.inf], 0).fillna(0) * 100
            di_minus = (smoothed_dm_minus / smoothed_tr).replace([np.inf, -np.inf], 0).fillna(0) * 100
            return di_plus, di_minus
        except Exception as e:
            print(f"Error in METACOG calculation: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _pine_smooth(self, series: pd.Series):
        """Pine-style smoothing recurrence used in backtest."""
        try:
            smoothed = pd.Series(0.0, index=series.index)
            length = max(1, int(self.analysis_length))
            for i in range(len(series)):
                if i == 0:
                    smoothed.iloc[i] = float(series.iloc[i])
                else:
                    prev_smoothed = smoothed.iloc[i-1]
                    current_value = float(series.iloc[i])
                    smoothed.iloc[i] = prev_smoothed - (prev_smoothed / length) + current_value
            return smoothed
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return series.astype(float)

    def detect_crossover_signal(self, di_plus, di_minus):
        try:
            if di_plus is None or di_minus is None:
                return None
            if len(di_plus) < 2 or len(di_minus) < 2:
                return None
            prev_di_plus = di_plus.iloc[-2]
            prev_di_minus = di_minus.iloc[-2]
            curr_di_plus = di_plus.iloc[-1]
            curr_di_minus = di_minus.iloc[-1]
            if pd.isna(prev_di_plus) or pd.isna(prev_di_minus) or pd.isna(curr_di_plus) or pd.isna(curr_di_minus):
                return None
            if prev_di_plus <= prev_di_minus and curr_di_plus > curr_di_minus:
                return "BUY"
            if prev_di_minus <= prev_di_plus and curr_di_minus > curr_di_plus:
                return "SELL"
            return None
        except Exception:
            return None

    def place_order(self, signal: str, price: float):
        try:
            # Calculate Kelly lot size before each trade
            self.lot_size = self.calculate_kelly_lot_size()
            
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": f"METACOG {signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            print(f"Placing {signal} order: {self.lot_size:.2f} lots at {price:.5f}")
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Order failed: {result.retcode} - {result.comment}")
                return False
            print(f"Order executed: {signal} {self.lot_size} lots at {result.price:.5f}")
            self.current_position = {
                'signal': signal,
                'volume': self.lot_size,
                'price': result.price,
                'time': datetime.now(),
                'order': result.order
            }
            self.moderator.on_order_executed(signal)
            return True
        except Exception as e:
            print(f"Error placing order: {e}")
            return False

    def close_position(self, price: float):
        if not self.current_position:
            return True
        try:
            order_type = mt5.ORDER_TYPE_SELL if self.current_position['signal'] == "BUY" else mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.current_position['volume'],
                "type": order_type,
                "position": self.current_position['order'],
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": f"Close METACOG {self.current_position['signal']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            print(f"Closing {self.current_position['signal']} position at {price:.5f}")
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Close order failed: {result.retcode} - {result.comment}")
                return False
            # For V100 (1s): 1 lot moving 1.00 = $100
            if self.current_position['signal'] == "BUY":
                pnl = (price - self.current_position['price']) * self.current_position['volume'] * 100
            else:
                pnl = (self.current_position['price'] - price) * self.current_position['volume'] * 100
            self.total_pnl += pnl
            # Track session P&L and volume for session-based Kelly sizing
            self.session_pnls.append({
                'pnl': pnl,
                'volume': self.current_position['volume']
            })
            print(f"Position closed at {result.price:.5f}, P&L: ${pnl:.2f}")
            if pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.current_position = None
            return True
        except Exception as e:
            print(f"Error closing position: {e}")
            return False

    def _kelly_volume_from_session(self, history_pnls, current_balance_local):
        """Compute Kelly lot using session P&L (same formula as backtest)."""
        try:
            if len(history_pnls) < self.min_trades_for_kelly:
                return self.min_lot
            # Normalize per-lot P&L using stored volumes when available
            wins = []
            losses = []
            for item in history_pnls:
                if isinstance(item, dict):
                    pnl = item.get('pnl', 0.0)
                    vol = item.get('volume', 0.0)
                else:
                    pnl = float(item)
                    vol = 0.0
                per_lot = (pnl / vol) if vol and vol > 0 else pnl
                if per_lot > 0:
                    wins.append(per_lot)
                elif per_lot < 0:
                    losses.append(abs(per_lot))
            if len(wins) == 0 or len(losses) == 0:
                return self.min_lot
            p = len(wins) / len(history_pnls)
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            b = avg_win / avg_loss if avg_loss > 0 else 0.0
            q = 1 - p
            f = (b * p - q) / b if b > 0 else 0.0
            f = max(0.0, min(f, self.max_risk_per_trade))
            f *= self.kelly_fraction
            if f <= 0:
                return self.min_lot
            # Convert desired risk dollars into lots using expected loss per lot
            account_risk = current_balance_local * f
            vol = account_risk / avg_loss
            # Round and clamp
            vol = round(vol / self.lot_step) * self.lot_step
            vol = max(self.min_lot, min(vol, self.max_lot))
            return vol
        except Exception:
            return self.min_lot

    def trading_loop(self):
        print("Starting trading loop...")
        while self.running:
            try:
                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(1)
                    continue
                df = self.get_m1_from_ticks(self.max_bars)
                if df is None or len(df) < self.analysis_length + 10:
                    time.sleep(5)
                    continue
                di_plus, di_minus = self.calculate_adx_components(df)
                if di_plus is None or di_minus is None:
                    time.sleep(5)
                    continue
                signal = None
                # Evaluate on closed bars only (df already excludes the last incomplete bar)
                current_bar_time = df.index[-1]
                if self.last_bar_time is None:
                    self.last_bar_time = current_bar_time
                if current_bar_time != self.last_bar_time:
                    self.last_bar_time = current_bar_time
                    if len(di_plus) >= 2 and len(di_minus) >= 2:
                        signal = self.detect_crossover_signal(di_plus, di_minus)
                if signal:
                    trade_price = current_price['ask'] if signal == "BUY" else current_price['bid']
                    bar_time_for_moderation = df.index[-2] if self.confirm_on_bar_close else df.index[-1]
                    approved, reason = self.moderator.approve(signal, bar_time_for_moderation, self.current_position)
                    if not approved:
                        print(f"FAULT: moderator rejected {signal} due to {reason}. Pausing.")
                        if self.pause_on_fault:
                            self.running = False
                        time.sleep(1)
                        continue
                    self.moderator.mark_processed(bar_time_for_moderation)
                    if self.current_position:
                        if self.current_position['signal'] != signal:
                            close_price = current_price['bid'] if self.current_position['signal'] == "BUY" else current_price['ask']
                            if self.close_position(close_price):
                                self.place_order(signal, trade_price)
                    else:
                        self.place_order(signal, trade_price)
                if self.current_position:
                    print(f"Current Position: {self.current_position['signal']} at {self.current_position['price']:.5f}")
                else:
                    print("No Position")
                print(f"Total P&L: ${self.total_pnl:.2f}")
                print(f"Wins: {self.winning_trades}, Losses: {self.losing_trades}")
                print("-" * 40)
                time.sleep(5)
            except KeyboardInterrupt:
                print("Trading loop interrupted by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)

    def start_trading(self):
        print("=" * 60)
        print("METACOG LIVE TRADER STARTING")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        # Lot size is computed dynamically; display placeholder
        print("Lot Size: dynamic (Kelly-based)")
        print(f"METACOG Length: {self.analysis_length}")
        print("=" * 60)
        try:
            if not self.connect_mt5():
                print("Failed to connect to MT5")
                return
            self.running = True
            self.trading_loop()
        except KeyboardInterrupt:
            print("Stopping trading...")
        except Exception as e:
            print(f"Trading error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.current_position:
                print("Closing open position before shutdown...")
                current_price = self.get_current_price()
                if current_price:
                    close_price = current_price['bid'] if self.current_position['signal'] == "BUY" else current_price['ask']
                    self.close_position(close_price)
            self.running = False
            mt5.shutdown()
            print("Trading system stopped")

def main():
    print("METACOG Live Trader")
    print("=" * 50)
    trader = ADXLiveTraderFixed()
    try:
        trader.start_trading()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        trader.running = False
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
