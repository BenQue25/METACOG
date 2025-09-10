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
        self.account = 5749910
        self.password = "@Ripper25"
        self.server = "Deriv-Demo"
        
        # Trading Settings
        self.symbol = "Volatility 10 Index.0"
        self.lot_size = 1.0
        self.magic_number = 987654321
        self.deviation = 20
        
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

        # Safety: Position moderation
        self.moderator = PositionModerator()
        self.pause_on_fault = True
        
        print("ADX Live Trader Fixed initialized")
        print(f"Symbol: {self.symbol}")
        print(f"Lot Size: {self.lot_size}")
        print(f"ADX Length: {self.analysis_length}")
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

    def get_historical_data(self, bars: int = 500):
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df[['open', 'high', 'low', 'close']]
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None

    def calculate_adx_components(self, df: pd.DataFrame):
        try:
            high, low, close = df['high'], df['low'], df['close']
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            prev_high = high.shift(1)
            prev_low = low.shift(1)
            high_diff = high - prev_high
            low_diff = prev_low - low
            dm_plus = np.where(high_diff > low_diff, np.maximum(high_diff, 0), 0)
            dm_minus = np.where(low_diff > high_diff, np.maximum(low_diff, 0), 0)
            true_range_series = pd.Series(true_range, index=df.index)
            dm_plus_series = pd.Series(dm_plus, index=df.index)
            dm_minus_series = pd.Series(dm_minus, index=df.index)
            smoothed_tr = self._smooth_series(true_range_series)
            smoothed_dm_plus = self._smooth_series(dm_plus_series)
            smoothed_dm_minus = self._smooth_series(dm_minus_series)
            di_plus = np.where(smoothed_tr > 0, (smoothed_dm_plus / smoothed_tr) * 100, 0)
            di_minus = np.where(smoothed_tr > 0, (smoothed_dm_minus / smoothed_tr) * 100, 0)
            di_plus = pd.Series(di_plus, index=df.index).fillna(0)
            di_minus = pd.Series(di_minus, index=df.index).fillna(0)
            return di_plus, di_minus
        except Exception as e:
            print(f"Error in ADX calculation: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _smooth_series(self, series: pd.Series):
        try:
            # Wilder's smoothing can be computed via an EMA with alpha = 1/N
            if self.analysis_length <= 0:
                return series.astype(float)
            return series.astype(float).ewm(alpha=1.0 / self.analysis_length, adjust=False).mean()
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return series

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
            order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": order_type,
                "price": price,
                "deviation": self.deviation,
                "magic": self.magic_number,
                "comment": f"ADX {signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            print(f"Placing {signal} order: {self.lot_size} lots at {price:.5f}")
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
                "comment": f"Close ADX {self.current_position['signal']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            print(f"Closing {self.current_position['signal']} position at {price:.5f}")
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"Close order failed: {result.retcode} - {result.comment}")
                return False
            if self.current_position['signal'] == "BUY":
                pnl = (price - self.current_position['price']) * self.lot_size * 1000
            else:
                pnl = (self.current_position['price'] - price) * self.lot_size * 1000
            self.total_pnl += pnl
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

    def trading_loop(self):
        print("Starting trading loop...")
        while self.running:
            try:
                current_price = self.get_current_price()
                if not current_price:
                    time.sleep(1)
                    continue
                df = self.get_historical_data(self.max_bars)
                if df is None or len(df) < self.analysis_length + 10:
                    time.sleep(5)
                    continue
                di_plus, di_minus = self.calculate_adx_components(df)
                if di_plus is None or di_minus is None:
                    time.sleep(5)
                    continue
                signal = None
                if self.confirm_on_bar_close:
                    current_bar_time = df.index[-1]
                    if self.last_bar_time is None:
                        self.last_bar_time = current_bar_time
                    if current_bar_time != self.last_bar_time:
                        self.last_bar_time = current_bar_time
                        if len(di_plus) >= 3 and len(di_minus) >= 3:
                            signal = self.detect_crossover_signal(di_plus.iloc[:-1], di_minus.iloc[:-1])
                else:
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
        print("ADX LIVE TRADER FIXED STARTING")
        print("=" * 60)
        print(f"Symbol: {self.symbol}")
        print(f"Lot Size: {self.lot_size}")
        print(f"ADX Length: {self.analysis_length}")
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
    print("ADX Live Trader - Fixed Version")
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
