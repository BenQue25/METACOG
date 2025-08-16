# METACOGNITION - Advanced ADX Crossover Trading System

## Overview
MetaCog is a high-frequency trading system that combines ADX crossover signals with mathematical Kelly Criterion position sizing, utilizing SOTA TCP_NODELAY technology for submillisecond execution.

## Features
- **ADX Crossover Logic**: Exact same algorithm as backtest (non-repainting)
- **Mathematical Kelly Criterion**: Fractional Kelly with 25% risk scaling
- **FOK Execution**: Fill-or-Kill order filling for immediate execution
- **TCP_NODELAY**: SOTA latency optimization for submillisecond performance
- **Real-time Trading**: Live MT5 integration with no simulations

## Requirements
- MetaTrader5 terminal
- Python 3.7+
- Required packages: `MetaTrader5`, `pandas`, `numpy`

## Installation
1. Ensure MT5 terminal is running and logged in
2. Install required packages: `pip install MetaTrader5 pandas numpy`
3. Place script in MT5's Python scripts folder or run from current location

## Usage
```bash
cd METACOGNITION
python metacog_live_trader.py
```

## Trading Logic
- **Entry**: BUY on DI+ crossover above DI-, SELL on DI- crossover above DI+
- **Exit**: Opposite signal (close BUY on SELL signal, close SELL on BUY signal)
- **Position Sizing**: Dynamic Kelly Criterion based on win rate and profit/loss ratios
- **Risk Management**: 25% fractional Kelly with volume constraints

## Performance Tracking
- Real-time P&L calculation
- Win rate and consecutive wins/losses
- Execution time measurement (milliseconds)
- Trade history with detailed metrics

## Technology Stack
- **Platform**: MetaTrader5
- **Language**: Python 3
- **Networking**: TCP_NODELAY for minimal latency
- **Order Type**: FOK (Fill or Kill)
- **Data**: Real-time tick data with 1-minute OHLC aggregation

## Risk Warning
This is a high-frequency trading system designed for experienced traders. Past performance does not guarantee future results. Use at your own risk.

## Support
For technical issues, check MT5 terminal logs and ensure proper connectivity.
