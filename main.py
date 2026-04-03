from utils.config import MODE, validate_config
import sys


def main():
    """
    Main entry point for the algorithmic trading bot
    """
    # Validate configuration based on mode
    try:
        validate_config()
        print(f"Configuration validated for {MODE} mode")
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    if MODE == "backtest":
        from backtest.backtest_engine import run_backtest
        from strategies.moving_average import MovingAverageStrategy, RSIStrategy
        from strategies.ml_predictive import MLPredictiveStrategy, MLPredictiveStrategyWithRiskManagement
        
        print("Running in backtesting mode...")

        # Strategy Selection
        from utils.config import STRATEGY, CONFIDENCE_THRESHOLD, RISK_REWARD_SL_MULT, RISK_REWARD_TP_MULT, TIMEFRAME, LOOKBACK_PERIOD, TRADING_SYMBOL
        import os
        
        print(f"Selected Strategy: {STRATEGY}")
        
        strategy_class = None
        kwargs = {}
        
        if STRATEGY == "moving_average":
            strategy_class = MovingAverageStrategy
        elif STRATEGY == "rsi":
            strategy_class = RSIStrategy
            kwargs = {"start": "2022-01-01", "end": "2023-01-01"}
        elif STRATEGY == "ml_predictive":
            # Check if model exists
            if not os.path.exists("models/ml_strategy_model.pkl"):
                print("Error: Model file not found at models/ml_strategy_model.pkl")
                print("Please train the model first using: python train_model.py")
                sys.exit(1)
            strategy_class = MLPredictiveStrategy
            # Pass configuration to strategy
            kwargs = {
                "start": "2022-01-01", 
                "end": "2023-01-01",
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "lookback_period": LOOKBACK_PERIOD
            }
        elif STRATEGY == "ml_predictive_risk_managed":
             # Check if model exists
            if not os.path.exists("models/ml_strategy_model.pkl"):
                print("Error: Model file not found at models/ml_strategy_model.pkl")
                print("Please train the model first using: python train_model.py")
                sys.exit(1)
            strategy_class = MLPredictiveStrategyWithRiskManagement
            # Pass configuration to strategy
            kwargs = {
                "start": "2022-01-01", 
                "end": "2023-01-01",
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "lookback_period": LOOKBACK_PERIOD,
                "stop_loss_atr_multiplier": RISK_REWARD_SL_MULT,
                "take_profit_atr_multiplier": RISK_REWARD_TP_MULT
            }
        else:
            print(f"Unknown strategy: {STRATEGY}. Defaulting to MovingAverageStrategy")
            strategy_class = MovingAverageStrategy

        if strategy_class:
            run_backtest(
                symbol=TRADING_SYMBOL,
                strategy_class=strategy_class,
                initial_cash=10000,
                interval=TIMEFRAME,
                **kwargs
            )

        # Run Forex backtest if requested (optional, can be controlled by another env var or separate command)
        # For now, we'll keep the specific XAU/USD test separate or user can comment/uncomment as needed
        # or we can add a specific strategy for it.
        
        if STRATEGY == "xau_usd":
             print("\nRunning XAU/USD Intraday Backtest...")
             run_backtest(
                symbol="XAU/USD",
                strategy_class=MovingAverageStrategy,
                start="2025-11-23",
                end="2025-11-24", 
                initial_cash=10000,
                asset_type="forex",
                interval="1h"
            )
        
    elif MODE == "paper" or MODE == "live":
        # Import live trading module only when needed to avoid urllib3 issues
        from live_trading.trader import run_paper_trade_example
        print(f"Running in {MODE} trading mode...")
        run_paper_trade_example()
        
    else:
        print(f"Unknown mode: {MODE}. Please set MODE to 'backtest', 'paper', or 'live' in your .env file.")
        sys.exit(1)


if __name__ == "__main__":
    main()