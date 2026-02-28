import questionary
import subprocess
import sys
import os


COMMON_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "BTC-USD",
    "ETH-USD",
    "XAU/USD",
    "SPY",
    "AAPL",
    "MSFT",
    "GOOGL",
    "Other (enter manually)"
]

def get_symbol(message="Enter Symbol:"):
    """Helper to select a symbol from common pairs or enter manually."""
    selection = questionary.select(
        message,
        choices=COMMON_PAIRS
    ).ask()
    
    if selection == "Other (enter manually)":
        return questionary.text("Enter Custom Symbol (e.g., TSLA, AUD/CAD):").ask()
    return selection

def run_command(command):
    """Executes a shell command and streams the output."""
    try:
        print(f"\n🚀 Running: {' '.join(command)}\n")
        subprocess.run(command, check=True)
        print("\n✅ Done!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: Command failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")

def main():
    print("\n🤖 Algo Trader CLI\n")

    while True:
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Train Model",
                "Predict",
                "Backtest",
                "Paper Trading",
                "Exit"
            ]
        ).ask()

        if action == "Exit":
            print("Goodbye! 👋")
            break

        if action == "Train Model":
            symbol = get_symbol("Select Symbol to Train:")
            start_date = questionary.text("Start Date (YYYY-MM-DD):", default="2020-01-01").ask()
            end_date = questionary.text("End Date (YYYY-MM-DD):", default="2023-01-01").ask()
            
            cmd = [sys.executable, "train_model.py", "--symbol", symbol, "--start", start_date, "--end", end_date]
            
            # Optional: Ask for custom model path if needed, but keeping it simple for now as per user guide defaults
            # If user wants to specify paths for Gold/BTC as per guide:
            if symbol not in ["AAPL", "SPY"]: # Heuristic check, or just ask
                custom_paths = questionary.confirm("Do you want to specify custom model/scaler paths?", default=False).ask()
                if custom_paths:
                    model_path = questionary.text("Model Path:", default=f"models/{symbol.lower().replace('/', '_')}_model.pkl").ask()
                    scaler_path = questionary.text("Scaler Path:", default=f"models/{symbol.lower().replace('/', '_')}_scaler.pkl").ask()
                    cmd.extend(["--model_path", model_path, "--scaler_path", scaler_path])

            run_command(cmd)

        elif action == "Predict":
            symbol = get_symbol("Select Symbol to Predict:")
            
            # Default advanced settings
            timeframe = "1d"
            lookback = "100"
            rr_sl = "2.0" 
            rr_tp = "3.0"
            
            custom_paths = questionary.confirm("Do you want to specify custom model/scaler paths?", default=False).ask()
            model_path = "models/ml_strategy_model.pkl"
            scaler_path = "models/ml_strategy_scaler.pkl"
            
            if custom_paths:
                model_path = questionary.text("Model Path:", default=f"models/{symbol.lower().replace('/', '_')}_model.pkl").ask()
                scaler_path = questionary.text("Scaler Path:", default=f"models/{symbol.lower().replace('/', '_')}_scaler.pkl").ask()

            # Customize parameters
            customize = questionary.confirm("Do you want to customize parameters (Timeframe, Risk/Reward)?", default=False).ask()
            if customize:
                timeframe = questionary.select(
                    "Select Timeframe:",
                    choices=["1d", "1h", "15m", "5m"],
                    default="1d"
                ).ask()
                lookback = questionary.text("Lookback Period (candles):", default="100").ask()
                rr_sl = questionary.text("Stop Loss ATR Multiplier:", default="2.0").ask()
                rr_tp = questionary.text("Take Profit ATR Multiplier:", default="3.0").ask()

            cmd = [
                sys.executable, "predict.py", 
                "--symbol", symbol,
                "--model_path", model_path,
                "--scaler_path", scaler_path,
                "--interval", timeframe,
                "--lookback", lookback,
                "--sl_mult", rr_sl,
                "--tp_mult", rr_tp
            ]
            
            run_command(cmd)

        elif action == "Backtest":
            strategy = questionary.select(
                "Select Strategy:",
                choices=[
                    "moving_average",
                    "rsi",
                    "ml_predictive",
                    "ml_predictive_risk_managed"
                ]
            ).ask()
            
            symbol = get_symbol("Select Symbol to Backtest:")
            
            # Default advanced settings
            timeframe = "1d"
            confidence = "0.7"
            lookback = "100"
            rr_sl = "2.0" 
            rr_tp = "3.0"

            # Ask if user wants to customize advanced settings
            customize = questionary.confirm("Do you want to customize strategy parameters (Timeframe, Confidence, Risk/Reward)?", default=False).ask()
            
            if customize:
                timeframe = questionary.select(
                    "Select Timeframe:",
                    choices=["1d", "1h", "15m", "5m"],
                    default="1d"
                ).ask()
                
                if "ml_predictive" in strategy:
                    confidence = questionary.text("Confidence Threshold (0.0-1.0):", default="0.7").ask()
                    lookback = questionary.text("Lookback Period (candles):", default="100").ask()
                
                if strategy == "ml_predictive_risk_managed":
                    rr_sl = questionary.text("Stop Loss ATR Multiplier:", default="2.0").ask()
                    rr_tp = questionary.text("Take Profit ATR Multiplier:", default="3.0").ask()

            env = os.environ.copy()
            env["STRATEGY"] = strategy
            env["MODE"] = "backtest"
            env["TRADING_SYMBOL"] = symbol
            env["TIMEFRAME"] = timeframe
            env["CONFIDENCE_THRESHOLD"] = confidence
            env["LOOKBACK_PERIOD"] = lookback
            env["RR_SL_MULT"] = rr_sl
            env["RR_TP_MULT"] = rr_tp
            
            print(f"\n🚀 Running Backtest with Strategy: {strategy} on {symbol} ({timeframe})\n")
            if "ml" in strategy:
                print(f"Confidence: {confidence}, Lookback: {lookback}")
            if strategy == "ml_predictive_risk_managed":
                print(f"Risk: {rr_sl}x ATR, Reward: {rr_tp}x ATR")
            print("")

            try:
                subprocess.run([sys.executable, "main.py"], env=env, check=True)
                print("\n✅ Done!")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error: Backtest failed with exit code {e.returncode}")

        elif action == "Paper Trading":
            symbol = get_symbol("Select Symbol to Trade:")
            
            env = os.environ.copy()
            env["MODE"] = "paper"
            env["TRADING_SYMBOL"] = symbol
            
            print(f"\n🚀 Starting Paper Trading for {symbol}...\n")
            try:
                subprocess.run([sys.executable, "main.py"], env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error: Paper trading stopped with exit code {e.returncode}")
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye! 👋")
