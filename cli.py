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
            
            cmd = [sys.executable, "predict.py", "--symbol", symbol]
            
            custom_paths = questionary.confirm("Do you want to specify custom model/scaler paths?", default=False).ask()
            if custom_paths:
                model_path = questionary.text("Model Path:", default=f"models/{symbol.lower().replace('/', '_')}_model.pkl").ask()
                scaler_path = questionary.text("Scaler Path:", default=f"models/{symbol.lower().replace('/', '_')}_scaler.pkl").ask()
                cmd.extend(["--model_path", model_path, "--scaler_path", scaler_path])
            
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
            
            # Backtest usually runs on AAPL by default in main.py but we can override if main.py supports it via env var or args
            # Looking at USER_GUIDE, TRADING_SYMBOL env var is used for paper/live, but main.py might use it for backtest too?
            # Guide says: STRATEGY=ml_predictive MODE=backtest python main.py
            # It doesn't explicitly say TRADING_SYMBOL works for backtest, but it's likely. 
            # Let's ask for symbol anyway and set the env var.
            
            symbol = get_symbol("Select Symbol to Backtest:")
            
            env = os.environ.copy()
            env["STRATEGY"] = strategy
            env["MODE"] = "backtest"
            env["TRADING_SYMBOL"] = symbol
            
            print(f"\n🚀 Running Backtest with Strategy: {strategy} on {symbol}\n")
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
