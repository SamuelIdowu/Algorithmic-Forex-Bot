import argparse
import pandas as pd
import mplfinance as mpf
from predict import predict_next_movement
import os
import sys
import glob

def get_available_models():
    """
    Scan the models directory for available model/scaler pairs.
    Returns a dictionary where keys are model names (e.g., 'gold') and values are dicts with paths.
    """
    models_dir = 'models'
    if not os.path.exists(models_dir):
        return {}

    model_files = glob.glob(os.path.join(models_dir, '*_model.pkl'))
    available_models = {}

    for model_path in model_files:
        # Extract base name (e.g., 'gold_model.pkl' -> 'gold')
        filename = os.path.basename(model_path)
        base_name = filename.replace('_model.pkl', '')
        
        scaler_path = os.path.join(models_dir, f"{base_name}_scaler.pkl")
        
        if os.path.exists(scaler_path):
            available_models[base_name] = {
                'model': model_path,
                'scaler': scaler_path
            }
    
    return available_models

def interactive_mode():
    """
    Interactive mode to select model and symbol.
    """
    print("\n=== Interactive Prediction Visualization ===\n")
    
    # 1. Select Model
    models = get_available_models()
    if not models:
        print("No models found in 'models/' directory.")
        return

    print("Available Models:")
    model_names = list(models.keys())
    for i, name in enumerate(model_names):
        print(f"{i + 1}. {name}")
    
    while True:
        try:
            choice = input("\nSelect a model (number): ")
            idx = int(choice) - 1
            if 0 <= idx < len(model_names):
                selected_model_name = model_names[idx]
                selected_model = models[selected_model_name]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

    # 2. Enter Symbol
    default_symbol = 'GC=F' if selected_model_name == 'gold' else 'BTC-USD'
    symbol = input(f"\nEnter symbol (default: {default_symbol}): ").strip()
    if not symbol:
        symbol = default_symbol

    print(f"\nSelected Model: {selected_model_name}")
    print(f"Target Symbol: {symbol}")
    print("-" * 30)

    visualize_prediction(symbol, selected_model['model'], selected_model['scaler'])

def visualize_prediction(symbol, model_path, scaler_path, output_file=None):
    # Get prediction and data
    print(f"Generating visualization for {symbol}...")
    result = predict_next_movement(symbol, model_path, scaler_path)
    
    if not result:
        print("Failed to get prediction.")
        return

    data = result['history']
    # Focus on the last 60 days for clarity
    plot_data = data.tail(60).copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(plot_data.index, pd.DatetimeIndex):
        plot_data.index = pd.to_datetime(plot_data.index)

    # Prediction details
    prediction = result['prediction']
    confidence = result['confidence']
    last_date = result['date']
    last_close = result['current_price']
    
    # Prepare markers for the prediction
    # We'll place a marker on the last candle
    # Create a series of NaNs
    marker_data = [float('nan')] * len(plot_data)
    
    # Set the last value to the close price (or slightly above/below)
    if prediction == "UP":
        marker_color = 'green'
        marker_type = '^'
        offset = last_close * 1.01
    else:
        marker_color = 'red'
        marker_type = 'v'
        offset = last_close * 0.99
        
    marker_data[-1] = offset

    # Create addplots for indicators and prediction
    addplots = []
    
    # Add SMAs if they exist in the data (predict.py now returns data with features)
    if 'sma_20' in plot_data.columns:
        addplots.append(mpf.make_addplot(plot_data['sma_20'], color='blue', width=1.0))
    if 'sma_50' in plot_data.columns: # Assuming sma_50 might be added or we can calculate it
        # If sma_50 is not in features, we can calculate it quickly here or skip
        # features.py only adds sma_20. Let's calculate sma_50 for visualization if possible
        sma_50 = plot_data['close'].rolling(window=50).mean()
        # Only plot if we have enough data
        if not sma_50.isnull().all():
             addplots.append(mpf.make_addplot(sma_50, color='red', width=1.0))
    
    # Add the prediction marker
    addplots.append(mpf.make_addplot(marker_data, type='scatter', markersize=200, marker=marker_type, color=marker_color))

    # Create the plot
    # Style: 'yahoo' is close to standard financial plots. 'charles' is also good.
    # Backtrader often uses a white background with red/green candles.
    
    title = f"Prediction: {prediction} ({confidence:.1%}) | {symbol}"
    
    # Prepare plot arguments
    plot_kwargs = dict(
        type='candle',
        style='yahoo',
        volume=True,
        addplot=addplots,
        title=title,
        ylabel='Price',
        ylabel_lower='Volume',
        figsize=(12, 8),
        tight_layout=True
    )

    if output_file:
        plot_kwargs['savefig'] = dict(fname=output_file, dpi=100, bbox_inches='tight')

    mpf.plot(plot_data, **plot_kwargs)
    
    if output_file:
        print(f"Visualization saved to {output_file}")
    else:
        mpf.show()

if __name__ == "__main__":
    # Check if arguments are provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Visualize next market movement')
        parser.add_argument('--symbol', type=str, default='GC=F', help='Symbol to predict')
        parser.add_argument('--model_path', type=str, default='models/ml_strategy_model.pkl', help='Path to trained model')
        parser.add_argument('--scaler_path', type=str, default='models/ml_strategy_scaler.pkl', help='Path to scaler')
        parser.add_argument('--output', type=str, default=None, help='Output file to save plot (optional)')
        
        args = parser.parse_args()
        
        visualize_prediction(args.symbol, args.model_path, args.scaler_path, args.output)
    else:
        # Interactive mode
        interactive_mode()
