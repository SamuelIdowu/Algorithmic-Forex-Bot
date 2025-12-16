from data.db_manager import DatabaseManager
import pandas as pd

db = DatabaseManager()
df = db.load_data("EURUSD=X")
print(f"Loaded {len(df)} rows")
print("Volume stats:")
print(df['volume'].describe())
print(f"Zero volume count: {(df['volume'] == 0).sum()}")
