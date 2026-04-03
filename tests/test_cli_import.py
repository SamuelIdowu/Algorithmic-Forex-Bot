import sys
import os

# Add the directory to path so we can import cli
sys.path.append(os.path.abspath("/home/samuel/Desktop/Algo Bot/algo_trader"))

try:
    import cli
    print("✅ Successfully imported cli module")
except ImportError as e:
    print(f"❌ ImportError: {e}")
except SyntaxError as e:
    print(f"❌ SyntaxError: {e}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
