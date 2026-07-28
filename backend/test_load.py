import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from models import cocoa_models
    print("Success: Models loaded successfully.")
except Exception as e:
    print(f"Error: Failed to load models. {e}")
    import traceback
    traceback.print_exc()
