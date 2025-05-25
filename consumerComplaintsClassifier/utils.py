# utils.py
import pickle
import os

def save_object(obj, filepath):
    """Saves a Python object to a file using pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")

def load_object(filepath):
    """Loads a Python object from a file using pickle."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}. Returning None.")
        return None
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj