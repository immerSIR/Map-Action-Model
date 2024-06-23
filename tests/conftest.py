# tests/conftest.py

import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Add the code directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'code')))

# Add the steps directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'code', 'steps')))
