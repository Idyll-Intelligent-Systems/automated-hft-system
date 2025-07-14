"""
HFT Trading System - Python Package
====================================

High-frequency trading system components implemented in Python.
This package provides analytics, backtesting, monitoring, and ML capabilities.
"""

__version__ = "1.0.0"
__author__ = "HFT Team"
__email__ = "team@hft-system.com"

from .analytics import *
from .backtesting import *
from .monitoring import *
from .api import *

__all__ = [
    "analytics",
    "backtesting", 
    "monitoring",
    "api",
]
