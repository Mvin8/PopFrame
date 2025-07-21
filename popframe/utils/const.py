"""
Constants for PopFrame library
Deprecated: Use popframe.config.constants instead
"""

import warnings
from ..config.constants import UNITS

warnings.warn(
    "popframe.utils.const is deprecated. Use popframe.config.constants instead.",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility
SQUARE_METERS_IN_HECTARE = UNITS['square_meters_in_hectare']
