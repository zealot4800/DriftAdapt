"""Simulation package, including import aliases used by the existing testbed."""
import sys
from . import drift, models
from .methods import caravan

for name, module in {'drift_signature': drift, 'trigger': drift,
                     'labelers': caravan, 'trainer': models}.items():
    sys.modules[f'{__name__}.{name}'] = module
