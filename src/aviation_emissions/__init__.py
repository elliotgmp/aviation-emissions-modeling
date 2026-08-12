"""Aviation Emissions Modeling & AI Dynamics.

A production-grade pipeline for modelling CO2 and non-CO2 emissions of
commercial aviation from operational flight records, built around a vectorised
implementation of the ICAO CORSIA CEM fuel-burn model.

Quick start
-----------
>>> from aviation_emissions import load_config, CEMLibrary
>>> cfg = load_config("configs/config.yaml")
>>> models = CEMLibrary.from_yaml("configs/icao_cem_coefficients.yaml")
>>> models["B738"].co2(1000.0)          # kg CO2 for a 1000 km B737-800 leg
"""

from .config import Config, load_config
from .emissions.cem import CEMLibrary, FleetCEM, PiecewiseCEM

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "CEMLibrary", "FleetCEM", "PiecewiseCEM",
           "__version__"]
