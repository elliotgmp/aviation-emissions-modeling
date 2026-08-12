"""Physical emissions models: CEM fuel burn, intensity, non-CO2, scenarios."""

from .cem import CEMLibrary, FleetCEM, PiecewiseCEM
from .intensity import (assign_flight_emissions, efficiency_table,
                        extrapolate_fleet_emissions, intensity_curve,
                        optimal_allocation_counterfactual)
from .non_co2 import (RegressionResult, bootstrap_slope_ci,
                      climb_speed_sensitivity, fit_nox_co2, phase_allocation)
from .scenarios import (apply_calibration, fleet_substitution,
                        piano_x_calibration, route_detour_penalty)

__all__ = [
    "CEMLibrary", "FleetCEM", "PiecewiseCEM",
    "intensity_curve", "efficiency_table", "assign_flight_emissions",
    "extrapolate_fleet_emissions", "optimal_allocation_counterfactual",
    "fit_nox_co2", "bootstrap_slope_ci", "phase_allocation",
    "climb_speed_sensitivity", "RegressionResult",
    "fleet_substitution", "route_detour_penalty", "piano_x_calibration",
    "apply_calibration",
]
