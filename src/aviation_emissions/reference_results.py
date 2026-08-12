"""
Frozen reference results - the regression suite of the whole project.

Two independent sources of truth are stored here:

* ``NOTEBOOK_*``  - values produced by the original exploratory notebook on the
  November-2025 extract (1 278 775 flights). They pin the *numerical* behaviour
  of the refactored pipeline: any change to :mod:`emissions` that moves one of
  these is a regression until proven otherwise.

* ``PIANOX_*``    - values produced by **Piano-X**, an independent trajectory-
  based performance tool (Lissys). Piano-X integrates an actual flight profile
  (climb schedule, step cruise, reserves) whereas the ICAO CEM is a regression
  on distance alone with conservative margins for weather and routing. Their
  disagreement is not noise: it is the *model risk* of the CEM, and it is
  systematic (CEM always higher, by 6.5% to 19.3%).

Keeping both in one module means the calibration bias can be measured, not
assumed - see :mod:`aviation_emissions.validation`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATASET = {
    "n_flights": 1_278_775,
    "n_features": 79,
    "period": "2025-11-10 .. 2025-11-16",
    "n_distance_non_null": 1_114_818,
    "distance_null_pct": 12.82,        # 163 957 rows
    "traffic_growth_vs_reference_pct": 27.81,  # 2025 extract vs COVID-era extract
}

# pandas .describe() on distance, after the nm -> km conversion (x 1.852)
NOTEBOOK_DISTANCE_DESCRIBE_KM = {
    "count": 1_114_818,
    "mean": 586.7956,
    "std": 878.7016,
    "min": 0.0,
    "p25": 103.25,
    "p50": 310.83,
    "p75": 729.1975,
    "max": 139_318.1,                  # corrupted track - see cleaning.py
}

# Global fleet mix, % of flights
NOTEBOOK_FLEET_MIX_PCT = {
    "B738": 9.112115,
    "A320": 7.998424,
    "C172": 7.046663,
    "A20N": 4.323268,
    "P28A": 4.070582,
}

# ANA sub-fleet mix, % of that operator's flights
NOTEBOOK_ANA_FLEET_MIX_PCT = {
    "A21N": 22.766458,
    "B763": 19.553292,
    "A20N": 17.163009,
    "B772": 11.794671,
    "B78X": 9.404389,
    "A321": 5.799373,
    "B77W": 4.584639,
    "B789": 3.918495,
    "B773": 2.860502,
    "A388": 1.097179,
    "B77L": 1.018809,
    "B738": 0.039185,
}

# ---------------------------------------------------------------------------
# CEM golden values (fuel in kg) - unit-test anchors
# ---------------------------------------------------------------------------

NOTEBOOK_FUEL_GOLDEN = {
    ("B738", 1_000.0): 3_921.52518366,
    ("B763", 50.0): 1_965.5708548,
    ("B763", 500.0): 4_487.71071747,
    ("B763", 1_000.0): 7_290.08834266,
    ("B763", 6_000.0): 36_648.61931372542,
    ("B763", 10_000.0): 61_977.32045123,
    ("B763", 60_000.0): 378_586.08467004,
    ("B763", 100_000.0): 631_873.09604508,
}

# Minimum specific emission (kg CO2 per tonne-km) and the distance at which it
# is reached, read off the intensity curves.
NOTEBOOK_OPTIMAL_RANGE = {
    "B738": {"distance_km": 3_004.51, "intensity": 0.531},
    "A20N": {"distance_km": 2_100.0, "intensity": 0.634},
    "A21N": {"distance_km": 2_500.0, "intensity": 0.559},
    "B763": {"distance_km": 4_165.08, "intensity": 0.329},
    "B78X": {"distance_km": 4_271.52, "intensity": 0.7433777134680729},
}

# ---------------------------------------------------------------------------
# ANA case study (one week)
# ---------------------------------------------------------------------------

NOTEBOOK_ANA = {
    "total_distance_km": 3_329_137.36524,
    "modelled_types": ("A20N", "A21N", "B763", "B78X"),
    "coverage_pct": 46.8223241538616,
    "co2_modelled_kg": 27_665_590.730244186,
    "co2_extrapolated_kg": 59_086_734.29209386,
    "co2_share_pct": {
        "A20N": 5.366389912864228,
        "A21N": 8.212959715569374,
        "B763": 23.55866331518481,
        "B78X": 9.684330,          # 5 721 951.70 / 59 086 734.29
    },
    "co2_floor_kg": 50_756_046.48593352,
    "optimisation_potential_pct": 14.099083163029086,
    "b78x_co2_kg": 5_721_951.700884949,
    "b78x_swapped_to_b738_kg": 4_857_250.904591219,
    "b78x_swap_saving_pct": 15.11199047975182,
    "b78x_seat_ratio": 330 / 177,
    "a20n_mean_distance_km": 688.7091349602123,
}

# ---------------------------------------------------------------------------
# Piano-X cross-validation (Step 3 of the Safran mini-project)
# ---------------------------------------------------------------------------

# mission -> (aircraft, distance_km, piano_x_co2_kg, icao_cem_co2_kg)
PIANOX_BENCHMARK = [
    ("A388 design range", "A388", 14_075.2, 657_804.0, 723_351.730896),
    ("B763 design range", "B763", 11_241.64, 202_420.0, 220_696.32),
    ("A346 design range", "A346", 14_227.064, 435_782.0, 464_055.32),
    ("B788 mean sector", "B788", 4_729.0, 72_984.0, 87_055.65),
    ("A388 modal sector", "A388", 6_000.0, 257_004.0, 285_969.20),
]

# Relative gap (CEM - PianoX) / PianoX, as reported
PIANOX_RELATIVE_GAP_PCT = {
    "A388 design range": 9.96,
    "B763 design range": 9.03,
    "A346 design range": 6.49,
    "B788 mean sector": 19.28,
    "A388 modal sector": 11.27,
}

# Non-CO2 species by phase, Piano-X (kg per mission)
PIANOX_PHASE_EMISSIONS = {
    ("B788", 4_729.0): {
        "takeoff": {"NOx": 13.7, "HC": 0.07},
        "climb": {"NOx": 50.3, "HC": 0.24},
        "descent": {"NOx": 186.3, "HC": 2.48},
        "total": {"NOx": 252.6, "HC": 3.28},
    },
    ("A388", 6_000.0): {
        "takeoff": {"NOx": 51.7, "HC": 0.0},
        "climb": {"NOx": 289.2, "HC": 0.0},
        "descent": {"NOx": 1_102.5, "HC": 0.0},
        "total": {"NOx": 1_454.7, "HC": 0.39},
    },
}

# Share of total mission emissions per flight phase (A388, 6 000 km)
PIANOX_PHASE_SHARE_PCT = {
    "taxi_takeoff": 3.15,
    "climb": 10.8,
    "cruise": 83.6,
    "descent": 1.2,
    "approach_taxi": 1.2,
}

# (CO2 kg, NOx kg) pairs used for the cross-species regression
PIANOX_CO2_NOX_POINTS = [
    (15_538.0, 42.4),     # Fokker F70
    (72_984.0, 252.6),    # B788, 4 729 km
    (151_719.0, 614.6),   # A306
    (161_486.0, 735.9),   # A346, 5 960 km
    (202_420.0, 757.8),   # B763
    (257_004.0, 1_454.7), # A388, 6 000 km
]

# OLS fitted on the first three long-haul points in the original report
PIANOX_CO2_NOX_OLS = {"slope": 0.006546, "intercept": -257.92}

# Climb-schedule sensitivity, A388 / 6 000 km / 48 900 kg payload,
# "250 kts below FL100" noise restriction active.
PIANOX_CLIMB_SENSITIVITY = {
    "standard_250kcas_M082": {"CO2": 257_610.0, "NOx": 1_436.0, "HC": 0.46},
    "fast_270kcas_M084": {"CO2": 256_791.0, "NOx": 1_437.0, "HC": 0.43},
    "slow_230kcas_M080": {"CO2": 260_052.0, "NOx": 1_449.0, "HC": 0.49},
}

# ---------------------------------------------------------------------------
# Route / geopolitical constraints (Step 4)
# ---------------------------------------------------------------------------

ROUTE_DETOURS = {
    "CDG-HND (AF274, B77W)": {
        "actual_km": 11_979.0,
        "gcd_km": 9_730.0,
        "reason": "Afghanistan overflight avoidance",
    },
    "BUD-ALA (THY1034, B38M)": {
        "actual_km": 5_327.0,     # 1 232 (BUD-IST) + 4 095 (IST-ALA)
        "gcd_km": 4_442.0,
        "reason": "Ukraine airspace closure, routed via Istanbul",
    },
}

ROUTE_DETOUR_IMPACT = {
    "aircraft": "B788",
    "route": "CDG-HND",
    "gcd": {"distance_km": 9_730.0, "CO2": 154_125.0, "NOx": 543.0, "HC": 6.2},
    "actual": {"distance_km": 11_979.0, "CO2": 194_645.0, "NOx": 693.0, "HC": 7.6},
    "delta_co2_kg": 40_520.0,
    "delta_co2_pct": 26.3,
}
