"""NCVD toolbox: Nonlinear Characteristic Value Decomposition.

    X1 = U Phi^H ,   X2 = U Lam Phi^H + R

Core
    ncvd, ncvd_pair, decompose, NCVDResult
Modal quantities
    modal_parameters, physical_modes, realify, mpc, mac, mac_matrix,
    mac_phase_matched, relative_error, match_modes, poles_and_shapes
Kinematics
    state_from_displacement, deriv_filter, correct_poles
Nonlinear tracking
    track, window_edges, band_frequency
Ensemble
    ncvd_ensemble
Reduced-order models
    rom.robustness_basis, rom.deflate, rom.escape, rom.closure, rom.project
Benchmarks
    benchmarks.gcvd, benchmarks.dmd, benchmarks.itd
Test systems
    systems.chain, systems.paper_chain10, systems.paper_chain3,
    systems.duffing_chain, systems.simulate_linear, systems.simulate_nonlinear,
    systems.add_noise
"""
from .core import (ncvd, ncvd_pair, decompose, NCVDResult, DEFAULT_RANK,
                   balanced_quotient, regression_operator, auto_time_scale, noise_variance)
from .modal import (modal_parameters, physical_modes, realify, mpc, mac, mac_matrix,
                    mac_phase_matched, relative_error, normalize_shape, match_modes,
                    poles_and_shapes)
from .kinematics import state_from_displacement, deriv_filter, correct_poles, DerivFilter
from .tracking import track, window_edges, band_frequency
from .ensemble import ncvd_ensemble
from . import rom, benchmarks, systems

__version__ = '1.0.0'
__all__ = ['ncvd', 'ncvd_pair', 'decompose', 'NCVDResult', 'DEFAULT_RANK',
           'balanced_quotient', 'regression_operator', 'auto_time_scale', 'noise_variance',
           'modal_parameters', 'physical_modes', 'realify', 'mpc', 'mac', 'mac_matrix',
           'mac_phase_matched', 'relative_error', 'normalize_shape', 'match_modes',
           'poles_and_shapes', 'state_from_displacement', 'deriv_filter',
           'correct_poles', 'DerivFilter', 'track', 'window_edges', 'band_frequency',
           'ncvd_ensemble', 'rom', 'benchmarks', 'systems']
