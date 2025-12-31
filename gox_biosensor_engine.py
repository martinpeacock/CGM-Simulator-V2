# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 23:24:15 2025

@author: martp
"""

# gox_biosensor_engine.py

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

# Henry's law-based conversion for O2 ppm → M
# 1 ppm ~ 1 mg/L; O2 MW = 32 g/mol → 1 ppm ≈ 3.125e-5 M
O2_PPM_TO_M = 3.125e-5  # rough teaching value, good enough for simulation

# Oxygen mass transfer coefficient for well-aerated mode (s⁻¹)
K_O2_MASS = 0.05  # adjustable educational parameter

# Scaling factor for current (arbitrary units)
CURRENT_SCALE = 1.0


# ---------------------------------------------------------
# Glucose protocol helper
# ---------------------------------------------------------

def build_glucose_profile(glucose_steps_mM, step_duration_s, n_points):
    """
    Build time array and corresponding glucose concentration in M
    from a step protocol defined in mM.
    """
    glucose_steps_mM = list(glucose_steps_mM)
    n_steps = len(glucose_steps_mM)
    t_end = n_steps * step_duration_s

    t = np.linspace(0.0, t_end, n_points)

    # For each time, determine which step we are in
    glucose_mM = np.zeros_like(t)
    for i, ti in enumerate(t):
        step_index = min(int(ti // step_duration_s), n_steps - 1)
        glucose_mM[i] = glucose_steps_mM[step_index]

    glucose_M = glucose_mM * 1e-3  # mM → M

    return t, glucose_M, glucose_mM


# ---------------------------------------------------------
# ODE system
# ---------------------------------------------------------

def gox_ode_system(t, y, params):
    """
    ODEs for reduced GOx ping-pong mechanism.

    y = [ES, E_red, P, O2, H2O2]

    params dict:
        k1, km1, k2, k3
        E_tot_M
        O2_mode
        O2_bath_M
        t_grid, glu_M_grid
    """
    ES, E_red, P, O2, H2O2 = y

    k1 = params["k1"]
    km1 = params["km1"]
    k2 = params["k2"]
    k3 = params["k3"]
    E_tot_M = params["E_tot_M"]
    O2_mode = params["O2_mode"]
    O2_bath_M = params["O2_bath_M"]
    t_grid = params["t_grid"]
    glu_M_grid = params["glu_M_grid"]

    # Enzyme conservation
    E_free = E_tot_M - ES - E_red
    if E_free < 0:
        E_free = 0.0

    # Glucose at time t via interpolation
    glu_M = np.interp(t, t_grid, glu_M_grid)

    # Reactions:
    # E + Glu ⇌ ES → E_red + P
    v_bind = k1 * E_free * glu_M
    v_unbind = km1 * ES
    v_cat = k2 * ES

    # E_red + O2 → E + H2O2
    v_ox = k3 * E_red * O2

    # d(ES)/dt
    dES = v_bind - v_unbind - v_cat

    # d(E_red)/dt
    dE_red = v_cat - v_ox

    # dP/dt (gluconolactone)
    dP = v_cat

    # dH2O2/dt
    dH2O2 = v_ox

    # O2 dynamics
    if O2_mode == "closed":
        J_O2 = 0.0
    else:  # "well-aerated"
        J_O2 = K_O2_MASS * (O2_bath_M - O2)

    dO2 = -v_ox + J_O2

    return [dES, dE_red, dP, dO2, dH2O2]


# ---------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------

def run_gox_simulation(
    k1,
    km1,
    k2,
    k3,
    E_tot_mM,
    O2_mode,
    O2_0_ppm,
    O2_bath_ppm,
    glucose_steps_mM,
    step_duration_s,
    n_points=2000,
):
    """
    Run GOx biosensor simulation with educational refinements:

    - Enzyme input in mM (typically derived from Units in UI)
    - Oxygen inputs in ppm (converted internally to M)
    - Glucose protocol as steps in mM
    - Reduced ping-pong mechanism
    - Current ∝ d[H2O2]/dt

    Returns dict with keys:
        "t"
        "ES_M"
        "Ered_M"
        "P_M"
        "O2_M"
        "H2O2_M"
        "glucose_M"
        "glucose_mM"
        "current_AU"
    """

    # Time grid and glucose profile
    t, glucose_M, glucose_mM = build_glucose_profile(
        glucose_steps_mM=glucose_steps_mM,
        step_duration_s=step_duration_s,
        n_points=n_points,
    )

    # Enzyme total in M
    E_tot_M = E_tot_mM * 1e-3  # mM → M

    # Oxygen in M
    O2_0_M = O2_0_ppm * O2_PPM_TO_M
    O2_bath_M = O2_bath_ppm * O2_PPM_TO_M

    # Initial conditions
    ES0 = 0.0
    Ered0 = 0.0
    P0 = 0.0
    O20 = O2_0_M
    H2O20 = 0.0

    y0 = [ES0, Ered0, P0, O20, H2O20]

    params = {
        "k1": k1,
        "km1": km1,
        "k2": k2,
        "k3": k3,
        "E_tot_M": E_tot_M,
        "O2_mode": O2_mode,
        "O2_bath_M": O2_bath_M,
        "t_grid": t,
        "glu_M_grid": glucose_M,
    }

    # Integrate ODEs
    sol = solve_ivp(
        fun=lambda tt, yy: gox_ode_system(tt, yy, params),
        t_span=(t[0], t[-1]),
        y0=y0,
        t_eval=t,
        method="LSODA",
        rtol=1e-6,
        atol=1e-9,
    )

    ES = sol.y[0, :]
    E_red = sol.y[1, :]
    P = sol.y[2, :]
    O2 = sol.y[3, :]
    H2O2 = sol.y[4, :]

    # Approximate current ∝ d[H2O2]/dt
    dH2O2_dt = np.gradient(H2O2, t)
    current_AU = CURRENT_SCALE * dH2O2_dt

    result = {
        "t": t,
        "ES_M": ES,
        "Ered_M": E_red,
        "P_M": P,
        "O2_M": O2,
        "H2O2_M": H2O2,
        "glucose_M": glucose_M,
        "glucose_mM": glucose_mM,
        "current_AU": current_AU,
    }

    return result