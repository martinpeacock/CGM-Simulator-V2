# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 23:26:09 2025

@author: martp
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from gox_biosensor_engine import run_gox_simulation


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

# Cylindrical electrode: length = 2 mm, diameter = 0.1 mm
# Side area only (film-coated): A = 2π r L
ELECTRODE_AREA_MM2 = 2 * np.pi * 0.05 * 2   # ≈ 0.628 mm²

def compute_k_glu_mass(film_thickness_um, D_glu=5e-10):
    L_m = film_thickness_um * 1e-6
    if L_m <= 0:
        return 0.0
    return D_glu / (L_m ** 2)

def units_per_electrode_to_mM(E_units, film_thickness_um, kcat_s_inv):
    """
    Convert enzyme loading in Units per electrode → effective mM inside the film,
    using the correct cylindrical electrode geometry.
    """
    if E_units <= 0 or film_thickness_um <= 0 or kcat_s_inv <= 0:
        return 0.0

    # 1 U = 1 µmol/min = 1e-6 mol/min
    rate_mol_per_s = E_units * 1e-6 / 60.0
    n_E_mol = rate_mol_per_s / kcat_s_inv

    # Film volume in liters: (electrode area mm²) * (thickness µm) * 1e-9
    V_film_L = ELECTRODE_AREA_MM2 * film_thickness_um * 1e-9
    if V_film_L <= 0:
        return 0.0

    return (n_E_mol / V_film_L) * 1e3  # M → mM


# ---------------------------------------------------------
# Streamlit layout
# ---------------------------------------------------------

st.set_page_config(layout="wide")
st.title("GOx Biosensor Simulator")

sidebar = st.sidebar
sidebar.header("Simulation Controls")

# =========================================================
# INPUTS
# =========================================================

# -----------------------------
# Glucose protocol
# -----------------------------
sidebar.subheader("Bulk Glucose Steps")

n_steps = sidebar.slider("Number of glucose steps", 1, 10, 6)
step_duration = sidebar.slider("Step duration (s)", 50, 1000, 150, 10)

default_concs = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20]
glucose_steps_mM = []

for i in range(n_steps):
    glucose_steps_mM.append(
        sidebar.number_input(
            f"Step {i+1} glucose (mM)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_concs[i]),
            step=0.5,
            key=f"glu_{i}",
        )
    )

# -----------------------------
# Kinetics
# -----------------------------
sidebar.subheader("Kinetics")

Km_glu_mM = sidebar.slider("Km (mM)", 0.1, 50.0, 10.0, 0.1)
kcat_glu = sidebar.slider("kcat (s⁻¹)", 0.01, 500.0, 100.0, 0.5)
k3 = sidebar.slider("k3 (M⁻¹ s⁻¹)", 0.01, 1e4, 100.0, 1.0)

Km_glu_M = Km_glu_mM * 1e-3
km1 = 1.0
k2 = kcat_glu
k1 = (km1 + k2) / Km_glu_M if Km_glu_M > 0 else 0.0

# -----------------------------
# Enzyme loading
# -----------------------------
sidebar.subheader("Enzyme Loading")

E_units = sidebar.slider("GOx loading (U)", 0.01, 10.0, 1.0, 0.01)
film_thickness_um = sidebar.slider("Film thickness (µm)", 1.0, 200.0, 20.0, 1.0)
k_glu_mass = compute_k_glu_mass(film_thickness_um)
sidebar.write(f"Effective glucose mass transfer rate = {k_glu_mass:.3e} s⁻¹")
E_tot_mM_sim = units_per_electrode_to_mM(E_units, film_thickness_um, kcat_glu)
sidebar.write(f"Effective [GOx] = {E_tot_mM_sim:.3g} mM")

# -----------------------------
# Oxygen
# -----------------------------
sidebar.subheader("Oxygen")

O2_ppm = sidebar.selectbox("Initial O₂ (ppm)", [0,1,2,3,4,5,6], index=6)
O2_bath_ppm = sidebar.selectbox("Bath O₂ (ppm)", [0,1,2,3,4,5,6], index=6)
O2_mode = sidebar.selectbox("O₂ mode", ["closed", "well-aerated"], index=0)

# -----------------------------
# Run button
# -----------------------------
run_sim = sidebar.button("Run Simulation")

# =========================================================
# SIMULATION
# =========================================================

if run_sim:

    result = run_gox_simulation(
        k1=k1,
        km1=km1,
        k2=k2,
        k3=k3,
        E_tot_mM=E_tot_mM_sim,
        O2_mode=O2_mode,
        O2_0_ppm=O2_ppm,
        O2_bath_ppm=O2_bath_ppm,
        glucose_steps_mM=glucose_steps_mM,
        step_duration_s=step_duration,
        n_points=2000,
        k_glu_mass=k_glu_mass,
    )

    # Extract outputs
    t = result["t"]
    ES = result["ES_M"]
    Ered = result["Ered_M"]
    Gluconolactone = result["P_M"] * 1e3
    H2O2 = result["H2O2_M"] * 1e3
    O2 = result["O2_M"] * 1e3  # convert M → µM
    glucose_sample_mM = result["glucose_sample_mM"]
    glucose_film_mM = result["glucose_film_mM"]
    current = result["current_uA"]

    # =====================================================
    # PLOTS
    # =====================================================

    # -----------------------------
    # Film species
    # -----------------------------
    st.subheader("Film Species")
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(t, Gluconolactone, label="Gluconolactone (µM)")
    ax1.plot(t, H2O2, label="H₂O₂ (mM)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Concentration (mM)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t, O2, color="red", linestyle="--", label="O₂ (mM)")
    ax2.set_ylabel("O₂ (mM)", color="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    st.pyplot(fig1)

   # -----------------------------
    # Glucose in sample vs film + O2 + H2O2
    # -----------------------------
    st.subheader("Glucose, Oxygen, and Hydrogen Peroxide")
    
    fig2, ax_glu = plt.subplots(figsize=(8,3))
    
    # --- Y1 axis (left): Glucose ---
    ax_glu.plot(t, glucose_sample_mM, color="tab:blue", label="Glucose in sample (mM)")
    ax_glu.plot(t, glucose_film_mM, color="tab:orange", linestyle="--", label="Glucose in film (mM)")
    
    ax_glu.set_xlabel("Time (s)")
    ax_glu.set_ylabel("Glucose (mM)")
    ax_glu.grid(True)
    
    # --- Y2 axis (right): O2 + H2O2 ---
    ax2 = ax_glu.twinx()
    
    ax2.plot(t, O2, color="tab:red", linestyle="-.", label="Oxygen (µM)")
    ax2.plot(t, H2O2, color="tab:green", linestyle=":", label="Hydrogen peroxide (µM)")
    
    ax2.set_ylabel("Oxygen / H₂O₂ (µM)")
    
    # --- Combined legend ---
    # We need to merge handles from both axes
    lines1, labels1 = ax_glu.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_glu.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    
    st.pyplot(fig2)
    # -----------------------------
    # Current
    # -----------------------------
    st.subheader("Current")
    fig3, ax_cur = plt.subplots(figsize=(8,3))
    ax_cur.plot(t, current, color="black")
    ax_cur.set_xlabel("Time (s)")
    ax_cur.set_ylabel("Current (µA)")
    ax_cur.grid(True)
    st.pyplot(fig3)