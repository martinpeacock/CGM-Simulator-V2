# GOx Biosensor Simulator — README

## Overview

The GOx Biosensor Simulator is an interactive Streamlit application designed to help students, engineers, and researchers explore the behavior of glucose oxidase (GOx) enzyme films under varying glucose and oxygen conditions. The simulator began as a mechanistic ODE model but evolved into a clean, intuitive, educational tool through a series of refinements described below.

The app models:

- Glucose oxidation by GOx
- Oxygen consumption
- Hydrogen peroxide production
- Amperometric current
- Oxygen‑limited vs oxygen‑replenished behavior
- Effects of enzyme loading, film thickness, and glucose steps

The goal is to provide a learning‑focused, approachable interface for understanding biosensor kinetics.

---

## ✨ Key Improvements Made During Development

Over time, the simulator underwent a series of conceptual, mathematical, and UI refinements. These changes transformed it from a dense mechanistic model into a clear, pedagogical tool.

---

## 1. Model Simplifications (Mechanistic → Educational)

### ✔️ 1.1 Replaced mechanistic species names with intuitive ones
To make the plots and UI more readable:

- “Gluconolactone” → **P**
- “Reduced enzyme” → **E_red**
- “ES complex” → **ES**
- Glucose tracked as **glucose_mM**

This made the outputs easier to interpret for learners.

---

### ✔️ 1.2 Enzyme loading changed from molarity → Units (U)
Originally the model required enzyme concentration in molarity, film volume, and electrode area.  
We replaced all of this with:

- **GOx loading (Units per electrode)**

Internally, the simulator converts:
