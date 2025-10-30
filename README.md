## Overview
This project implements a **Geant4 simulation** of a **horn focusing system** for the front end of a **Muon Collider**.  
The design models the interaction of an **8 GeV proton beam** on a **graphite target**, generating secondary particles (primarily pions and muons) that are captured and focused by magnetic horn structures.  

The simulation includes:
- **Graphite target** as the primary interaction medium for the proton beam.  
- **Horn focusing structures** modeled with parabolic inner conductors and cylindrical outer conductors.  
- **Current-carrying conductors** producing strong azimuthal magnetic fields to focus charged secondaries.  
- **Downstream detectors** to record particle flux and phase space at defined distances.  
- **Logging and output** of particle six-dimensional vectors (position and momentum) into CSV files.  

The ultimate goal is to study the collection efficiency of muons/pions from the target region, evaluate acceptance into the cooling channel, and optimize horn parameters for the collider front end.

---

## Features
- **Geometry**  
  - World volume (argon fill by default).  
  - Graphite block target.  
  - Two horn systems:  
    - Horn 1: focusing, 0–2000 mm along z.  
    - Horn 2: defocusing, 3000–5000 mm along z.  
  - Inner parabolic conductors with cavities, outer copper conductors, and field regions.  

- **Magnetic Field**  
  - Horn fields implemented via custom `HornField` class.  
  - Configurable current (e.g., ±61 kA or ±150 kA) to adjust focusing strength.  
  - Integration with `G4MagIntegratorDriver` and `G4ChordFinder`.  

- **Beam Setup**  
  - Primary proton beam energy: **8 GeV**.  
  - Target interaction produces hadronic showers (requires FTFP_BERT or equivalent physics list).  

- **Detectors**  
  - Multiple cylindrical disk detectors placed downstream (e.g., at 2.1 m and 5.1 m). And updtaed as per the input parameters 
  - Record entries of selected particles with six-dimensional vector output.  

- **Output**   
  - Merge the CSV output (using the command given below in the `build/` directory) for combining worker outputs. 
  - Analyze the particle output using in the ***6D_merged.csv*** file.  
  - ```bash
    { head -n 1 6D_vector_run0_t0.csv; for f in 6D_vector_run0_t{0..$(nproc)-1}.csv; do tail -n +2 "$f"; done; } > 6D_merged.csv
    ```
    
---

## Command-line options
The simulation binary accepts a small set of command-line flags parsed in `src/Cli.cc`. These directly set horn geometry, field/current and run settings. Flags and meanings:

- --a_mm <double>  
  - Inner parabolic parameter a (mm). Controls the parabola curvature of the inner conductor profile.
- --r_neck_mm <double>  
  - Neck radius of the inner conductor (mm).
- --Rout_mm <double>  
  - Outer conductor radius (mm).
- --zMin_mm <double>  
  - Start z position of the horn segment (mm).
- --zMax_mm <double>  
  - End z position of the horn segment (mm).
- --r_max_mm <double>  
  - Maximum radial aperture used by the geometry or field region (mm).
- --I_A <double>  
  - Current through the horn conductor (A). Use negative sign to flip polarity.
- --spacing_mm <double>  
  - Spacing between elements (mm) — used where the code refers to spacing between horns/structures.
- --n_events <int>  
  - Number of primary events to simulate (overrides defaults).
- --out_dir <string>  
  - Output directory where CSV and log files will be written.

Example:
```bash
./horn_focus --a_mm 0.5 --r_neck_mm 5 --Rout_mm 50 --zMin_mm 0 --zMax_mm 2000 --I_A 61000 --n_events 10000 --out_dir ./out_run1
```

Notes:
- Units are millimeters (mm) for geometry parameters and Amperes (A) for current.
- The CLI parsing in `src/Cli.cc` will throw a runtime error if a flag that expects a value is given without one.

---

## HornXGB_cluster.py — purpose and usage
This repository includes a Python helper script `HornXGB_cluster.py` (see file in repo root or scripts/). The script uses XGBoost to build a model that assists analysis of simulation outputs (for example: classify or cluster particles / horn parameter combinations according to collection efficiency or other figures-of-merit).

Typical pipeline implemented by the script:
1. Load simulation-derived CSV data (e.g., 6D vectors, per-particle features, or per-run summaries).
2. Preprocess features: select/merge columns, handle NaNs, scale or bin variables as needed.
3. Train an XGBoost model (classifier or regressor depending on the target). Common steps:
   - Split dataset into train/test (or k-fold CV).
   - Set XGBoost hyperparameters (learning_rate, max_depth, n_estimators, etc.).
   - Train and evaluate (accuracy, ROC AUC, RMSE or other metrics).
4. Optional: use model outputs to assign cluster/labels, or to predict acceptance/efficiency for new parameter sets.
5. Produce artifacts:
   - Saved model (joblib / XGBoost native format).
   - Evaluation metrics and logs.
   - Feature importance plots and optionally SHAP explanations.
   - Predictions CSV with appended cluster/score columns.

Dependencies (install with pip):
- python >= 3.8
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib (optional)
- joblib (optional)
- shap (optional, for explanation)

Example usage patterns:
- Train and validate:
  ```bash
  python HornXGB_cluster.py --input 6D_merged.csv --target accepted --task classification --out model_run1
  ```
- Predict with a saved model:
  ```bash
  python HornXGB_cluster.py --predict --model model_run1.xgb --input new_sim_data.csv --out predictions.csv
  ```

What to expect in the results:
- A trained XGBoost model that can score particle or run-level entries for the provided target (e.g., identify particles that are within acceptance).
- Feature importance ranking showing which geometry, field or kinematic features most affect the target.
- CSV output with predicted labels/scores that can be merged back into downstream analysis.

If you want, I can:
- Add concrete command-line parsing examples for `HornXGB_cluster.py` (flags and defaults) if you paste the script.
- Add a minimal example script that trains a small XGBoost model on a subset of the simulation CSV for demonstration.
