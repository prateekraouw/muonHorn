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
  - Multiple cylindrical disk detectors placed downstream (e.g., at 2.1 m and 5.1 m).  
  - Record entries of selected particles with six-dimensional vector output.  

- **Output**   
  - Merge the CSV output (using the command given below in the `build/` directory) for combining worker outputs. 
  - Analyze the particle output using in the ***6D_merged.csv*** file.  
  - ```bash
    { head -n 1 6D_vector_run0_t0.csv; for f in 6D_vector_run0_t{0..$(nproc)-1}.csv; do tail -n +2 "$f"; done; } > 6D_merged.csv
    

---

## Building
### Requirements
- [Geant4 11.x](https://geant4.web.cern.ch) with multithreading support enabled.  
- CMake ≥ 3.16.  
- A C++17 compiler (GCC/Clang/MSVC).  

### Build Instructions
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./horn_focus # Runs in GUI mode
#======OR==========
./horn_focus run.mac # Runs in batch mode
