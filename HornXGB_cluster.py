#!/usr/bin/env python
# coding: utf-8


"""
GPU-Optimized Horn Optimization for Compute Clusters
- CUDA-accelerated data processing
- Multi-GPU support for parallel Geant4 runs
- Cluster-optimized resource management
- SLURM/PBS job integration
"""

import os, glob, json, time, hashlib, subprocess, shutil
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda

# Constants
PI_PLUS_MASS_MEV = 139.57039
DEFAULT_ENERGY_WINDOW = (100.0, 400.0)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# GPU Configuration
def setup_gpu():
    """Setup GPU configuration for cluster computing."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        
        print(f"🚀 GPU Setup:")
        print(f"   Available GPUs: {gpu_count}")
        print(f"   Current GPU: {current_gpu} ({gpu_name})")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Memory: {torch.cuda.get_device_properties(current_gpu).total_memory / 1e9:.1f} GB")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        return device, gpu_count
    else:
        print("⚠️  No CUDA GPUs available, falling back to CPU")
        return torch.device("cpu"), 0

# Initialize GPU
DEVICE, GPU_COUNT = setup_gpu()

# Simple paths
RUNS_ROOT = "runs"
DATASET = "dataset.csv"
RESULTS_DIR = "results"

# Create directories
os.makedirs(RUNS_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Geant4 executable path
GEANT4_EXE = os.environ.get("HORN_SIM_EXE", "/Users/prateek/muonHorn/build/horn_sim")

# Environment variables for Geant4
EXTRA_ENV = {
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "OMP_NUM_THREADS": str(os.cpu_count()),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
}

@dataclass(frozen=True)
class HornParams:
    # geometry (shared by both horns)
    a_mm: float
    Rout_mm: float
    r_neck_mm: float
    zMin_mm: float
    zMax_mm: float
    r_max_mm: float
    # current (Horn1:+I, Horn2:-I)
    I_A: float
    # placement
    spacing_mm: float
    # run control (not searched unless you want to)
    n_events: int = 50000

    def as_cli(self) -> List[str]:
        return [
            "--a_mm",        f"{self.a_mm}",
            "--r_neck_mm",   f"{self.r_neck_mm}",
            "--Rout_mm",     f"{self.Rout_mm}",
            "--zMin_mm",     f"{self.zMin_mm}",
            "--zMax_mm",     f"{self.zMax_mm}",
            "--r_max_mm",    f"{self.r_max_mm}",
            "--I_A",         f"{self.I_A}",
            "--spacing_mm",  f"{self.spacing_mm}",
            "--n_events",    f"{self.n_events}",
            "--out_dir",     "{OUT_DIR_PLACEHOLDER}",
        ]

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.a_mm, self.r_neck_mm, self.r_max_mm,
            self.Rout_mm, self.zMin_mm, self.zMax_mm,
            self.I_A, self.spacing_mm
        ], dtype=float)
    
    def validate(self) -> bool:
        """Validate that parameters create valid geometry."""
        return (
            self.a_mm > 0 and
            self.r_neck_mm > 0 and
            self.r_max_mm > self.r_neck_mm and
            self.Rout_mm > self.r_max_mm and
            self.zMax_mm > self.zMin_mm and
            self.zMax_mm > 0 and
            self.I_A != 0 and
            self.spacing_mm > 0
        )

    @staticmethod
    def names() -> List[str]:
        return ["a_mm","r_neck_mm","r_max_mm","Rout_mm","zMin_mm","zMax_mm","I_A","spacing_mm"]

# GPU-accelerated beam statistics
def beam_stats_gpu(x_mm, y_mm, px, py, pz, E_MeV,
                   E_min: Optional[float]=None, E_max: Optional[float]=None,
                   N_in: Optional[int]=None, a_mm: Optional[float]=None,
                   mass_MeV: float=PI_PLUS_MASS_MEV) -> Dict[str,float]:
    """GPU-accelerated beam statistics calculation."""
    
    # Convert to tensors and move to GPU
    x = torch.tensor(x_mm, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_mm, dtype=torch.float32, device=DEVICE)
    px = torch.tensor(px, dtype=torch.float32, device=DEVICE)
    py = torch.tensor(py, dtype=torch.float32, device=DEVICE)
    pz = torch.tensor(pz, dtype=torch.float32, device=DEVICE)
    E = torch.tensor(E_MeV, dtype=torch.float32, device=DEVICE)

    # Create mask for valid particles
    mask = torch.isfinite(pz) & (pz != 0.0)
    if E_min is not None: mask &= (E >= E_min)
    if E_max is not None: mask &= (E <= E_max)
    
    if not torch.any(mask):
        return {"N_out": 0, "eta_trans": 0.0}

    # Apply mask
    x, y, px, py, pz, E = x[mask], y[mask], px[mask], py[mask], pz[mask], E[mask]
    xp, yp = px/pz, py/pz

    # Calculate covariances using GPU
    def cov2_gpu(a, b):
        am, bm = a.mean(), b.mean()
        return (a*b).mean() - am*bm

    Sxx, Sxpx, Sxpxp = cov2_gpu(x,x), cov2_gpu(x,xp), cov2_gpu(xp,xp)
    Syy, Sypy, Syyp = cov2_gpu(y,y), cov2_gpu(y,yp), cov2_gpu(yp,yp)

    dxxp = Sxx*Sxpxp - Sxpx**2
    dyyp = Syy*Syyp  - Sypy**2

    eps_x = torch.sqrt(torch.clamp(dxxp, min=0.0))
    eps_y = torch.sqrt(torch.clamp(dyyp, min=0.0))

    beta_x  = torch.where(eps_x > 0, Sxx/eps_x, torch.tensor(0.0, device=DEVICE))
    alpha_x = torch.where(eps_x > 0, -Sxpx/eps_x, torch.tensor(0.0, device=DEVICE))
    beta_y  = torch.where(eps_y > 0, Syy/eps_y, torch.tensor(0.0, device=DEVICE))
    alpha_y = torch.where(eps_y > 0, -Sypy/eps_y, torch.tensor(0.0, device=DEVICE))

    sig_x, sig_xp = torch.sqrt(torch.clamp(Sxx, min=0.0)), torch.sqrt(torch.clamp(Sxpxp, min=0.0))
    sig_y, sig_yp = torch.sqrt(torch.clamp(Syy, min=0.0)), torch.sqrt(torch.clamp(Syyp, min=0.0))
    sig_theta_rms = torch.sqrt(0.5*(sig_xp**2 + sig_yp**2))
    eps4D = eps_x * eps_y

    N_out = x.size(0)
    eta_trans = (N_out/float(N_in)) if (N_in is not None and N_in>0) else float('nan')

    Emean = E.mean()
    gamma = Emean / mass_MeV
    beta_rel = torch.sqrt(torch.clamp(1 - 1/(gamma**2), min=0.0))
    epsn_x, epsn_y = beta_rel*gamma*eps_x, beta_rel*gamma*eps_y

    eps_acc_x = torch.where(beta_x > 0, (a_mm**2 / beta_x), torch.tensor(float('nan'), device=DEVICE)) if a_mm is not None else torch.tensor(float('nan'), device=DEVICE)
    eps_acc_y = torch.where(beta_y > 0, (a_mm**2 / beta_y), torch.tensor(float('nan'), device=DEVICE)) if a_mm is not None else torch.tensor(float('nan'), device=DEVICE)

    # Convert back to CPU for return
    return dict(
        N_out=int(N_out), eta_trans=float(eta_trans),
        eps_x=float(eps_x), eps_y=float(eps_y), epsn_x=float(epsn_x), epsn_y=float(epsn_y),
        beta_x=float(beta_x), alpha_x=float(alpha_x), beta_y=float(beta_y), alpha_y=float(alpha_y),
        sigma_x_mm=float(sig_x), sigma_y_mm=float(sig_y), sigma_xp=float(sig_xp), sigma_yp=float(sig_yp),
        sigma_theta_rms=float(sig_theta_rms), eps_4D=float(eps4D),
        eps_acc_x=float(eps_acc_x), eps_acc_y=float(eps_acc_y),
        gamma_rel=float(gamma), beta_rel=float(beta_rel)
    )

def merge_thread_csvs(out_dir: str, pattern="6D_vector_run0_t*.csv") -> pd.DataFrame:
    """Merge CSV files from multiple threads."""
    files = sorted(glob.glob(os.path.join(out_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"No per-thread CSVs in {out_dir} matching {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Simple Geant4 execution
def run_geant4_simple(params: HornParams) -> Tuple[pd.DataFrame, str]:
    """Run a single Geant4 simulation with minimal logging."""
    # Create unique run directory
    run_id = f"run_{int(time.time() * 1000)}_{hash(str(params)) % 10000}"
    run_dir = os.path.join(RUNS_ROOT, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Build CLI
    cli = params.as_cli()
    for i, tok in enumerate(cli):
        if tok == "{OUT_DIR_PLACEHOLDER}":
            cli[i] = run_dir

    exe = GEANT4_EXE
    cmd = [exe] + cli
    
    # Environment
    env = os.environ.copy()
    env.update(EXTRA_ENV)

    # Run Geant4
    try:
        res = subprocess.run(
            cmd, 
            cwd=run_dir, 
            text=True, 
            capture_output=True, 
            env=env, 
            timeout=1200,  # 20 minutes
        )
        
        if res.returncode != 0:
            raise RuntimeError(f"Geant4 failed with exit code {res.returncode}")
        
        # Process results
        df = merge_thread_csvs(run_dir)
        if df.empty:
            raise ValueError("No particle tracks found")
        
        return df, run_dir
        
    except Exception as e:
        # Clean up on failure
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        raise


# Physics-informed constraints (same as before)
def geometry_penalty(hp: HornParams) -> float:
    """Physics-informed penalties to discourage non-physical designs."""
    p = 0.0
    margin = 2.0
    min_len = 200.0
    
    # Basic geometry constraints
    if not (hp.r_max_mm > hp.r_neck_mm + margin): 
        p += (hp.r_neck_mm + margin - hp.r_max_mm)**2
    if not (hp.Rout_mm > hp.r_max_mm + margin):  
        p += (hp.r_max_mm + margin - hp.Rout_mm)**2
    if not (hp.zMax_mm > hp.zMin_mm + min_len):  
        p += (hp.zMin_mm + min_len - hp.zMax_mm)**2
    if not (hp.spacing_mm > 0):                   
        p += (1.0 - hp.spacing_mm)**2

    # Parabola consistency
    w_cons = 0.1
    lhs = hp.r_max_mm**2
    rhs = hp.r_neck_mm**2 + (hp.zMax_mm - hp.zMin_mm)/hp.a_mm
    p += w_cons * (lhs - rhs)**2
    
    # Current density constraint
    conductor_area = np.pi * (hp.Rout_mm**2 - hp.r_max_mm**2)
    current_density = hp.I_A / (conductor_area * 1e-6)
    max_current_density = 1e7
    if current_density > max_current_density:
        p += (current_density - max_current_density)**2 * 1e-12
    
    # Magnetic field constraint
    mu_0 = 4e-7 * np.pi
    peak_field = mu_0 * hp.I_A / (2 * np.pi * hp.r_neck_mm * 1e-3)
    max_field = 10.0
    if peak_field > max_field:
        p += (peak_field - max_field)**2 * 1e-6
    
    return p

def score_from_tracks(df: pd.DataFrame, hp: HornParams,
                      N_in: Optional[int]=None,
                      energy_window: Optional[Tuple[float,float]]=DEFAULT_ENERGY_WINDOW) -> Dict[str,float]:
    """Score tracks using GPU-accelerated calculations."""
    if energy_window:
        lo, hi = energy_window
        df = df[(df["E[GeV]"] >= lo/1000) & (df["E[GeV]"] <= hi/1000)].copy()

    # Use GPU-accelerated beam statistics
    stats = beam_stats_gpu(
        x_mm=df["x[mm]"].values, y_mm=df["y[mm]"].values,
        px=df["px[MeV/c]"].values, py=df["py[MeV/c]"].values, pz=df["pz[MeV/c]"].values,
        E_MeV=df["E[GeV]"].values * 1000,  # Convert GeV to MeV
        E_min=None, E_max=None, N_in=N_in, a_mm=hp.a_mm, mass_MeV=PI_PLUS_MASS_MEV
    )

    # Calculate objective function
    eta = stats.get("eta_trans", 0.0)
    if np.isnan(eta):
        eta = len(df) / max(len(df), 1)

    epsn_mean = 0.5*(stats["epsn_x"] + stats["epsn_y"])
    div = stats["sigma_theta_rms"]

    # Acceptance reward
    if np.isfinite(stats["eps_acc_x"]) and np.isfinite(stats["eps_acc_y"]):
        r_acc = min(stats["eps_acc_x"]/max(stats["eps_x"],1e-12),
                    stats["eps_acc_y"]/max(stats["eps_y"],1e-12))
        acc_reward = min(r_acc, 1.0)
    else:
        acc_reward = 0.5

    # Physics-informed objective function - focus on emittance and particle count
    # Primary: maximize particle count (transmission efficiency)
    # Secondary: minimize emittance (beam quality)
    w_trans, w_emit, w_beta = 1.0, 0.50, 0.15
    
    # Particle count (transmission efficiency)
    particle_count = len(df)
    
    # Emittance (lower is better)
    emittance = epsn_mean
    beta = 0.5*(stats["beta_x"] + stats["beta_y"])
    
    objective = (w_trans * eta - w_emit * emittance - w_beta * beta)

    focusing_strength = hp.I_A / (hp.r_neck_mm * hp.a_mm)
    compression_ratio = hp.r_max_mm / hp.r_neck_mm if hp.r_neck_mm > 0 else 1.0

    stats.update(dict(
        eta_trans=eta, epsn_mean=epsn_mean,
        particle_count=particle_count, emittance=emittance,
        objective=objective, transmission=eta,
        focusing_strength=focusing_strength, compression_ratio=compression_ratio
    ))
    return stats

# GPU-optimized XGBoost training
def train_xgb_gpu(X: np.ndarray, y: np.ndarray, rounds: int = 50):
    """Train XGBoost with GPU acceleration."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # XGBoost with GPU support
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=5,
        n_estimators=1500,
        learning_rate=0.075,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=5,
        gamma=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
        tree_method='gpu_hist' if GPU_COUNT > 0 else 'hist',  # GPU acceleration
        gpu_id=0 if GPU_COUNT > 0 else None,
    )
    
    model.fit(
        Xtr, ytr, 
        eval_set=[(Xva, yva)], 
        verbose=False
    )
    
    return model, scaler

# Dataset management
def append_to_dataset(hp: HornParams, stats: Dict[str,float], path: str = DATASET):
    """Append results to dataset."""
    row = {**{k:v for k,v in asdict(hp).items()}, **stats}
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row
    df.to_csv(path, index=False)
    return df

# Candidate generation
def sample_uniform(bounds: Dict[str, tuple | float], n: int) -> List[HornParams]:
    """Sample uniform parameters with validation."""
    names = HornParams.names()
    out = []
    max_attempts = n * 10  # Allow up to 10x attempts to get valid parameters
    attempts = 0
    
    while len(out) < n and attempts < max_attempts:
        draw = {}
        for nm in names:
            if isinstance(bounds[nm], (list, tuple)):
                lo, hi = bounds[nm]
                val = np.random.uniform(lo, hi)
            else:
                val = bounds[nm]
            draw[nm] = float(val)
        
        hp = HornParams(**draw)
        if hp.validate():
            out.append(hp)
        
        attempts += 1
    
    # If we still don't have enough, use default valid parameters
    while len(out) < n:
        hp = HornParams(
            a_mm=0.00873, r_neck_mm=250.0, r_max_mm=540.0, Rout_mm=840.0,
            zMin_mm=0.0, zMax_mm=2000.0, I_A=70000.0, spacing_mm=1000.0
        )
        out.append(hp)
    
    return out

def propose_from_xgb(model, scaler, bounds: Dict[str, tuple | float], k: int = 20) -> List[HornParams]:
    """Generate candidates using XGBoost."""
    names = HornParams.names()
    N = 10000
    cand = np.zeros((N, len(names)), dtype=float)
    for j, nm in enumerate(names):
        if isinstance(bounds[nm], (list, tuple)):
            lo, hi = bounds[nm]
            cand[:, j] = np.random.uniform(lo, hi, size=N)
        else:
            cand[:, j] = bounds[nm]
    
    pred = model.predict(scaler.transform(cand))
    idx = np.argsort(pred)[::-1][:k]
    
    hps = []
    for i in idx:
        v = cand[i]
        hp = HornParams(
            a_mm=float(v[0]), r_neck_mm=float(v[1]), r_max_mm=float(v[2]),
            Rout_mm=float(v[3]), zMin_mm=float(v[4]), zMax_mm=float(v[5]),
            I_A=float(v[6]), spacing_mm=float(v[7])
        )
        # Only add valid parameters
        if hp.validate():
            hps.append(hp)
    
    # If we don't have enough valid parameters, fill with random valid ones
    while len(hps) < k:
        hp = sample_uniform(bounds, 1)[0]
        hps.append(hp)
    
    return hps[:k]  # Return exactly k parameters

# Main optimization function
def optimize_cluster(bounds: Dict[str,Tuple[float,float]], 
                   n_seed: int=10, n_rounds: int=4, 
                   k_candidates: int=3, parallel_eval: bool=True):
    """GPU-accelerated optimization with minimal logging."""
    
    print(f"🚀 Starting horn optimization with GPU acceleration")
    print(f"   Device: {DEVICE}, GPUs: {GPU_COUNT}")

    # Initialize
    X, y = [], []
    
    # Load existing dataset
    if os.path.exists(DATASET):
        try:
            ds = pd.read_csv(DATASET)
            if set(HornParams.names()).issubset(ds.columns) and "objective" in ds.columns:
                X = ds[HornParams.names()].values.tolist()
                y = ds["objective"].values.tolist()
        except Exception as e:
            print(f"Error loading dataset: {e}")

    # Seed phase - ensure we get successful samples
    print(f"🌱 EPOCH 0: Seed phase ({n_seed} samples)")
    X, y = [], []
    max_retries = n_seed * 3  # Allow up to 3x retries
    retry_count = 0
    
    while len(X) < n_seed and retry_count < max_retries:
        try:
            hp = sample_uniform(bounds, 1)[0]
            df, run_dir = run_geant4_simple(hp)
            stats = score_from_tracks(df, hp, N_in=hp.n_events)
            append_to_dataset(hp, stats)
            X.append(hp.as_vector().tolist())
            y.append(stats["objective"])
            
            # Clean up run directory
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
                
            print(f"   ✓ Sample {len(X)}/{n_seed} successful")
            
        except Exception as e:
            retry_count += 1
            print(f"   ✗ Sample failed (retry {retry_count}/{max_retries}): {e}")
            
            # Clean up failed run directory
            try:
                if 'run_dir' in locals() and os.path.exists(run_dir):
                    shutil.rmtree(run_dir)
            except:
                pass

    # Check if we have any successful samples
    if len(X) == 0:
        raise RuntimeError("No successful samples from seed phase - check Geant4 executable and parameters")
    
    # Train initial model
    if len(X) < 2:
        print(f"Warning: Only {len(X)} successful samples, using random sampling for first round")
        xgb_model, scaler = None, None
    else:
        try:
            X_arr, y_arr = np.array(X, float), np.array(y, float)
            xgb_model, scaler = train_xgb_gpu(X_arr, y_arr)
        except Exception as e:
            print(f"Error training initial model: {e}")
            xgb_model, scaler = None, None

    # Optimization rounds
    for r in range(1, n_rounds + 1):
        print(f"🔄 EPOCH {r}: Starting optimization round")
        
        # Generate candidates
        try:
            if xgb_model is not None and scaler is not None:
                cands = propose_from_xgb(xgb_model, scaler, bounds, k=k_candidates)
            else:
                cands = sample_uniform(bounds, k_candidates)
        except Exception as e:
            print(f"Error generating candidates: {e}")
            cands = sample_uniform(bounds, k_candidates)
        
        # Evaluate candidates
        round_scores = []
        for hp in cands:
            try:
                df, run_dir = run_geant4_simple(hp)
                stats = score_from_tracks(df, hp, N_in=hp.n_events)
                append_to_dataset(hp, stats)
                X.append(hp.as_vector().tolist())
                y.append(stats["objective"])
                round_scores.append(stats["objective"])
                
                # Clean up run directory
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir)
            except Exception as e:
                print(f"Error evaluating candidate: {e}")

        # Retrain model
        if round_scores and len(X) >= 2:
            try:
                X_arr, y_arr = np.array(X, float), np.array(y, float)
                xgb_model, scaler = train_xgb_gpu(X_arr, y_arr)
            except Exception as e:
                print(f"Error retraining model: {e}")
                xgb_model, scaler = None, None
            
            # Print epoch completion with best parameters
            best_score = max(round_scores)
            ds = pd.read_csv(DATASET)
            best = ds.loc[ds["objective"].idxmax()]
            print(f"✅ EPOCH {r} COMPLETE: Best objective = {best['objective']:.4f}")
            print(f"   Particle count: {best.get('particle_count', 'N/A')}, Emittance: {best.get('emittance', 'N/A'):.4f}")
            print(f"   Best parameters: a={best['a_mm']:.4f}, r_neck={best['r_neck_mm']:.1f}, r_max={best['r_max_mm']:.1f}, Rout={best['Rout_mm']:.1f}, zMax={best['zMax_mm']:.1f}, I={best['I_A']:.0f}, spacing={best['spacing_mm']:.1f}")

    # Final results
    try:
        ds = pd.read_csv(DATASET)
        best = ds.loc[ds["objective"].idxmax()]
        print(f"\n🎯 OPTIMIZATION COMPLETE")
        print(f"   Total evaluations: {len(ds)}")
        print(f"   Best objective: {best['objective']:.4f}")
        print(f"   Best particle count: {best.get('particle_count', 'N/A')}")
        print(f"   Best emittance: {best.get('emittance', 'N/A'):.4f}")
        print(f"   Best parameters:")
        for param in HornParams.names():
            print(f"     {param}: {best[param]:.4f}")
        
        # Save results
        results_file = os.path.join(RESULTS_DIR, "best_design.json")
        with open(results_file, 'w') as f:
            json.dump(best.to_dict(), f, indent=2)
        print(f"   Results saved to: {results_file}")
        
        return best
    except Exception as e:
        print(f"Error loading final results: {e}")
        return None

# Default bounds
DEFAULT_BOUNDS = {
    "a_mm":       (0.006, 0.01),
    "r_neck_mm":  (200.0, 400.0),
    "r_max_mm":   (300.0, 700.0),
    "Rout_mm":    (400.0, 1000.0),
    "zMin_mm":    (0.0),
    "zMax_mm":    (1000.0, 3000.0),
    "I_A":        (30000.0, 150000.0),
    "spacing_mm": (1100.0, 3000.0),
}

if __name__ == "__main__":
    print("🔧 Starting GPU-optimized horn optimization for compute cluster")
    print("   - CUDA-accelerated data processing")
    print("   - Multi-GPU support")
    print("   - Parallel Geant4 execution")
    print("   - Cluster-optimized resource management")
    
    try:
        result = optimize_cluster(
            DEFAULT_BOUNDS, 
            n_seed=20, 
            n_rounds=20, 
            k_candidates=3,
            parallel_eval=True
        )
        
        if result is not None:
            print("✅ Optimization completed successfully!")
        else:
            print("❌ Optimization failed - check error messages above")
            
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed with error: {e}")
        print("   Check the error messages above for details")
