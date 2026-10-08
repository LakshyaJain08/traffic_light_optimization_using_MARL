# Multi-Objective Traffic Signal Optimization
### Multi-Agent Reinforcement Learning + ARIMA in SUMO

> Implements the system described in the SRS:  
> *"Multi-Objective Traffic Signal Optimization Using Multi-Agent Reinforcement
> Learning with ARIMA-Based Traffic Prediction in SUMO"*

---

## Architecture

```
Traffic Network (SUMO)
        ↓
Real-Time Data Extraction (TraCI)
        ↓
ARIMA Traffic Prediction Module
        ↓
State Construction Module
        ↓
Multi-Agent RL Controller (RLlib PPO)
        ↓
Signal Phase Decisions
        ↓
SUMO Simulation Step
        ↓
Reward Computation  R = -(w₁·Wₜ + w₂·Cₜ + w₃·Fₜ)
        ↓
Policy Update
```

---

## Project Structure

```
project/
├── config/
│   └── config.yaml              # All tunable parameters
├── sumo_files/
│   ├── generate_sumo_files.py   # Generates network, routes, config
│   ├── network.net.xml          # 2×2 grid network (4 TL intersections)
│   ├── routes.rou.xml           # Vehicle flows
│   ├── traffic_lights.add.xml   # Induction loop detectors
│   └── config.sumocfg           # SUMO master config
├── src/
│   ├── arima_predictor.py       # ARIMA(5,1,0) short-term inflow forecasting
│   ├── reward.py                # Multi-objective reward  R = -(w1·W + w2·C + w3·F)
│   ├── sumo_env.py              # Gymnasium multi-agent environment (TraCI)
│   ├── marl_trainer.py          # RLlib PPO trainer (shared policy)
│   ├── baseline.py              # Fixed-time control baseline
│   └── evaluation.py           # Evaluation & comparison plots
├── experiments/
│   └── run_experiments.py       # All 5 SRS experiment modes
├── results/                     # Auto-created; stores logs and plots
├── main.py                      # Entry point
└── requirements.txt
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| SUMO | 1.15+ | Set `SUMO_HOME` env variable |
| Ray RLlib | 2.7+ | `pip install ray[rllib]` |
| Gymnasium | 0.28+ | |
| statsmodels | 0.14+ | ARIMA |
| PyTorch | 2.x | RLlib backend |

### Install Python packages

```bash
pip install -r requirements.txt
```

### Install SUMO

Download from [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php)  
then set:

**Windows (PowerShell)**
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
```

**Linux / macOS**
```bash
export SUMO_HOME=/usr/share/sumo
export PATH=$SUMO_HOME/bin:$PATH
```

---

## Quick Start

### Step 1 – Generate SUMO network files

```bash
python sumo_files/generate_sumo_files.py
```

This creates `network.net.xml`, `routes.rou.xml`, `traffic_lights.add.xml`,
and `config.sumocfg` inside `sumo_files/`.

### Step 2 – Run the full pipeline

```bash
python main.py
```

This runs:
1. MARL training (PPO, shared policy, 200 episodes by default)
2. Evaluation against fixed-time baseline
3. Saves results to `results/`

---

## Experiment Modes (SRS §5)

| # | Mode | Command |
|---|---|---|
| 1 | Fixed-time baseline | `python main.py --mode baseline` |
| 2 | Waiting time only | `python experiments/run_experiments.py --mode wait_only` |
| 3 | CO2 only | `python experiments/run_experiments.py --mode co2_only` |
| 4 | Fuel only | `python experiments/run_experiments.py --mode fuel_only` |
| 5 | Full multi-objective | `python experiments/run_experiments.py --mode multi_objective` |
| – | All experiments | `python experiments/run_experiments.py --mode all` |

### Faculty Demo Mode

Run a single clean GUI episode with slower playback and camera-friendly defaults:

```bash
python main.py --mode demo --checkpoint results/checkpoints
```

### Baseline-Only Demo (No Model)

Run a GUI simulation with fixed-time control only (no RL model involved):

```bash
python main.py --mode baseline_demo
```

---

## Configuration

All parameters are in [`config/config.yaml`](config/config.yaml).

Key settings:

```yaml
reward:
  w1: 0.5   # waiting time weight
  w2: 0.3   # CO2 weight
  w3: 0.2   # fuel weight

arima:
  p: 5
  d: 1
  q: 0
  steps_ahead: 5

training:
  algorithm: PPO
  num_episodes: 200
  learning_rate: 3.0e-4
  gamma: 0.99
```

---

## State Space (per agent)

```
S = [ queue_length × n_lanes,        # vehicles halted per lane
      waiting_time × n_lanes,        # seconds waited per lane
      current_phase_one_hot × 4,     # one-hot encoded TL phase
      ARIMA_forecast × 5 ]           # predicted inflows for next 5 steps
```

## Action Space (discrete, per agent)

| Action | Description |
|---|---|
| 0 | Maintain current phase |
| 1 | Switch to next phase (via yellow) |
| 2 | Extend green by 5 s |
| 3 | Reduce green by 5 s |

---

## Outputs (`results/`)

| File | Contents |
|---|---|
| `training_logs.csv` | Episode rewards, waiting time, CO2, fuel per episode |
| `evaluation_metrics.csv` | Baseline vs MARL comparison with % reduction |
| `reward_plot.png` | Training progress curves |
| `comparison_plot.png` | Bar chart: baseline vs MARL |
| `baseline_metrics.csv` | Raw fixed-time metrics |
| `checkpoints/` | RLlib model checkpoints |

---

## Validation Criteria (SRS §9)

| Metric | Required reduction |
|---|---|
| Waiting time | > 10% vs baseline |
| CO2 emissions | > 5% vs baseline |
| Fuel consumption | > 5% vs baseline |

---

## Module Independence

Each module can be run/tested independently:

```bash
# Test ARIMA predictor
python -c "from src.arima_predictor import ARIMAPredictor; p = ARIMAPredictor(); [p.update(i) for i in range(30)]; print(p.predict())"

# Test reward calculator
python -c "from src.reward import RewardCalculator; r = RewardCalculator(); print(r.compute(100, 5000, 200))"

# Test baseline only
python main.py --mode baseline

# Generate network files only
python main.py --mode generate
```

---

## Future Extensions (SRS §10)

- [ ] Pareto front computation for weight sensitivity analysis
- [ ] Dynamic speed limit control
- [ ] Real traffic dataset integration (SUMO `osmWebWizard.py`)
- [ ] LSTM-based deep forecasting replacing ARIMA
- [ ] MADDPG algorithm (`training.algorithm: MADDPG`)
