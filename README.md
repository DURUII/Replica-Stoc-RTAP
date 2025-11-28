# Replica-Stoc-RTAP

This repository hosts the official implementation of Robust Task Assignment in Vehicular Crowdsensing, with implementations of both the offline and online stochastic algorithms.

## Project Structure

```
src/
├── algos/                # Task assignment algorithms
│   ├── offline/          # Offline algorithms (e.g., greedy, submodular, random)
│   ├── stochastic/       # Online algorithms (e.g., explore-then-commit)
├── models/               # Core data models (e.g., Task, Vehicle)
├── utils/                # Utility functions (e.g., PBD computations)
├── data/                 # Preprocessed dataset (e.g., CD18, CD14, XA18)
├── figs/                 # Figures and visualizations (e.g., objective v.s. varying parameters)    
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Python API

To use the provided algorithms, simply instantiate a `Problem` and an `Algorithm`, then run `run_framework()` to obtain results.

```python
from src.algos.offline.proposed import SubmodularPolicy # Replace with any other algorithm as needed.
from src.models.bean import Problem, Task, Vehicle
import numpy as np

# Generate sample tasks and vehicles
tasks = [Task(idx=i) for i in range(7000)]
vehicles = [
    Vehicle(
        idx=i,
        p={j: np.random.random() for j in range(200)},
        r=np.random.random(),
        c={j: np.random.randint(1, 100) for j in range(200)}
    )
    for i in range(1000)
]

# Configure and solve the problem
problem = Problem(tasks=tasks, vehicles=vehicles, l=5, beta=0.5)
algorithm = SubmodularPolicy(problem)
algorithm.run_framework()
print("Total Expected Cost:", algorithm.get_total_expected_cost())
```

### Command-Line Usage

Run algorithms under offline or stochastic settings on some specific dataset:

```bash
python offline.py -D [DATASET]
python online.py -D [DATASET]
python lens.py -D [DATASET]
```

Replace `[DATASET]` with one of: `CD18`, `CD14`, `XA18`, `SYN` or `ALL` (runs all datasets sequentially).

- `offline.py`: Runs offline algorithms (RTA-P, RTA-PI, Random, Greedy, Genetic).
- `online.py`: Runs online/stochastic algorithms.
- `lens.py`: Analyzes the estimation gap and algorithm performance.
