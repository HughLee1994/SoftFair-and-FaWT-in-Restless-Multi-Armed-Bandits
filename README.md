# Fair Resource Allocation in Restless Multi-Armed Bandits
Includes two works:
- [![arXiv](https://img.shields.io/badge/arXiv-2207.13343-b31b1b.svg) SoftFair: Towards Soft Fairness in Restless Multi-Armed Bandits](https://arxiv.org/abs/2207.13343)
- [![arXiv](https://img.shields.io/badge/arXiv-2206.03883-b31b1b.svg) Efficient Resource Allocation with Fairness Constraints in Restless Multi-Armed Bandits](https://arxiv.org/pdf/2206.03883)

## Overview

This repository contains implementations of fairness-aware algorithms for Restless Multi-Armed Bandit (RMAB) problems. RMABs are powerful frameworks for allocating limited resources under uncertainty, with critical applications in public health interventions, patient adherence monitoring, and resource allocation.

Traditional RMAB solutions (e.g., Whittle index-based methods) tend to starve certain arms (beneficiaries) by focusing exclusively on maximizing cumulative rewards. This can have severe consequences in public health settings where consistent interventions are crucial for program effectiveness.

We provide two complementary approaches to ensuring fairness in RMAB:

### 1. **FaWT (Fair Whittle Thresholding)** - Hard Fairness Constraints
*From: "Efficient Resource Allocation with Fairness Constraints in Restless Multi-Armed Bandits" *

- **Fairness Definition**: Guarantees a minimum activation frequency η within any time window L
- **Approach**: Modifies Whittle index policy to ensure fairness constraints are satisfied
- **Key Features**: 
  - Handles both known and unknown transition models
  - Works with finite and infinite horizons
  - Provides Q-Learning based variant (FaWT-Q) for model-free learning
  - Supports Thompson sampling (FaWT-U) for uncertain transitions

### 2. **SoftFair** - Soft Fairness Constraints  
*From: "Towards Soft Fairness in Restless Multi-Armed Bandits" *

- **Fairness Definition**: Never probabilistically favors one arm over another if its long-term cumulative reward is lower
- **Approach**: Softmax-based value iteration with temperature parameter c
- **Key Features**:
  - No indexability assumption required
  - Adjustable fairness-optimality trade-off via parameter c
  - Asymptotically optimal with theoretical guarantees
  - No need to specify fairness parameters a priori

## Problem Formulation

### Restless Multi-Armed Bandit (RMAB)

- **N independent arms**, each evolving according to a Markov Decision Process (MDP)
- **Resource constraint**: Select at most k arms at each time step (k ≪ N)
- **Finite/Infinite horizon**: T time steps per episode or infinite horizon
- **States**: Binary states {0, 1} representing "bad" and "good" states
- **Actions**: Active (intervention) or passive for each arm
- **Partial observability**: States of unactivated arms are unobserved

### Fairness Constraints - Two Approaches

#### Hard Fairness (FaWT)

For any arm i and any time window [u, u+L]:

```
Σ_{t=u}^{u+L} a_i^t ≥ η    ∀u ∈ {1, ..., T-L}
```

Where:
- L is the time window length
- η is the minimum number of activations required within window L
- Requires: k × L > N × (η - 1) for feasibility

**Intuition**: Every arm must be activated at least η times in any consecutive L time steps.

#### Soft Fairness (SoftFair)

A stochastic policy π is **fair** if for any time step t ∈ [T], joint state s, and actions a, a':

```
π_t(s, a) ≥ π_t(s, a') only if Q*(s, a) ≥ Q*(s, a')
```

Where Q*(s, a) is the optimal state-action value function.

**Intuition**: Never probabilistically prefer an action with lower expected long-term value.

### Comparison of Fairness Notions

| Aspect | FaWT (Hard Fairness) | SoftFair (Soft Fairness) |
|--------|---------------------|-------------------------|
| **Guarantee** | Minimum activations per window | Probabilistic preference based on value |
| **Parameters** | L (window), η (min activations) | c (temperature parameter) |
| **A priori knowledge** | Must specify L and η | No fairness parameters needed |
| **Constraint type** | Hard (must satisfy) | Soft (probabilistic) |
| **Flexibility** | Fixed guarantee | Adjustable via c |
| **Real-world use** | When minimum service required | When relative fairness important |

## Methods

### Method 1: FaWT (Fair Whittle Thresholding)

FaWT modifies the Whittle index algorithm to handle hard fairness constraints with minimal overhead.

#### Algorithm Overview

1. **Compute Whittle indices** for all arms
2. **Check fairness violations**: Identify arms that haven't been activated η times in the last L steps
3. **Priority selection**: 
   - First, select arms violating fairness constraints
   - Then, select top-k remaining arms by Whittle index
4. **Execute** and observe next states

#### Variants

**FaWT-U (Uncertain Transitions)**: Uses Thompson Sampling when transition probabilities are partially unknown
- Sample transition probabilities from posterior Beta distributions
- Update posteriors based on observations

**FaWT-Q (Unknown Transitions)**: Q-Learning based approach when transitions are completely unknown
- Learn Q-values Q(s, a, l) where l = steps since last activation
- Whittle index = Q(s, a=1, l) - Q(s, a=0, l)


### Method 2: SoftFair (Softmax Fair Value Iteration)

SoftFair integrates softmax-based value iteration with RMAB to satisfy soft fairness constraints.

#### Key Steps

1. **Compute logit values** for each arm based on value functions:
   ```
   λ_i = Q_i(s_i, a_i=1) - Q_i(s_i, a_i=0)
   ```

2. **Softmax selection** with temperature parameter c:
   ```
   π(s, a=I{i}) = exp(c·λ_i) / Σ_j exp(c·λ_j)
   ```

3. **Sample k arms** without replacement based on computed probabilities

4. **Update value functions** using observed transitions:
   ```
   V^{ep}_{i,t}(s) = Σ_{a} Pr(a|s) * Σ_{s'} Pr(s'|s,a) * [R(s,a) + γV^{ep-1}_{i,t+1}(s')]
   ```


## Installation

```bash
git clone https://github.com/yourusername/softfair-rmab.git
cd softfair-rmab
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib (for visualization)

## Usage

### Example 1: FaWT with Known Transitions

```python
from fairness_rmab import FaWT
import numpy as np

# Define transition matrices for each arm
# P[arm_id][action][from_state][to_state]
transition_matrices = [...] 

# Initialize FaWT
rmab = FaWT(
    n_arms=100,
    k=10,  # resource constraint
    T=1000,  # time horizon
    L=50,  # fairness window
    eta=2,  # minimum activations per window
    transition_matrices=transition_matrices
)

# Run the policy
initial_beliefs = np.random.rand(100)  # belief states
selected_arms, rewards = rmab.run(initial_beliefs)

print(f"Selected arms: {selected_arms}")
print(f"Average reward: {np.mean(rewards)}")
```

### Example 2: FaWT-Q (Model-free Learning)

```python
from fairness_rmab import FaWT_Q

# Initialize without transition matrices
rmab_q = FaWT_Q(
    n_arms=100,
    k=10,
    L=50,
    eta=2,
    epsilon=0.1,  # exploration rate
    learning_rate=0.01
)

# Train through interaction
for episode in range(1000):
    initial_states = np.random.randint(0, 2, size=100)
    episode_reward = rmab_q.run_episode(initial_states)
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")

# Get learned policy
policy = rmab_q.get_policy()
```

### Example 3: SoftFair with Adjustable Fairness

```python
from fairness_rmab import SoftFairRMAB
import numpy as np

# Define transition matrices
transition_matrices = [...] 

# Initialize SoftFair
rmab = SoftFairRMAB(
    n_arms=100,
    k=10,
    T=80,
    c=2.0,  # fairness-optimality trade-off
    gamma=1.0,
    transition_matrices=transition_matrices
)

# Train the policy
rmab.train(n_episodes=1000)

# Execute policy
initial_states = np.random.randint(0, 2, size=100)
selected_arms, rewards = rmab.select_arms(initial_states)

print(f"Selected arms: {selected_arms}")
print(f"Expected reward: {rewards}")

# Check fairness metrics
entropy = rmab.compute_action_entropy()
print(f"Action entropy (fairness measure): {entropy:.4f}")
```

### Example 4: Comparing FaWT and SoftFair

```python
from fairness_rmab import FaWT, SoftFairRMAB
import matplotlib.pyplot as plt

# Same setup for both
n_arms, k, T = 100, 10, 1000
transition_matrices = [...]

# FaWT with hard constraints
fawt = FaWT(n_arms, k, T, L=50, eta=2, 
            transition_matrices=transition_matrices)

# SoftFair with soft constraints
softfair = SoftFairRMAB(n_arms, k, T, c=2.0, gamma=1.0,
                        transition_matrices=transition_matrices)
softfair.train(n_episodes=100)

# Compare performance
fawt_rewards = []
softfair_rewards = []

for trial in range(50):
    initial_beliefs = np.random.rand(n_arms)
    _, r1 = fawt.run(initial_beliefs)
    _, r2 = softfair.select_arms(initial_beliefs)
    fawt_rewards.append(np.mean(r1))
    softfair_rewards.append(np.mean(r2))

print(f"FaWT avg reward: {np.mean(fawt_rewards):.4f} ± {np.std(fawt_rewards):.4f}")
print(f"SoftFair avg reward: {np.mean(softfair_rewards):.4f} ± {np.std(softfair_rewards):.4f}")

# Visualize selection distribution
fawt.plot_selection_distribution()
softfair.plot_selection_distribution()
```

### Advanced Configuration

#### FaWT with Thompson Sampling (Uncertain Model)

```python
from fairness_rmab import FaWT_U

# Initialize with prior distributions
rmab_u = FaWT_U(
    n_arms=100,
    k=10,
    L=50,
    eta=2,
    # Beta prior parameters for transition probabilities
    prior_alpha=np.ones((100, 2, 2, 2)),
    prior_beta=np.ones((100, 2, 2, 2))
)

# Run and update posteriors
for episode in range(500):
    beliefs = rmab_u.get_beliefs()
    selected_arms, rewards, observations = rmab_u.run_episode(beliefs)
    rmab_u.update_posteriors(selected_arms, observations)
```

#### SoftFair with Different Fairness Levels

```python
# Test different c values (fairness-optimality trade-off)
c_values = [0.5, 1.0, 2.0, 5.0, float('inf')]
results = {}

for c in c_values:
    rmab = SoftFairRMAB(n_arms=100, k=10, T=100, c=c, gamma=0.99,
                        transition_matrices=transition_matrices)
    rmab.train(n_episodes=1000)
    
    # Evaluate
    rewards, entropy = rmab.evaluate(n_trials=50)
    results[c] = {'reward': np.mean(rewards), 'entropy': entropy}
    
    print(f"c={c}: Reward={results[c]['reward']:.4f}, Entropy={results[c]['entropy']:.4f}")
```


## Reproducing Results

### FaWT Experiments

```bash
# Run FaWT on synthetic data
python experiments/run_fawt_synthetic.py --n_arms 100 --k 10 --T 1000 --L 50 --eta 2

# Run FaWT-Q (model-free)
python experiments/run_fawt_q.py --n_arms 100 --k 10 --epsilon 0.1 --episodes 1000

# Run FaWT-U (Thompson sampling)
python experiments/run_fawt_u.py --n_arms 100 --k 10 --L 50 --eta 2

# Vary fairness constraint strength
python experiments/fairness_ablation.py --L 15 30 50 --eta 2
```

### SoftFair Experiments

```bash
# Run SoftFair on CAPA dataset
python experiments/run_capa_experiment.py --n_arms 100 --k 10 --T 80 --c 2.0

# Run SoftFair on synthetic data
python experiments/run_synthetic_experiment.py --n_arms 100 --k 10 --T 50 --c 1.0

# Ablation study on c parameter
python experiments/c_parameter_study.py --c_values 0.5 1.0 2.0 5.0
```


## Citation

If you use this code in your research, please cite the relevant papers:

### FaWT (Fair Whittle Thresholding)

```bibtex
@inproceedings{li2022efficient,
  title={Efficient Resource Allocation with Fairness Constraints in Restless Multi-Armed Bandits},
  author={Li, Dexun and Varakantham, Pradeep},
  booktitle={Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2022}
}
```

### SoftFair

```bibtex
@article{li2022softfair,
  title={Towards Soft Fairness in Restless Multi-Armed Bandits},
  author={Li, Dexun and Varakantham, Pradeep},
  journal={arXiv preprint arXiv:2207.13343},
  year={2022}
}
```

---
