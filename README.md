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




## Installation

```bash
git clone https://github.com/HughLee1994/softfair-rmab.git
cd softfair-rmab
pip install -r requirements.txt
```

## Usage

```bash
python policy.py
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
