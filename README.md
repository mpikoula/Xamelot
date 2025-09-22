# Welcome to Xamelot
## A library about eXplAinable MachinE Learning for Organ Transplant

The disparity between the size of the waiting list and the availability of donor kidneys means that clinicians often consider organs from suboptimal donors for transplantation. Such decisions can be challenging, and no decision support tools are currently available to help support clinicians and patients.
Therefore, this project aims to create a clinical decision support tool that will help a clinician and patient make a personalised decision about a given organ offer. Notably, a suite of interpretable algorithms will be developed to predict post-transplant graft and patient survival, as well as patient outcomes if an organ offer is declined.

_Please note that this work is exploratory._

## Installation

Once this repository has been cloned, you can then install **Xamelot** with the following command line:

`pip3 install --upgrade --force-reinstall THE_APPROPRIATE_PATH/xmlot`

You can finally test your installation with:

`pytest THE_APPROPRIATE_PATH/xmlot`

This functionality is quite handy to check that nothing is broken if you brought any local change to the library.

## Weighted DeepHit Implementation

### Overview

This implementation extends the DeepHit model in the PyCox module to support sample weights for addressing underrepresented subgroups in survival analysis datasets. The modification allows you to incorporate sample weights into the loss function, giving higher importance to underrepresented subgroups during training.

### What Was Implemented

#### 1. Modified DeepHit Classes (`xmlot/models/pycox.py`)
- Added `use_weights` parameter to enable weighted training for both `DeepHit` and `DeepHitSingle`
- Created custom weighted loss function that extends PyCox's standard loss with sample weights
- Added `_df_to_xyw_()` method to extract features, targets, and weights

#### 2. Key Features

#### Weighted Loss Function
- Extends PyCox's standard DeepHit loss with sample weights
- Maintains original loss structure and behavior
- Supports multiple ablation modes for different training strategies

#### Backward Compatibility
- Standard DeepHit still works without changes
- `use_weights=False` (default) maintains original behavior exactly
- `use_weights=True` enables weighted training with custom loss

#### Ablation Modes
- **`"full"`**: Standard weighted training with all components
- **`"unweighted"`**: Uses standard PyCox training for comparison studies
- **`"likelihood_only"`**: Applies weights to the likelihood component of the loss function - NOT IMPLEMENTED
- **`"ranking_only"`**: Applies weights to the ranking component of the loss function - NOT IMPLEMENTED

### Quick Usage Example

```python
import pandas as pd
import numpy as np
from xmlot.models.pycox import DeepHit

# 1. Prepare your data with weights
df = pd.DataFrame({
    'age': [65, 70, 55, 80],
    'race': ['White', 'Black', 'Hispanic', 'Asian'],
    'stage': ['I', 'II', 'III', 'IV'],
    'duration': [120, 85, 200, 45],
    'event': [1, 0, 1, 1],
    'weights': [1.0, 2.0, 1.5, 3.0]  # Sample weights
})

# 2. Set up accessor (assuming you have a weighted accessor)
# This would typically be done through your data preprocessing pipeline

# 3. Train weighted DeepHit model
model = DeepHit(
    accessor_code="surv",
    hyperparameters={
        "in_features": 4,
        "num_nodes_shared": [64, 64],
        "num_nodes_indiv": [32],
        "num_risks": 1,
        "out_features": 50,
        "cuts": [0, 10, 20, 30, 40, 50],
        "batch_norm": True,
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "seed": 42,
        "use_weights": True,  # Enable weighted training
        "ablation_mode": "full"  # Use full weighted training
    }
)

# 4. Train the model
model.fit(data_train, train_params)
```

## Fairness-Aware DeepHit Implementation

This module extends PyCox's DeepHit models with fairness-aware loss functions that penalize differences in mean risk between demographic subgroups.

## Overview

The fairness loss function implements a squared-difference penalty on mean risk between groups:

```
Loss = α·NLL + (1-α)·ranking_loss + fairness_weight·[(μ_group1 - μ_group2)²]
```

where:
- `α·NLL + (1-α)·ranking_loss` is the standard DeepHit loss
- `fairness_weight` controls the strength of the fairness penalty
- `μ_group1` and `μ_group2` are the mean risks for different demographic subgroups

## Key Features

- **Direct Fairness Penalty**: Penalizes differences in mean risk between groups
- **Flexible Group Definition**: Works with any binary group indicator (race, gender, age group, etc.)
- **Configurable Strength**: Adjust `fairness_weight` to control penalty strength
- **Backward Compatible**: Standard DeepHit still works without changes
- **Comprehensive Monitoring**: Tracks loss components and group risk differences
- **Visualization**: Built-in plotting of fairness metrics during training

## Usage

### Basic Fairness Training

```python
from xmlot.models.pycox import DeepHitSingle

# Create fairness-aware model
model = DeepHitSingle(
    accessor_code="surv",
    hyperparameters={
        "in_features": 10,
        "num_nodes": [32, 32],
        "out_features": 50,
        "batch_norm": True,
        "dropout": 0.1,
        "alpha": 0.2,
        "sigma": 0.1,
        "seed": 42,
        "use_fairness": True,  # Enable fairness mode
        "fairness_weight": 0.1  # Control penalty strength
    }
)

# Train with group indicators
model.fit(train_data, train_params)
```

### Data Requirements

Your data accessor must include `group_indicators`:

```python
class MyAccessor:
    @property
    def features(self):
        return self.df[feature_columns]
    
    @property
    def durations(self):
        return self.df['duration']
    
    @property
    def events(self):
        return self.df['event']
    
    @property
    def group_indicators(self):  # Required for fairness
        # 0 = group1, 1 = group2
        return self.df['race'].map({'White': 0, 'Black': 1})
```

### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_fairness` | bool | False | Enable fairness-aware training |
| `fairness_weight` | float | 0.1 | Strength of fairness penalty |
| `alpha` | float | 0.2 | Weight for NLL vs ranking loss |
| `sigma` | float | 0.1 | Temperature for ranking loss |

### Monitoring Training

```python
# Get loss history
loss_history = model.get_loss_history()

# Plot fairness components
model.plot_loss_components(save_path="fairness_plot.png")

# Access specific metrics
print(f"Final fairness penalty: {loss_history['fairness_loss'][-1]:.4f}")
print(f"Group 1 mean risk: {loss_history['group1_mean_risk'][-1]:.4f}")
print(f"Group 2 mean risk: {loss_history['group2_mean_risk'][-1]:.4f}")
```

## Example: Black vs Non-Black Patients

```python
# Create group indicators for Black vs non-Black patients
def create_race_indicators(df):
    # 0 = non-Black, 1 = Black
    return (df['ethnicity'] == 'Black').astype(int)

# Train fairness-aware model
model = DeepHitSingle(
    accessor_code="surv",
    hyperparameters={
        # ... other parameters ...
        "use_fairness": True,
        "fairness_weight": 0.2  # Stronger penalty ethnic minority group
    }
)

# The model will learn to reduce differences in mean risk between Black and non-Black patients
```

## Comparison with Sample Weighting

| Approach | Pros | Cons |
|----------|------|------|
| **Sample Weighting** | Simple implementation, preserves ranking loss | May not directly address risk differences |
| **Fairness Penalty** | Directly penalizes risk differences, principled approach | Requires tuning fairness_weight, may impact overall performance |

## Implementation Details

### Loss Function

The fairness penalty is computed as:

1. **Risk Calculation**: `risk = sum(PMF[event_times])` for each sample
2. **Group Means**: `μ_group1 = mean(risk[group1])`, `μ_group2 = mean(risk[group2])`
3. **Penalty**: `fairness_loss = (μ_group1 - μ_group2)²`

### Gradient Flow

The fairness penalty affects gradients through:
- Direct impact on total loss
- Backpropagation through risk calculation
- Influence on PMF computation via softmax

### Numerical Stability

- Uses epsilon (1e-8) to prevent division by zero
- Clamps risk values to prevent extreme gradients
- Handles edge cases with empty groups




