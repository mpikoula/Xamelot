# Weighted DeepHit: Addressing Underrepresented Subgroups in Survival Analysis

## Overview

This guide explains how to use the modified DeepHit model with sample weights to improve performance on underrepresented subgroups in survival analysis datasets. The implementation allows you to incorporate sample weights into the loss function, giving higher importance to underrepresented subgroups during training.

## Problem Statement

In many real-world survival analysis datasets, certain subgroups (e.g., specific demographic groups, disease stages, or treatment types) may be underrepresented. Standard machine learning models tend to optimize for the majority groups, leading to poor performance on underrepresented subgroups. This can have serious implications in healthcare applications where fair and accurate predictions for all patient groups are crucial.

## Solution: Weighted DeepHit

The weighted DeepHit model addresses this issue by:

1. **Sample Weighting**: Assigning higher weights to samples from underrepresented subgroups
2. **Custom Loss Function**: Modifying the DeepHit loss function to incorporate sample weights
3. **Flexible Weighting Strategies**: Supporting multiple approaches to compute sample weights

## Installation and Setup

The weighted DeepHit functionality is included in the main xmlot package. No additional installation is required.

```python
from xmlot.models.pycox import DeepHit
from xmlot.models.weighting import (
    compute_subgroup_weights,
    compute_survival_weights,
    add_weights_to_dataframe,
    analyze_subgroup_distribution
)
from xmlot.data.dataframes import build_weighted_survival_accessor
```

## Quick Start Example

### Step 1: Prepare Your Data

```python
import pandas as pd
import numpy as np

# Your survival data should include:
# - Features (e.g., age, gender, disease stage)
# - Duration column (time to event or censoring)
# - Event column (1 for event, 0 for censoring)
# - Subgroup columns (e.g., race, ethnicity, treatment type)

df = pd.DataFrame({
    'age': [65, 70, 55, 80, ...],
    'gender': ['Male', 'Female', 'Male', 'Female', ...],
    'race': ['White', 'Black', 'Hispanic', 'Asian', ...],
    'stage': ['I', 'II', 'III', 'IV', ...],
    'duration': [120, 85, 200, 45, ...],
    'event': [1, 0, 1, 1, ...]
})
```

### Step 2: Analyze Subgroup Distribution

```python
# Analyze the distribution of subgroups to identify imbalances
analysis = analyze_subgroup_distribution(
    df, 
    subgroup_columns=['race', 'stage'], 
    duration_col='duration', 
    event_col='event'
)
print(analysis)
```

### Step 3: Compute Sample Weights

```python
# Compute weights to address underrepresented subgroups
weights = compute_survival_weights(
    df,
    duration_col='duration',
    event_col='event',
    subgroup_columns=['race', 'stage'],
    weight_strategy="inverse_frequency",
    time_bins=10
)

# Add weights to your DataFrame
df_with_weights = add_weights_to_dataframe(df, weights, "sample_weights")
```

### Step 4: Set Up Accessor

```python
# Create a weighted survival accessor
build_weighted_survival_accessor(
    event="event",
    duration="duration",
    weight_column="sample_weights",
    accessor_code="surv",
    exceptions=[]
)
```

### Step 5: Train the Weighted DeepHit Model

```python
from xmlot.data.discretise import BalancedDiscretiser
from xmlot.models.pycox import DeepHit
import torchtuples as tt

# Set up discretization
SIZE_GRID = 50
DISCRETISER = BalancedDiscretiser(df_train, "surv", SIZE_GRID)
data_train = DISCRETISER(df_train.copy())

# Model hyperparameters
hyperparameters = {
    "in_features": data_train.surv.features.shape[1],
    "num_nodes_shared": [64, 64],
    "num_nodes_indiv": [32],
    "num_risks": 1,
    "out_features": SIZE_GRID,
    "cuts": DISCRETISER.grid,
    "batch_norm": True,
    "dropout": 0.1,
    "alpha": 0.2,
    "sigma": 0.1,
    "seed": 42,
    "use_weights": True  # Enable weighted training
}

# Training parameters
train_params = {
    'batch_size': 256,
    'epochs': 50,
    'callbacks': [tt.callbacks.EarlyStopping(patience=10)],
    'verbose': True,
    'val_data': data_val,
    'lr': None,
    'tolerance': 10
}

# Create and train the model
model = DeepHit(
    accessor_code="surv",
    hyperparameters=hyperparameters
)
model.fit(data_train, train_params)
```

## Weighting Strategies

### 1. Inverse Frequency Weighting

Weights are inversely proportional to subgroup frequency:

```python
weights = compute_subgroup_weights(
    df, 
    subgroup_columns=['race', 'stage'],
    weight_strategy="inverse_frequency",
    min_weight=0.1,
    max_weight=10.0
)
```

### 2. Balanced Weighting

Equal weights for all subgroups:

```python
weights = compute_subgroup_weights(
    df, 
    subgroup_columns=['race', 'stage'],
    weight_strategy="balanced",
    min_weight=0.1,
    max_weight=10.0
)
```

### 3. Custom Weighting

Use your own weights for specific subgroups:

```python
custom_weights = {
    'White_I': 1.0,
    'White_II': 1.0,
    'Black_I': 2.5,
    'Black_II': 2.0,
    'Hispanic_I': 3.0,
    'Asian_I': 4.0
}

weights = compute_subgroup_weights(
    df, 
    subgroup_columns=['race', 'stage'],
    weight_strategy="custom",
    custom_weights=custom_weights,
    min_weight=0.1,
    max_weight=10.0
)
```

### 4. Survival-Specific Weighting

Combines subgroup weighting with event rarity and temporal distribution:

```python
weights = compute_survival_weights(
    df,
    duration_col='duration',
    event_col='event',
    subgroup_columns=['race', 'stage'],
    weight_strategy="inverse_frequency",
    time_bins=10  # Number of time bins for temporal weighting
)
```

## Model Evaluation

### Overall Performance

```python
from xmlot.metrics.survival import concordance

# Evaluate on test set
concordance_score = concordance(model, data_test)
print(f"Overall Concordance Index: {concordance_score:.4f}")
```

### Subgroup-Specific Performance

```python
# Evaluate performance on specific subgroups
races = data_test['race'].unique()
for race in races:
    mask = data_test['race'] == race
    subset = data_test[mask]
    
    if len(subset) > 10:  # Only evaluate if enough samples
        concordance_score = concordance(model, subset)
        print(f"{race}: {concordance_score:.4f} (n={len(subset)})")
```

## Advanced Usage

### Comparing Standard vs Weighted Models

```python
# Standard DeepHit (no weights)
model_standard = DeepHit(
    accessor_code="surv",
    hyperparameters={**common_params, "use_weights": False}
)

# Weighted DeepHit
model_weighted = DeepHit(
    accessor_code="surv",
    hyperparameters={**common_params, "use_weights": True}
)

# Compare performance
concordance_standard = concordance(model_standard, data_test)
concordance_weighted = concordance(model_weighted, data_test)

print(f"Improvement: {concordance_weighted - concordance_standard:.4f}")
```

### Custom Weight Computation

You can implement your own weight computation logic:

```python
def custom_weight_function(df, subgroup_columns):
    # Your custom logic here
    weights = np.ones(len(df))
    
    # Example: Weight based on clinical importance
    for i, row in df.iterrows():
        if row['stage'] == 'IV':
            weights[i] *= 2.0  # Higher weight for advanced stage
        if row['age'] > 75:
            weights[i] *= 1.5  # Higher weight for elderly patients
    
    return weights

# Apply custom weights
custom_weights = custom_weight_function(df, ['stage', 'age'])
df_with_weights = add_weights_to_dataframe(df, custom_weights, "sample_weights")
```

## Best Practices

### 1. Data Analysis
- Always analyze subgroup distribution before applying weights
- Identify which subgroups are truly underrepresented
- Consider clinical relevance when choosing subgroups

### 2. Weight Selection
- Start with inverse frequency weighting
- Use survival-specific weighting for complex temporal patterns
- Avoid extreme weights (use min_weight and max_weight parameters)
- Validate weight distribution across subgroups

### 3. Model Training
- Use early stopping to prevent overfitting
- Monitor validation performance on underrepresented subgroups
- Consider ensemble methods combining standard and weighted models

### 4. Evaluation
- Evaluate performance on each subgroup separately
- Use multiple metrics beyond concordance index
- Consider clinical significance of improvements

## Troubleshooting

### Common Issues

1. **Extreme Weights**: If weights are too extreme, adjust min_weight and max_weight parameters
2. **Overfitting**: Use early stopping and regularization
3. **Poor Performance**: Ensure sufficient samples in each subgroup for meaningful evaluation
4. **Memory Issues**: Reduce batch size or use smaller network architectures

### Debugging Tips

```python
# Check weight distribution
print(f"Weight statistics:")
print(f"Mean: {weights.mean():.3f}")
print(f"Std: {weights.std():.3f}")
print(f"Min: {weights.min():.3f}")
print(f"Max: {weights.max():.3f}")

# Check subgroup sizes
for subgroup in df[subgroup_columns].drop_duplicates().values:
    mask = (df[subgroup_columns] == subgroup).all(axis=1)
    print(f"{subgroup}: {mask.sum()} samples")
```

## References

- DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks (Lee et al., 2018)
- PyCox: Survival Analysis with Neural Networks (Kvamme et al., 2019)
- Fairness in Machine Learning for Healthcare (Chen et al., 2020)

## Support

For questions and issues related to the weighted DeepHit implementation, please refer to the main xmlot documentation or create an issue in the project repository. 