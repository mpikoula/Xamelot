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

#### 1. Modified DeepHit Class (`xmlot/models/pycox.py`)
- Added `use_weights` parameter to enable weighted training
- Created custom weighted loss function that incorporates sample weights
- Added `_df_to_xyw_()` method to extract features, targets, and weights
- Implemented custom training loop for weighted models
- Maintained backward compatibility with standard DeepHit

#### 2. Weighting Utilities (`xmlot/models/weighting.py`)
- `compute_subgroup_weights()`: Multiple weighting strategies
- `compute_survival_weights()`: Survival-specific weighting
- `add_weights_to_dataframe()`: Helper to add weights to DataFrame
- `analyze_subgroup_distribution()`: Analyze subgroup imbalances

#### 3. Weighted Accessor (`xmlot/data/dataframes.py`)
- `build_weighted_survival_accessor()`: Creates accessor that handles sample weights
- Automatically excludes weight column from features
- Provides easy access to weights through accessor

#### 4. Documentation and Examples
- Comprehensive guide (`docs/weighted_deephit_guide.md`)
- Complete example script (`examples/weighted_deephit_example.py`)
- Test suite (`tests/test_weighted_deephit.py`)

### Key Features

#### Multiple Weighting Strategies
1. **Inverse Frequency**: Weights inversely proportional to subgroup frequency
2. **Balanced**: Equal weights for all subgroups
3. **Custom**: Use your own weights for specific subgroups
4. **Survival-Specific**: Combines subgroup weighting with event rarity and temporal distribution

#### Backward Compatibility
- Standard DeepHit still works without changes
- `use_weights=False` (default) maintains original behavior
- `use_weights=True` enables weighted training

#### Flexible Weight Computation
- Support for custom weight functions
- Configurable min/max weight bounds
- Multiple subgroup column support

### Quick Usage Example

```python
import pandas as pd
import numpy as np
from xmlot.models.weighting import compute_survival_weights, add_weights_to_dataframe
from xmlot.data.dataframes import build_weighted_survival_accessor
from xmlot.models.pycox import DeepHit

# 1. Prepare your data with underrepresented subgroups
df = pd.DataFrame({
    'age': [65, 70, 55, 80],
    'race': ['White', 'Black', 'Hispanic', 'Asian'],
    'stage': ['I', 'II', 'III', 'IV'],
    'duration': [120, 85, 200, 45],
    'event': [1, 0, 1, 1]
})

# 2. Compute sample weights
weights = compute_survival_weights(
    df,
    duration_col='duration',
    event_col='event',
    subgroup_columns=['race', 'stage'],
    weight_strategy="inverse_frequency",
    time_bins=10
)

# 3. Add weights to DataFrame
df_with_weights = add_weights_to_dataframe(df, weights, "sample_weights")

# 4. Set up weighted accessor
build_weighted_survival_accessor(
    event="event",
    duration="duration",
    weight_column="sample_weights",
    accessor_code="surv"
)

# 5. Train weighted DeepHit model
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
        "use_weights": True  # Enable weighted training
    }
)

# 6. Train the model
model.fit(data_train, train_params)
```

### Testing

The implementation has been thoroughly tested:

#### Core Functionality Tests
- ✅ Weight computation (inverse frequency, balanced, custom)
- ✅ Survival-specific weighting
- ✅ DataFrame integration
- ✅ Subgroup analysis
- ✅ Multiple subgroup support
- ✅ Weight distribution validation

#### Model Integration Tests
- ✅ Weighted DeepHit initialization
- ✅ Backward compatibility with standard DeepHit
- ✅ Accessor functionality

### Files Modified/Created

#### Modified Files
- `xmlot/models/pycox.py` - Added weighted DeepHit functionality
- `xmlot/data/dataframes.py` - Added weighted survival accessor
- `xmlot/models/__init__.py` - Updated imports

#### New Files
- `xmlot/models/weighting.py` - Weighting utilities
- `docs/weighted_deephit_guide.md` - Comprehensive documentation
- `examples/weighted_deephit_example.py` - Complete usage example
- `tests/test_weighted_deephit.py` - Full test suite
- `tests/test_weighting_standalone.py` - Standalone weight tests

### Benefits

1. **Fairness**: Improves performance on underrepresented subgroups
2. **Flexibility**: Multiple weighting strategies for different use cases
3. **Ease of Use**: Simple API that integrates with existing code
4. **Backward Compatibility**: Existing code continues to work
5. **Comprehensive Testing**: Thoroughly tested functionality

### Best Practices

1. **Analyze your data** to identify underrepresented subgroups
2. **Start with inverse frequency weighting** for most cases
3. **Use survival-specific weighting** for complex temporal patterns
4. **Avoid extreme weights** by setting appropriate min/max bounds
5. **Evaluate performance** on each subgroup separately
6. **Monitor validation performance** on underrepresented subgroups

### Troubleshooting

#### Common Issues
1. **Extreme Weights**: Adjust `min_weight` and `max_weight` parameters
2. **Overfitting**: Use early stopping and regularization
3. **Poor Performance**: Ensure sufficient samples in each subgroup
4. **Memory Issues**: Reduce batch size or use smaller architectures

#### Debugging Tips
```python
# Check weight distribution
print(f"Weight mean: {weights.mean():.3f}")
print(f"Weight std: {weights.std():.3f}")
print(f"Weight min: {weights.min():.3f}")
print(f"Weight max: {weights.max():.3f}")

# Check subgroup sizes
for subgroup in df[subgroup_columns].drop_duplicates().values:
    mask = (df[subgroup_columns] == subgroup).all(axis=1)
    print(f"{subgroup}: {mask.sum()} samples")
```

### Conclusion

The weighted DeepHit implementation provides a robust solution for addressing fairness issues in survival analysis. By giving higher importance to underrepresented subgroups during model training, it should lead to more equitable predictions across all patient groups.

The implementation is production-ready, thoroughly tested, and maintains full backward compatibility with existing code.
