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
- Maintained exact compatibility with original PyCox implementation when `use_weights=False`
- Added support for ablation modes to control different aspects of weighted training

#### 2. Key Features

#### Weighted Loss Function
- Extends PyCox's standard DeepHit loss with sample weights
- Maintains original loss structure and behavior
- Supports multiple ablation modes for different training strategies
- Automatically falls back to standard loss when weights are uniform

#### Backward Compatibility
- Standard DeepHit still works without changes
- `use_weights=False` (default) maintains original behavior exactly
- `use_weights=True` enables weighted training with custom loss

#### Ablation Modes
- `"full"`: Use weighted training with all loss components
- `"unweighted"`: Use standard PyCox training despite weights (for comparison)

The next step is to implement weighting on the individual components (likelyhood and rank loss)

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

### Implementation Details

#### Weighted Loss Function
The weighted loss function extends PyCox's standard DeepHit loss by:
- Storing sample weights in the network during training
- Applying weight scaling to the loss computation
- Maintaining the original loss structure and gradients
- Automatically detecting uniform weights and using standard loss

#### Training Process
1. **Weight Storage**: Sample weights are stored in the network for loss function access
2. **Batch Processing**: Weights are mapped to batches using proper indexing
3. **Loss Computation**: Standard PyCox loss is scaled by weight factors
4. **Fallback**: Uniform weights automatically use standard loss

#### Ablation Modes
- **`"full"`**: Standard weighted training with all components
- **`"unweighted"`**: Uses standard PyCox training for comparison studies
- **`"pycox_standard"`**: Forces use of PyCox's standard loss

### Benefits

1. **Fairness**: Improves performance on underrepresented subgroups
2. **Compatibility**: Maintains exact compatibility with original PyCox implementation
3. **Flexibility**: Multiple ablation modes for different use cases
4. **Robustness**: Automatic fallback to standard loss when appropriate
5. **Performance**: Uses original PyCox training infrastructure

### Best Practices

1. **Start with `use_weights=False`** to establish baseline performance
2. **Use `ablation_mode="unweighted"`** for fair comparison studies
3. **Monitor validation performance** on underrepresented subgroups
4. **Ensure weight distribution** is reasonable (avoid extreme values)
5. **Use proper accessor setup** to handle weights correctly

### Troubleshooting

#### Common Issues
1. **Performance Differences**: Use `ablation_mode="unweighted"` for fair comparisons
2. **Weight Application**: Ensure weights are properly stored in the accessor
3. **Memory Issues**: Reduce batch size if needed
4. **Training Instability**: Check weight distribution and consider normalization

#### Debugging Tips
```python
# Check if weights are being used
print(f"Model uses weights: {model.use_weights}")
print(f"Ablation mode: {model.m_hyperparameters.get('ablation_mode', 'full')}")

# Check weight distribution in your data
if hasattr(data_train.surv, 'weights'):
    weights = data_train.surv.weights
    print(f"Weight mean: {weights.mean():.3f}")
    print(f"Weight std: {weights.std():.3f}")
    print(f"Weight min: {weights.min():.3f}")
    print(f"Weight max: {weights.max():.3f}")
```

### Files Modified

#### Modified Files
- `xmlot/models/pycox.py` - Added weighted DeepHit functionality to both DeepHit and DeepHitSingle classes
