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

### Files Modified

#### Modified Files
- `xmlot/models/pycox.py` - Added weighted DeepHit functionality to both DeepHit and DeepHitSingle classes
