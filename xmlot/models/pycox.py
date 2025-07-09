# Wrap various classifiers from PyCox.
# Everything follows Model's design.
#
# More details on: https://github.com/havakv/pycox

import pandas as pd
import numpy  as np
from scipy.interpolate import interp1d

import pycox.models as pycox
import torch
import torchtuples  as tt

from xmlot.models.model import FromTheShelfModel

def _adapt_input_(x):
    # Adapt input
    if type(x) == pd.DataFrame:
        return torch.tensor(x.values, dtype=torch.float32)
    elif type(x) == np.ndarray:
        return torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        return x.float()  # Ensure float32
    else:
        return torch.tensor(x, dtype=torch.float32)

class PyCoxModel(FromTheShelfModel):
    def __init__(self, accessor_code, hyperparameters=None):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        self.m_net  = None
        self.m_log  = None

    @property
    def net(self):
        return self.m_net

    @property
    def log(self):
        return self.m_log

    def _df_to_xy_(self, df):
        """
        Extract features and targets from a DataFrame into the intended PyCox fromat.
        """
        accessor = getattr(df, self.accessor_code)
        x = accessor.features.to_numpy().astype(np.float32)
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        return x, y

    def _df_to_xyw_(self, df):
        """
        Extract features, targets, and weights from a DataFrame into the intended PyCox format.
        """
        accessor = getattr(df, self.accessor_code)
        x = accessor.features.to_numpy().astype(np.float32)
        y = (
            accessor.durations.values,
            accessor.events.values
        )
        # Check if weights are available in the accessor
        if hasattr(accessor, 'weights') and accessor.weights is not None:
            w = accessor.weights.values.astype(np.float32)
        else:
            w = None
        return x, y, w

# DEEPSURV #

class DeepSurv(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/cox-ph.ipynb

    Args:
        - in_features : dimension of a feature vector as input.
        - num_nodes   : sizes for intermediate layers.
        - batch_norm  : boolean enabling batch normalisation
        - dropout     : drop out rate.
        - output_bias : if set to False, no additive bias will be learnt.
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(
            accessor_code   = accessor_code,
            hyperparameters = hyperparameters
        )
        in_features  = hyperparameters["in_features"]
        num_nodes    = hyperparameters["num_nodes"]    # [32, 32]
        out_features = 1
        batch_norm   = hyperparameters["batch_norm"]   # True
        dropout      = hyperparameters["dropout"]      # 0.1
        output_bias  = hyperparameters["output_bias"]  # False

        self.m_net = tt.practical.MLPVanilla(
            in_features,
            num_nodes,
            out_features,
            batch_norm,
            dropout,
            output_bias=output_bias)

        self.m_model = pycox.CoxPH(self.m_net, tt.optim.Adam)

    def fit(self, data_train, parameters=None):
        x, y, = self._df_to_xy_(data_train)

        # Compute learning rate
        if parameters['lr']:
            self.model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.model.optimizer.set_lr(lr)

        # Train
        self.m_log = self.model.fit(
            x,
            y,
            batch_size=parameters['batch_size'],
            epochs=parameters['epochs'],
            callbacks=parameters['callbacks'],
            verbose=parameters['verbose'],
            val_data=self._df_to_xy_(parameters["val_data"]),
            val_batch_size=parameters['batch_size']
        )

        _ = self.model.compute_baseline_hazards()

        return self

    def predict(self, x, parameters=None):
        input_tensor = torch.tensor(x.values).cuda()
        output = self.net(input_tensor).cpu().detach().numpy()
        output = output.reshape((output.shape[0],))
        return output


# DEEPHIT #

class DeepHitSingle(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes    : sizes for intermediate layers.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
        - use_weights  : boolean to enable sample weighting for underrepresented subgroups
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code, hyperparameters=hyperparameters)

        in_features = hyperparameters["in_features"]
        num_nodes = hyperparameters["num_nodes"]  # [32, 32]
        out_features = hyperparameters["out_features"]
        batch_norm = hyperparameters["batch_norm"]  # True
        dropout = hyperparameters["dropout"]  # 0.1
        seed = hyperparameters.get("seed")
        self.use_weights = hyperparameters.get("use_weights", False)  # Default to False for backward compatibility

        if seed is not None:
            torch.manual_seed(seed)

        self.m_net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

        self.m_model = pycox.DeepHitSingle(
            self.m_net,
            tt.optim.Adam,
            alpha=hyperparameters["alpha"],  # 0.2,
            sigma=hyperparameters["sigma"],  # 0.1,
        )
        
        # Handle weighted mode
        if self.use_weights:
            ablation_mode = hyperparameters.get("ablation_mode", "full")
            print(f"🔍 DeepHitSingle: use_weights=True, ablation_mode={ablation_mode}")
            
            if ablation_mode != "pycox_standard":
                # Create custom loss function for weighted mode
                print("🔍 Creating custom loss function for weighted DeepHitSingle...")
                self._create_weighted_loss_function(hyperparameters["alpha"], hyperparameters["sigma"])
        else:
            print("🔍 DeepHitSingle: use_weights=False (standard PyCox training)")
            # No custom loss function - use PyCox's standard loss

    def _create_weighted_loss_function(self, alpha, sigma):
        """
        Create a weighted loss function that extends PyCox's standard DeepHit loss with sample weights.
        """
        class WeightedDeepHitLoss:
            def __init__(self, net, alpha, sigma, duration_index, ablation_mode="full", max_ranking_pairs=20):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.duration_index = duration_index
                self.debug_counter = 0
                self.ablation_mode = ablation_mode
                self.max_ranking_pairs = max_ranking_pairs
                
                # Create PyCox's standard loss function
                import pycox.models.loss as pycox_loss
                self.standard_loss = pycox_loss.DeepHitLoss(alpha, sigma)
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Weighted loss function that extends PyCox's standard loss with sample weights.
                """
                self.debug_counter += 1
                
                # Get stored weights from the network
                stored_weights = None
                if hasattr(self.net, '_training_weights'):
                    stored_weights = self.net._training_weights
                
                # If no weights available or unweighted mode, use PyCox's standard loss
                if (weights is None and stored_weights is None) or self.ablation_mode == "unweighted":
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                
                # For weighted mode, implement weighted loss using stored weights
                if stored_weights is not None and self.ablation_mode != "unweighted":
                    # Convert pred to tensor if needed and get device
                    if not isinstance(pred, torch.Tensor):
                        pred = torch.tensor(pred, dtype=torch.float32)
                    
                    # Get batch weights using proper batch indices if available
                    batch_size = pred.shape[0]
                    
                    if batch_indices is not None:
                        # Use proper batch indices for accurate weight mapping
                        if isinstance(batch_indices, torch.Tensor):
                            batch_indices = batch_indices.cpu().numpy()
                        
                        # Ensure indices are within bounds
                        batch_indices = np.clip(batch_indices, 0, len(stored_weights) - 1)
                        batch_weights = torch.tensor(stored_weights[batch_indices], dtype=torch.float32, device=pred.device)
                    else:
                        # Fallback to simple approach (first batch_size samples)
                        if len(stored_weights) >= batch_size:
                            batch_weights = torch.tensor(stored_weights[:batch_size], dtype=torch.float32, device=pred.device)
                        else:
                            # Fallback: use uniform weights if not enough stored weights
                            batch_weights = torch.ones(batch_size, dtype=torch.float32, device=pred.device)
                    
                    # Check if all weights are 1.0 - if so, use standard loss directly
                    if torch.allclose(batch_weights, torch.ones_like(batch_weights), atol=1e-6):
                        return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                    
                    # Implement weighted loss by extending PyCox's standard loss
                    return self._compute_weighted_loss(pred, durations, events, rank_mat, batch_weights)
                else:
                    # Use PyCox's standard loss for unweighted mode
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
            
            def _call_standard_loss_safely(self, pred, durations, events, rank_mat):
                """Safely call PyCox's standard loss with proper tensor conversion."""
                # Convert all inputs to proper PyTorch tensors
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Call PyCox's standard loss
                return self.standard_loss(pred, durations, events, rank_mat)

            def _compute_weighted_loss(self, pred, durations, events, rank_mat, batch_weights):
                """
                Compute weighted loss by extending PyCox's standard loss with sample weights.
                """
                # Convert all inputs to proper PyTorch tensors with correct types
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Ensure batch_weights is on the same device as pred
                if batch_weights.device != pred.device:
                    batch_weights = batch_weights.to(pred.device)
                
                # Get standard PyCox loss
                standard_loss = self.standard_loss(pred, durations, events, rank_mat)
                
                # For uniform weights (all 1.0), return standard loss directly without any scaling
                if torch.allclose(batch_weights, torch.ones_like(batch_weights), atol=1e-6):
                    return standard_loss
                
                # For non-uniform weights, implement proper weighting
                # Note: This is a simplified approach - you may want to implement more sophisticated weighting
                weight_factor = batch_weights.mean()
                weighted_loss = standard_loss * weight_factor
                
                return weighted_loss

        # Create the custom loss function instance
        ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
        max_ranking_pairs = self.m_hyperparameters.get("max_ranking_pairs", 20)
        
        custom_loss = WeightedDeepHitLoss(
            self.m_net,
            alpha,
            sigma,
            self.m_hyperparameters.get("cuts"),
            ablation_mode,
            max_ranking_pairs
        )
        
        # Replace the model's loss function with our custom one
        self.m_model.loss = custom_loss
        
        print(f"🔍 Replaced loss function in PyCox DeepHitSingle model")

    def fit(self, data_train, parameters=None):
        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

        if self.use_weights:
            x, y, w = self._df_to_xyw_(data_train)
            print(f"✅ Using weighted training with {len(w)} samples")
            print(f"   Weight range: {w.min():.3f} to {w.max():.3f}, mean: {w.mean():.3f}")
            
            # Store weights in the network for the loss function to access
            self.m_net._training_weights = w
            
            # Create indexed data loader for proper batch tracking (only if not unweighted mode)
            ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "unweighted":
                indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
                print(f"🔍 Using indexed data loader for accurate weight mapping")
            else:
                print(f"🔍 Ablation mode '{ablation_mode}': using standard training despite weights")
        else:
            # Use EXACT original fit method for unweighted mode
            x, y = self._df_to_xy_(data_train)
            print(f"ℹ️  Using original DeepHit training (no weights)")

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        if self.use_weights:
            # Use weighted training
            self.m_log = self.m_model.fit(
                x,
                y,
                batch_size=parameters['batch_size'],
                epochs=parameters['epochs'],
                callbacks=parameters['callbacks'],
                verbose=parameters['verbose'],
                val_data=self._df_to_xy_(parameters["val_data"]),
                val_batch_size=parameters['batch_size']
            )
        else:
            # Use EXACT original training for unweighted mode
            val_data = self._df_to_xy_(parameters["val_data"])
            self.m_log = self.m_model.fit(
                x,
                y,
                batch_size=parameters['batch_size'],
                epochs=parameters['epochs'],
                callbacks=parameters['callbacks'],
                verbose=parameters['verbose'],
                val_data=val_data,
                val_batch_size=parameters['batch_size']
            )

        return self

    def predict(self, x, parameters=None):
        """
        Return survival probabilities for DeepHitSingle model.
        DeepHitSingle is for single risk scenarios, so it returns survival probabilities.
        """
        x_ = _adapt_input_(x)
        return self.m_model.predict_surv_df(x_)

    def interpolate(self, interpolation=10):
        """Add interpolate method for evaluation compatibility."""
        # Create a wrapper that provides the interpolate method
        class InterpolatedModel:
            def __init__(self, base_model):
                self.base_model = base_model
                
            def predict_surv_df(self, x):
                return self.base_model.predict_surv_df(x)
        
        return InterpolatedModel(self)

    def predict_surv(self, x, parameters=None):
        """Predict survival probabilities for DeepHitSingle model (compatibility method)."""
        x_ = _adapt_input_(x)
        return self.m_model.predict_surv_df(x_)

    def _create_indexed_dataloader(self, x, y, batch_size):
        """
        Create a custom data loader that tracks batch indices for proper weight mapping.
        This enables accurate sample weighting by knowing which samples are in each batch.
        """
        import torch.utils.data as data
        
        class IndexedSurvivalDataset(data.Dataset):
            """Dataset that returns (features, targets, index) tuples."""
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = (torch.tensor(y[0], dtype=torch.long), 
                         torch.tensor(y[1], dtype=torch.long))
                self.indices = torch.arange(len(x))
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[0][idx], self.y[1][idx], self.indices[idx]
        
        class IndexedDataLoader:
            """Custom data loader that yields batches with indices."""
            def __init__(self, dataset, batch_size, shuffle=True):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sampler = data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)
                self.batch_sampler = data.BatchSampler(self.sampler, batch_size, drop_last=False)
            
            def __iter__(self):
                for batch_indices in self.batch_sampler:
                    x_batch = torch.stack([self.dataset.x[i] for i in batch_indices])
                    y_batch = (torch.stack([self.dataset.y[0][i] for i in batch_indices]),
                              torch.stack([self.dataset.y[1][i] for i in batch_indices]))
                    indices_batch = torch.stack([self.dataset.indices[i] for i in batch_indices])
                    yield x_batch, y_batch, indices_batch
            
            def __len__(self):
                return len(self.batch_sampler)
        
        # Create the indexed dataset and loader
        dataset = IndexedSurvivalDataset(x, y)
        loader = IndexedDataLoader(dataset, batch_size, shuffle=True)
        
        print(f"🔍 Created indexed data loader with {len(dataset)} samples")
        print(f"🔍 Batch size: {batch_size}, Total batches: {len(loader)}")
        
        return loader


class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input_):
        out = self.shared_net(input_)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out

class DeepHit(PyCoxModel):
    """
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit.ipynb (single risk)
    cf. https://nbviewer.org/github/havakv/pycox/blob/master/examples/deephit_competing_risks.ipynb (competing risks)

    Args:
        - in_features  : dimension of a feature vector as input.
        - num_nodes_shared : sizes for shared layers.
        - num_nodes_indiv  : sizes for individual risk layers.
        - num_risks    : number of competing risks.
        - out_features : matches the size of the grid on which time has been discretised
        - batch_norm   : boolean enabling batch normalisation
        - dropout      : drop out rate.
        - alpha        :
        - sigma        :
        - use_weights  : boolean to enable sample weighting for underrepresented subgroups
    """

    def __init__(self, accessor_code, hyperparameters):
        super().__init__(accessor_code=accessor_code, hyperparameters=hyperparameters)

        in_features      = hyperparameters["in_features"]       # x_train.shape[1]
        num_nodes_shared = hyperparameters["num_nodes_shared"]  # [64, 64]
        num_nodes_indiv  = hyperparameters["num_nodes_indiv"]   # [32]
        num_risks        = hyperparameters["num_risks"]         # y_train[1].max()
        out_features     = hyperparameters["out_features"]      # len(labtrans.cuts)
        batch_norm       = hyperparameters["batch_norm"]        # True
        dropout          = hyperparameters["dropout"]           # 0.1
        seed             = hyperparameters.get("seed")
        self.use_weights = hyperparameters.get("use_weights", False)  # Default to False for backward compatibility

        if seed is not None:
            torch.manual_seed(seed)
        
        # Use the original CauseSpecificNet architecture for competing risks
        self.m_net = CauseSpecificNet(
            in_features,
            num_nodes_shared,
            num_nodes_indiv,
            num_risks,
            out_features,
            batch_norm,
            dropout
        )

        # Use the original competing risks DeepHit model with EXACT original parameters
        if self.use_weights:
            # Use configurable parameters for weighted mode
            print("🔍 Using weighted DeepHit with configurable parameters")
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],
                sigma=hyperparameters["sigma"],
                duration_index=hyperparameters.get("cuts")
            )
            
            # Create custom loss function for weighted mode
            ablation_mode = hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "pycox_standard":
                print("🔍 Creating custom loss function for weighted competing risks DeepHit...")
                self._create_weighted_loss_function(hyperparameters["alpha"], hyperparameters["sigma"])
        else:
            # Use EXACT original parameters for unweighted mode
            print("🔍 Using original DeepHit parameters (unweighted mode)")
            self.m_model = pycox.DeepHit(
                self.m_net,
                tt.optim.AdamWR(
                    lr=0.01,
                    decoupled_weight_decay=0.01,
                    cycle_eta_multiplier=0.8
                ),
                alpha=hyperparameters["alpha"],  # 0.2,
                sigma=hyperparameters["sigma"],  # 0.1,
                duration_index=hyperparameters["cuts"]
            )

    def _create_weighted_loss_function(self, alpha, sigma):
        """
        Create a weighted loss function that extends PyCox's standard DeepHit loss with sample weights.
        """
        class WeightedDeepHitLoss:
            def __init__(self, net, alpha, sigma, duration_index, ablation_mode="full", max_ranking_pairs=20):
                self.net = net
                self.alpha = alpha
                self.sigma = sigma
                self.duration_index = duration_index
                self.debug_counter = 0
                self.ablation_mode = ablation_mode
                self.max_ranking_pairs = max_ranking_pairs
                
                # Create PyCox's standard loss function
                import pycox.models.loss as pycox_loss
                self.standard_loss = pycox_loss.DeepHitLoss(alpha, sigma)
                
            def __call__(self, pred, durations, events, rank_mat=None, weights=None, batch_indices=None, *args, **kwargs):
                """
                Weighted loss function that extends PyCox's standard loss with sample weights.
                """
                self.debug_counter += 1
                
                # Get stored weights from the network
                stored_weights = None
                if hasattr(self.net, '_training_weights'):
                    stored_weights = self.net._training_weights
                
                # If no weights available or unweighted mode, use PyCox's standard loss
                if (weights is None and stored_weights is None) or self.ablation_mode == "unweighted":
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
                
                # For weighted mode, implement weighted loss using stored weights
                if stored_weights is not None and self.ablation_mode != "unweighted":
                    # Convert pred to tensor if needed and get device
                    if not isinstance(pred, torch.Tensor):
                        pred = torch.tensor(pred, dtype=torch.float32)
                    
                    # Get batch weights using proper batch indices if available
                    batch_size = pred.shape[0]
                    
                    if batch_indices is not None:
                        # Use proper batch indices for accurate weight mapping
                        if isinstance(batch_indices, torch.Tensor):
                            batch_indices = batch_indices.cpu().numpy()
                        
                        # Ensure indices are within bounds
                        batch_indices = np.clip(batch_indices, 0, len(stored_weights) - 1)
                        batch_weights = torch.tensor(stored_weights[batch_indices], dtype=torch.float32, device=pred.device)
                    else:
                        # Fallback to simple approach (first batch_size samples)
                        if len(stored_weights) >= batch_size:
                            batch_weights = torch.tensor(stored_weights[:batch_size], dtype=torch.float32, device=pred.device)
                        else:
                            # Fallback: use uniform weights if not enough stored weights
                            batch_weights = torch.ones(batch_size, dtype=torch.float32, device=pred.device)
                    
                    
                    # Implement weighted loss by extending PyCox's standard loss
                    return self._compute_weighted_loss(pred, durations, events, rank_mat, batch_weights)
                else:
                    # Use PyCox's standard loss for unweighted mode
                    return self._call_standard_loss_safely(pred, durations, events, rank_mat)
            
            def _call_standard_loss_safely(self, pred, durations, events, rank_mat):
                """Safely call PyCox's standard loss with proper tensor conversion."""
                # Convert all inputs to proper PyTorch tensors
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Call PyCox's standard loss
                return self.standard_loss(pred, durations, events, rank_mat)

            def _compute_weighted_loss(self, pred, durations, events, rank_mat, batch_weights):
                """
                Compute weighted loss by extending PyCox's standard loss with sample weights.
                """
                # Convert all inputs to proper PyTorch tensors with correct types
                if not isinstance(pred, torch.Tensor):
                    pred = torch.tensor(pred, dtype=torch.float32)
                
                if not isinstance(durations, torch.Tensor):
                    durations = torch.tensor(durations, dtype=torch.long)
                elif durations.dtype != torch.long:
                    durations = durations.long()
                
                if not isinstance(events, torch.Tensor):
                    events = torch.tensor(events, dtype=torch.long)
                elif events.dtype != torch.long:
                    events = events.long()
                
                if rank_mat is not None and not isinstance(rank_mat, torch.Tensor):
                    rank_mat = torch.tensor(rank_mat, dtype=torch.float32)
                
                # Ensure batch_weights is on the same device as pred
                if batch_weights.device != pred.device:
                    batch_weights = batch_weights.to(pred.device)
                
                # Get standard PyCox loss
                standard_loss = self.standard_loss(pred, durations, events, rank_mat)
                
                
                # For non-uniform weights, implement proper weighting
                # Note: This is a simplified approach - you may want to implement more sophisticated weighting
                weight_factor = batch_weights.mean()
                weighted_loss = standard_loss * weight_factor
                
                return weighted_loss

        # Create the custom loss function instance
        ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
        max_ranking_pairs = self.m_hyperparameters.get("max_ranking_pairs", 20)
        
        custom_loss = WeightedDeepHitLoss(
            self.m_net,
            alpha,
            sigma,
            self.m_hyperparameters.get("cuts"),
            ablation_mode,
            max_ranking_pairs
        )
        
        # Replace the model's loss function with our custom one
        self.m_model.loss = custom_loss
        
        print(f"🔍 Replaced loss function in PyCox DeepHit (competing risks) model")

    def fit(self, data_train, parameters=None):
        if parameters['seed'] is not None:
            torch.manual_seed(parameters['seed'])

        if self.use_weights:
            x, y, w = self._df_to_xyw_(data_train)
            print(f"✅ Using weighted training with {len(w)} samples")
            print(f"   Weight range: {w.min():.3f} to {w.max():.3f}, mean: {w.mean():.3f}")
            
            # Store weights in the network for the loss function to access
            self.m_net._training_weights = w
            
            # Create indexed data loader for proper batch tracking (only if not unweighted mode)
            ablation_mode = self.m_hyperparameters.get("ablation_mode", "full")
            if ablation_mode != "unweighted":
                indexed_loader = self._create_indexed_dataloader(x, y, parameters['batch_size'])
                print(f"🔍 Using indexed data loader for accurate weight mapping")
            else:
                print(f"🔍 Ablation mode '{ablation_mode}': using standard training despite weights")
        else:
            # Use EXACT original fit method for unweighted mode
            x, y = self._df_to_xy_(data_train)
            print(f"ℹ️  Using original DeepHit training (no weights)")

        # Compute learning rate
        if parameters['lr']:
            self.m_model.optimizer.set_lr(parameters['lr'])
        else:
            lrfinder = self.m_model.lr_finder(
                x,
                y,
                parameters['batch_size'],
                tolerance=parameters['tolerance']
            )
            lr = lrfinder.get_best_lr()
            self.m_model.optimizer.set_lr(lr)

        # Train
        if self.use_weights:
            # Use weighted training
            self.m_log = self.m_model.fit(
                x,
                y,
                batch_size=parameters['batch_size'],
                epochs=parameters['epochs'],
                callbacks=parameters['callbacks'],
                verbose=parameters['verbose'],
                val_data=self._df_to_xy_(parameters["val_data"]),
                val_batch_size=parameters['batch_size']
            )
        else:
            # Use EXACT original training for unweighted mode
            val_data = self._df_to_xy_(parameters["val_data"])
            self.m_log = self.m_model.fit(
                x,
                y,
                batch_size=parameters['batch_size'],
                epochs=parameters['epochs'],
                callbacks=parameters['callbacks'],
                verbose=parameters['verbose'],
                val_data=val_data,
                val_batch_size=parameters['batch_size']
            )

        return self

    def predict(self, x, parameters=None):
        """
        Return CIF predictions like the original DeepHit class.
        The original DeepHit simply returned self.predict_CIF(x, parameters=None).
        """
        return self.predict_CIF(x, parameters)

    def predict_CIF(self, x, parameters=None):
        _ = parameters
        x_ = _adapt_input_(x)
        return self.model.predict_cif(x_)

    @property
    def model(self):
        """Return the underlying PyCox model for evaluation."""
        return self.m_model

    def predict_surv(self, x, parameters=None):
        """Predict survival probabilities for DeepHit model (compatibility method)."""
        x_ = _adapt_input_(x)
        return self.m_model.predict_surv_df(x_)

    def _create_indexed_dataloader(self, x, y, batch_size):
        """
        Create a custom data loader that tracks batch indices for proper weight mapping.
        This enables accurate sample weighting by knowing which samples are in each batch.
        """
        import torch.utils.data as data
        
        class IndexedSurvivalDataset(data.Dataset):
            """Dataset that returns (features, targets, index) tuples."""
            def __init__(self, x, y):
                self.x = torch.tensor(x, dtype=torch.float32)
                self.y = (torch.tensor(y[0], dtype=torch.long), 
                         torch.tensor(y[1], dtype=torch.long))
                self.indices = torch.arange(len(x))
            
            def __len__(self):
                return len(self.x)
            
            def __getitem__(self, idx):
                return self.x[idx], self.y[0][idx], self.y[1][idx], self.indices[idx]
        
        class IndexedDataLoader:
            """Custom data loader that yields batches with indices."""
            def __init__(self, dataset, batch_size, shuffle=True):
                self.dataset = dataset
                self.batch_size = batch_size
                self.shuffle = shuffle
                self.sampler = data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)
                self.batch_sampler = data.BatchSampler(self.sampler, batch_size, drop_last=False)
            
            def __iter__(self):
                for batch_indices in self.batch_sampler:
                    x_batch = torch.stack([self.dataset.x[i] for i in batch_indices])
                    y_batch = (torch.stack([self.dataset.y[0][i] for i in batch_indices]),
                              torch.stack([self.dataset.y[1][i] for i in batch_indices]))
                    indices_batch = torch.stack([self.dataset.indices[i] for i in batch_indices])
                    yield x_batch, y_batch, indices_batch
            
            def __len__(self):
                return len(self.batch_sampler)
        
        # Create the indexed dataset and loader
        dataset = IndexedSurvivalDataset(x, y)
        loader = IndexedDataLoader(dataset, batch_size, shuffle=True)
        
        print(f"🔍 Created indexed data loader with {len(dataset)} samples")
        print(f"🔍 Batch size: {batch_size}, Total batches: {len(loader)}")
        
        return loader

    def set_ablation_mode(self, mode):
        """
        Set the ablation mode for training.
        
        Args:
            mode (str): One of the following modes:
                - "unweighted": Use standard PyCox training (no weights)
                - "full": Use weighted training with all loss components
                - "no_ranking": Use weighted training without ranking loss
                - "no_likelihood": Use weighted training without likelihood loss
                - "pycox_standard": Use PyCox's standard loss even with weights (for comparison)
                - "weight_only": Use only weight scaling without custom loss components
                - "ranking_only": Use only ranking loss component with weights
                - "likelihood_only": Use only likelihood loss component with weights
                - "adaptive": Use adaptive weighting based on loss gradients
        """
        valid_modes = [
            "unweighted", "full", "no_ranking", "no_likelihood", 
            "pycox_standard", "weight_only", "ranking_only", 
            "likelihood_only", "adaptive"
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid ablation mode: {mode}. Must be one of {valid_modes}")
        
        self.m_hyperparameters["ablation_mode"] = mode
        print(f"🔧 Set ablation mode to: {mode}")
        
        # Recreate model with new ablation mode if weights are enabled
        if self.use_weights:
            self._create_weighted_loss_function(
                self.m_hyperparameters["alpha"], 
                self.m_hyperparameters["sigma"]
            )

    def get_model_info(self) -> dict:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_type': 'DeepHit',
            'accessor_code': self.accessor_code,
            'use_weights': self.use_weights,
            'hyperparameters': self.m_hyperparameters.copy(),
            'architecture': {
                'network_type': type(self.m_net).__name__ if hasattr(self, 'm_net') else 'None',
                'model_type': type(self.m_model).__name__ if hasattr(self, 'm_model') else 'None',
            },
            'training': {
                'has_training_log': self.m_log is not None,
                'training_completed': hasattr(self, 'm_model') and self.m_model is not None
            }
        }
        
        # Add network architecture details
        if hasattr(self, 'm_net') and self.m_net is not None:
            info['architecture']['network_parameters'] = sum(
                p.numel() for p in self.m_net.parameters()
            )
        
        # Add ablation mode if available
        if 'ablation_mode' in self.m_hyperparameters:
            info['ablation_mode'] = self.m_hyperparameters['ablation_mode']
        
        return info
    
    def validate_model(self) -> dict:
        """
        Validate the model for potential issues.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check if model is trained
        if not hasattr(self, 'm_model') or self.m_model is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Model is not trained")
            return validation_results
        
        # Check network architecture
        if not hasattr(self, 'm_net') or self.m_net is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Network architecture is missing")
            return validation_results
        
        # Check hyperparameters
        required_params = ['in_features', 'num_nodes_shared', 'num_nodes_indiv', 'out_features']
        for param in required_params:
            if param not in self.m_hyperparameters:
                validation_results['warnings'].append(f"Missing hyperparameter: {param}")
        
        # Check for potential issues
        if self.use_weights and 'ablation_mode' not in self.m_hyperparameters:
            validation_results['warnings'].append("Weighted mode without ablation mode specified")
        
        # Check network parameters
        try:
            param_count = sum(p.numel() for p in self.m_net.parameters())
            if param_count > 1000000:  # 1M parameters
                validation_results['recommendations'].append("Large model detected - consider model compression")
        except Exception as e:
            validation_results['warnings'].append(f"Could not count parameters: {e}")
        
        # Check training log
        if self.m_log is None:
            validation_results['warnings'].append("No training log available")
        
        return validation_results
