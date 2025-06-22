import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import random
import time
import math # For ceiling in adaptive pooling calculation

# --- Set Pandas option ---
pd.set_option('future.no_silent_downcasting', True)

# --- Determine Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# --- MLP Model (Model2 - Reference) ---
class Model2(nn.Module):
    # ... (Keep Model2 as is for comparison if needed) ...
    def __init__(self, in_features, h1, h2, h3, h4,
                 drop1_rate, drop2_rate, drop3_rate, drop4_rate,
                 out_features=11):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(drop1_rate)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(drop2_rate)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.drop3 = nn.Dropout(drop3_rate)
        self.fc4 = nn.Linear(h3, h4)
        self.bn4 = nn.BatchNorm1d(h4)
        self.drop4 = nn.Dropout(drop4_rate)
        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), 0.01))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), 0.01))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x)), 0.01))
        x = self.drop4(F.leaky_relu(self.bn4(self.fc4(x)), 0.01))
        x = self.out(x)
        return x

# --- Revised 1D CNN Model ---
class CNNModelRevised(nn.Module):
    def __init__(self, in_channels, initial_sequence_length,
                 # Conv Block 1
                 c1_filters, c1_kernel, c1_dropout,
                 pool1_kernel,
                 # Conv Block 2
                 c2_filters, c2_kernel, c2_dropout,
                 pool2_kernel,
                 # Conv Block 3 (Optional, can be controlled by setting filters to 0)
                 c3_filters, c3_kernel, c3_dropout,
                 pool3_kernel,
                 # FC Layers
                 fc1_units, fc_dropout,
                 out_features=11):
        super().__init__()

        self.layers = nn.ModuleList()
        current_channels = in_channels
        current_seq_len = initial_sequence_length

        # Conv Block 1
        self.layers.append(nn.Conv1d(current_channels, c1_filters, c1_kernel, padding='same'))
        self.layers.append(nn.BatchNorm1d(c1_filters))
        self.layers.append(nn.LeakyReLU(0.1)) # Changed to LeakyReLU, can also try ReLU
        if pool1_kernel > 1: # Only add pooling if kernel > 1
            self.layers.append(nn.MaxPool1d(pool1_kernel))
            current_seq_len = current_seq_len // pool1_kernel
        if c1_dropout > 0: self.layers.append(nn.Dropout(c1_dropout))
        current_channels = c1_filters

        # Conv Block 2
        self.layers.append(nn.Conv1d(current_channels, c2_filters, c2_kernel, padding='same'))
        self.layers.append(nn.BatchNorm1d(c2_filters))
        self.layers.append(nn.LeakyReLU(0.1))
        if pool2_kernel > 1:
            self.layers.append(nn.MaxPool1d(pool2_kernel))
            current_seq_len = current_seq_len // pool2_kernel
        if c2_dropout > 0: self.layers.append(nn.Dropout(c2_dropout))
        current_channels = c2_filters

        # Conv Block 3 (Optional)
        if c3_filters > 0 and c3_kernel > 0: # Check if block 3 is active
            self.layers.append(nn.Conv1d(current_channels, c3_filters, c3_kernel, padding='same'))
            self.layers.append(nn.BatchNorm1d(c3_filters))
            self.layers.append(nn.LeakyReLU(0.1))
            if pool3_kernel > 1:
                self.layers.append(nn.MaxPool1d(pool3_kernel))
                current_seq_len = current_seq_len // pool3_kernel
            if c3_dropout > 0: self.layers.append(nn.Dropout(c3_dropout))
            current_channels = c3_filters

        self.layers.append(nn.Flatten())

        flattened_features = current_channels * current_seq_len
        if flattened_features <= 0:
            raise ValueError(f"Flattened features non-positive ({flattened_features}). SeqLen: {current_seq_len}, Channels: {current_channels}")

        self.layers.append(nn.Linear(flattened_features, fc1_units))
        self.layers.append(nn.BatchNorm1d(fc1_units))
        self.layers.append(nn.LeakyReLU(0.1))
        if fc_dropout > 0: self.layers.append(nn.Dropout(fc_dropout))

        self.layers.append(nn.Linear(fc1_units, out_features))

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1) # (batch, features) -> (batch, 1, features)
        for layer in self.layers:
            x = layer(x)
        return x


# --- Data Loading and Preprocessing ---
# ... (Keep data loading and preprocessing as is) ...
try:
    trainDF_orig = pd.read_csv("train10(11000)(1)(extractedALT6).csv")
    testDF_orig = pd.read_csv("test90(11000)(1)(extractedALT6).csv")
except FileNotFoundError:
    print("Error: One or both data files are not found. Please check file paths.")
    exit()

if trainDF_orig.empty or testDF_orig.empty:
    print("Error: One or both dataframes are empty. Exiting.")
    exit()

trainDF = trainDF_orig.copy()
testDF = testDF_orig.copy()

variant_to_float = {
    'Alpha': 0.0, 'Beta': 1.0, 'Gamma': 2.0, 'Delta': 3.0, 'Epsilon': 4.0,
    'Zeta': 5.0, 'Eta': 6.0, 'Iota': 7.0, 'Lambda': 8.0, 'Mu': 9.0, 'Omicron': 10.0
}
float_to_variant = {v: k for k, v in variant_to_float.items()}

trainDF['Variant'] = trainDF['Variant'].replace(variant_to_float).infer_objects(copy=False)
testDF['Variant'] = testDF['Variant'].replace(variant_to_float).infer_objects(copy=False)

try:
    trainDF['Variant'] = pd.to_numeric(trainDF['Variant'], errors='raise')
    testDF['Variant'] = pd.to_numeric(testDF['Variant'], errors='raise')
except ValueError as e:
    print(f"Error converting 'Variant' column to numeric: {e}")
    exit()

if trainDF['Variant'].isnull().any() or testDF['Variant'].isnull().any():
    print("Warning: NaNs found in 'Variant' column. Dropping rows with NaNs in 'Variant'.")
    trainDF.dropna(subset=['Variant'], inplace=True)
    testDF.dropna(subset=['Variant'], inplace=True)
    if trainDF.empty or testDF.empty:
        print("Error: DataFrames became empty after dropping NaNs. Exiting.")
        exit()

X_train_df_full = trainDF.iloc[:, 1:-1]
y_train_df_full = trainDF.iloc[:, -1]
X_test_df = testDF.iloc[:, 1:-1]
y_test_df = testDF.iloc[:, -1]

if X_train_df_full.empty or X_test_df.empty:
    print("Error: Feature set (X_train_df_full or X_test_df) is empty. Check CSV structure and slicing.")
    exit()

in_features_mlp = X_train_df_full.shape[1] # For MLP if used
initial_sequence_length_cnn = X_train_df_full.shape[1] # For CNN
print(f"Initial sequence length for CNN (number of k-mer features): {initial_sequence_length_cnn}")
out_features = len(variant_to_float)

scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_df_full)
X_test_scaled = scaler.transform(X_test_df)

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_train_scaled_full, y_train_df_full.values,
    test_size=0.15, random_state=42, stratify=y_train_df_full.values
)

X_test_np = X_test_scaled
y_test_np = y_test_df.values

X_train_tensor = torch.FloatTensor(X_train_np)
y_train_tensor = torch.LongTensor(y_train_np)
X_val_tensor = torch.FloatTensor(X_val_np)
y_val_tensor = torch.LongTensor(y_val_np)
X_test_tensor = torch.FloatTensor(X_test_np)
y_test_tensor = torch.LongTensor(y_test_np)

batch_size = 64
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

# --- Ensemble Configuration ---
models = []
optimizers = []
schedulers = []
model_configs = []
train_loaders_ensemble = []

modelsUsed = 1  # Let's try one well-configured CNN first
USE_CNN_MODEL = True # Set to True for CNNModelRevised

# --- More CONSTRAINED and FOCUSED Hyperparameter Ranges for CNNModelRevised ---
# Conv Block 1
c1_filters_options = [32, 64, 128]
c1_kernel_options = [3, 5, 7]
c1_dropout_options = [0.0, 0.1, 0.2] # Less dropout in early conv layers
pool1_kernel_options = [1, 2] # 1 means no pooling

# Conv Block 2
c2_filters_options = [64, 128, 256] # Often increase filters in deeper layers
c2_kernel_options = [3, 5]
c2_dropout_options = [0.1, 0.2, 0.25]
pool2_kernel_options = [1, 2]

# Conv Block 3 (Optional - set filters to 0 to disable)
c3_filters_options = [0, 128, 256, 512] # 0 to disable this block
c3_kernel_options = [3]
c3_dropout_options = [0.2, 0.3]
pool3_kernel_options = [1, 2]


# FC Layer
fc1_units_options = [128, 256, 512]
fc_dropout_options = [0.25, 0.3, 0.4, 0.5] # Higher dropout in FC layers is common

# Optimizer
lr_options = [5e-5, 1e-4, 2e-4, 5e-4] # More focused LR range
weight_decay_options = [1e-5, 5e-5, 1e-4]
# L1 regularization (l1_lambda) can be added if needed, but let's focus on architecture first

print(f"\n--- Configuring {modelsUsed} Ensemble Models ({'CNN Revised' if USE_CNN_MODEL else 'MLP'}) ---")

for i in range(modelsUsed): # For now, let's try a single, more carefully chosen config
    if USE_CNN_MODEL:
        # Example Fixed Configuration (Manually tune this based on intuition/trials)
        # You can replace this with random.choice from the options above for a search
        config = {
            'model_type': 'CNN_Revised',
            'c1_filters': random.choice(c1_filters_options), #64,
            'c1_kernel': random.choice(c1_kernel_options), #5,
            'c1_dropout': random.choice(c1_dropout_options), #0.1,
            'pool1_kernel': random.choice(pool1_kernel_options),#2,

            'c2_filters': random.choice(c2_filters_options), #128,
            'c2_kernel': random.choice(c2_kernel_options), #3,
            'c2_dropout': random.choice(c2_dropout_options), #0.2,
            'pool2_kernel': random.choice(pool2_kernel_options),#2,

            'c3_filters': random.choice(c3_filters_options), #0, # Set to 0 to disable 3rd conv block
            'c3_kernel': random.choice(c3_kernel_options), #3,
            'c3_dropout': random.choice(c3_dropout_options), #0.0,
            'pool3_kernel': random.choice(pool3_kernel_options), #1,

            'fc1_units': random.choice(fc1_units_options), #256,
            'fc_dropout': random.choice(fc_dropout_options), #0.3,

            'lr': random.choice(lr_options), #0.0001,
            'weight_decay': random.choice(weight_decay_options), #0.00005,
            # 'l1_lambda': 0 # For now
        }
        # Ensure c3_kernel is valid if c3_filters > 0
        if config['c3_filters'] == 0:
            config['c3_kernel'] = 0 # No kernel if no filters
            config['c3_dropout'] = 0.0
            config['pool3_kernel'] = 1


        print(f"Attempting CNN Config: {config}")
        try:
            model = CNNModelRevised(in_channels=1,
                                   initial_sequence_length=initial_sequence_length_cnn,
                                   c1_filters=config['c1_filters'], c1_kernel=config['c1_kernel'],
                                   c1_dropout=config['c1_dropout'], pool1_kernel=config['pool1_kernel'],
                                   c2_filters=config['c2_filters'], c2_kernel=config['c2_kernel'],
                                   c2_dropout=config['c2_dropout'], pool2_kernel=config['pool2_kernel'],
                                   c3_filters=config['c3_filters'], c3_kernel=config['c3_kernel'],
                                   c3_dropout=config['c3_dropout'], pool3_kernel=config['pool3_kernel'],
                                   fc1_units=config['fc1_units'], fc_dropout=config['fc_dropout'],
                                   out_features=out_features)
            model.to(device)
            model_configs.append(config)
            models.append(model)
        except ValueError as e:
            print(f"Skipping CNN config due to error: {e}")
            print(f"Problematic config was: {config}")
            continue
    else: # MLP (Model2)
        # ... (Your MLP config code, keep as is if you want to run MLP) ...
        # For this focused run, I'll assume USE_CNN_MODEL is True
        print("MLP configuration skipped as USE_CNN_MODEL is True.")
        pass


    if models and len(models) == len(model_configs): # If model was successfully added
        optimizer = torch.optim.AdamW(models[-1].parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        optimizers.append(optimizer)
        schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=10, min_lr=1e-7, verbose=True # Reduced patience, added verbose
        ))

        # Bootstrap for ensemble (though we're running 1 model now for focused tuning)
        n_samples_train = X_train_tensor.shape[0]
        if n_samples_train == 0: print("Error: No training samples. Exiting."); exit()
        bootstrap_indices = torch.randint(0, n_samples_train, (n_samples_train,))
        X_train_bootstrap = X_train_tensor[bootstrap_indices]
        y_train_bootstrap = y_train_tensor[bootstrap_indices]
        train_dataset_bootstrap = TensorDataset(X_train_bootstrap, y_train_bootstrap)
        train_loader_bootstrap = DataLoader(train_dataset_bootstrap, batch_size=batch_size, shuffle=True,
                                            pin_memory=torch.cuda.is_available())
        train_loaders_ensemble.append(train_loader_bootstrap)

if not models:
    print("No models were configured successfully. Exiting.")
    exit()
modelsUsed = len(models)


criterion = nn.CrossEntropyLoss()

# EarlyStopping class (Unchanged)
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print): # Increased patience
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0 : # Print more frequently if verbose
                 self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

epochs = 300 # Reduced epochs for faster iteration during tuning, can increase later

print("\n--- Training Ensemble Models ---")
for model_idx in range(modelsUsed):
    print(f"\n--- Training Model {model_idx+1}/{modelsUsed} ({model_configs[model_idx]['model_type']}) ---")
    config_str = ", ".join([f"{k}={v}" for k, v in model_configs[model_idx].items()])
    print(f"Config: {config_str}")

    model = models[model_idx]
    optimizer = optimizers[model_idx]
    scheduler = schedulers[model_idx]
    current_train_loader = train_loaders_ensemble[model_idx]
    # current_l1_lambda_val = model_configs[model_idx]['l1_lambda'] # L1 removed for now for simplicity

    early_stopper = EarlyStopping(patience=25, verbose=True, path=f'model_{model_idx}_checkpoint.pt') # Increased patience, verbose
    epoch_print_interval = 10 # Print every 10 epochs

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in current_train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # if current_l1_lambda_val > 0: # L1 removed for simplicity
            #     l1_norm = sum(p.abs().sum() for p in model.parameters())
            #     loss = loss + current_l1_lambda_val * l1_norm

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(current_train_loader) if len(current_train_loader) > 0 else 0

        model.eval()
        epoch_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                batch_X_val, batch_y_val = batch_X_val.to(device), batch_y_val.to(device)
                y_pred_val = model(batch_X_val)
                val_loss_item = criterion(y_pred_val, batch_y_val)
                epoch_val_loss += val_loss_item.item()
                _, predicted_val = torch.max(y_pred_val.data, 1)
                total_val += batch_y_val.size(0)
                correct_val += (predicted_val == batch_y_val).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = 100 * correct_val / total_val if total_val > 0 else 0
        scheduler.step(avg_val_loss) # Pass validation loss to scheduler

        if epoch % epoch_print_interval == 0 or epoch == epochs -1 :
            epoch_time_taken = time.time() - epoch_start_time
            print(f'M{model_idx+1} Ep:{epoch:03d}/{epochs} | TrL:{avg_train_loss:.4f} | VaL:{avg_val_loss:.4f} | VaAcc:{val_accuracy:.2f}% | LR:{optimizer.param_groups[0]["lr"]:.1e} | Time:{epoch_time_taken:.2f}s')

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Model {model_idx+1} early stopping at epoch {epoch}.")
            model.load_state_dict(torch.load(early_stopper.path, map_location=device))
            break
    if not early_stopper.early_stop:
        try:
            model.load_state_dict(torch.load(early_stopper.path, map_location=device))
        except FileNotFoundError:
            print(f"Warning: Checkpoint file {early_stopper.path} not found for model {model_idx+1}.")

    print(f"Model {model_idx+1} Training Completed. Best Val Loss: {early_stopper.val_loss_min:.4f}")

# --- Evaluation ---
# ... (Keep Evaluation section as is) ...
print("\n--- Evaluating Ensemble ---")
all_model_predictions = []
y_true_for_report = None

for model_idx in range(modelsUsed):
    model = models[model_idx] # Model should already be on the device from training
    try:
        model.load_state_dict(torch.load(f'model_{model_idx}_checkpoint.pt', map_location=device))
    except FileNotFoundError:
        print(f"Warning: Checkpoint file model_{model_idx}_checkpoint.pt not found. Skipping model {model_idx+1}.")
        all_model_predictions.append([])
        continue
    except Exception as e:
        print(f"Error loading checkpoint for model {model_idx+1}: {e}. Skipping this model.")
        all_model_predictions.append([])
        continue

    model.eval()
    preds_for_this_model = []
    targets_for_this_model = [] # Will be the same for all, but good to collect per model pass

    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            batch_X_test, batch_y_test = batch_X_test.to(device), batch_y_test.to(device) # <<<< MOVE BATCH TO DEVICE
            y_pred_logits = model(batch_X_test)
            _, predicted_indices = torch.max(y_pred_logits, 1)
            preds_for_this_model.extend(predicted_indices.cpu().numpy())
            targets_for_this_model.extend(batch_y_test.cpu().numpy())

    all_model_predictions.append(preds_for_this_model)
    if y_true_for_report is None and targets_for_this_model: # Ensure targets_for_this_model is not empty
        y_true_for_report = np.array(targets_for_this_model)

valid_model_predictions = [p for p in all_model_predictions if p] # Only keep non-empty prediction lists
predicted_np_eval = None

if not valid_model_predictions:
    print("Error: No valid model predictions available for ensemble or single model evaluation.")
elif len(valid_model_predictions) > 1 : # modelsUsed > 1 and at least two models gave predictions
    print("\n--- Ensemble Performance (Majority Vote) ---")
    from scipy import stats
    stacked_predictions_np = np.array(valid_model_predictions)
    majority_vote_predictions, _ = stats.mode(stacked_predictions_np, axis=0, keepdims=False)
    predicted_np_eval = majority_vote_predictions

    if y_true_for_report is not None:
        print("Ensemble Accuracy:", accuracy_score(y_true_for_report, majority_vote_predictions))
        target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
        print("\nEnsemble Classification Report:")
        print(classification_report(y_true_for_report, majority_vote_predictions, target_names=target_names_report, zero_division=0))
    else:
        print("Error: True labels for reporting not available.")

elif len(valid_model_predictions) == 1: # Only one model successfully provided predictions
    model_type_str = model_configs[0]['model_type'] if model_configs and model_idx < len(model_configs) else 'N/A' # model_idx here might be out of scope, better to use 0
    print(f"\n--- Single Model Performance ({model_configs[0]['model_type'] if model_configs else 'N/A'}) ---")
    predicted_np_eval = np.array(valid_model_predictions[0])
    if y_true_for_report is not None:
        print("Single Model Accuracy:", accuracy_score(y_true_for_report, predicted_np_eval))
        target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
        print("\nSingle Model Classification Report:")
        print(classification_report(y_true_for_report, predicted_np_eval, target_names=target_names_report, zero_division=0))
    else:
        print("Error: True labels for reporting not available.")


if predicted_np_eval is not None and y_true_for_report is not None:
    y_test_np_eval = y_true_for_report
    mismatched_indices = np.where(predicted_np_eval != y_test_np_eval)[0]
    misclassified_data = []

    if len(mismatched_indices) > 0:
        print(f"\nFound {len(mismatched_indices)} misclassified samples.")
        for original_test_idx in mismatched_indices: # Corrected variable name
            if original_test_idx < len(testDF_orig): # Ensure index is within bounds of original test dataframe
                 virus_id = testDF_orig.iloc[original_test_idx, 0] # Assumes ID is the first column
            else:
                virus_id = f"Unknown ID (index {original_test_idx} out of bounds for original test data)"

            predicted_label_num = predicted_np_eval[original_test_idx]
            actual_label_num = y_test_np_eval[original_test_idx]

            predicted_variant_name = float_to_variant.get(float(predicted_label_num), f"Unknown {predicted_label_num}")
            actual_variant_name = float_to_variant.get(float(actual_label_num), f"Unknown {actual_label_num}")

            misclassified_data.append({
                'Virus ID': virus_id,
                'Predicted Variant': predicted_variant_name,
                'Actual Variant': actual_variant_name
            })

        misclassified_df = pd.DataFrame(misclassified_data)
        print("\nTable of Misclassified Samples:")
        print(misclassified_df.to_string())
    else:
        print("\nNo misclassified samples in the test set with the final ensemble/model! Great job!")
else:
    if predicted_np_eval is None:
        print("\nSkipping misclassification analysis: Final predictions (predicted_np_eval) not generated.")
    if y_true_for_report is None:
        print("\nSkipping misclassification analysis: True labels for reporting (y_true_for_report) not available.")