import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import random # For randomizing hyperparameters

# --- Set Pandas option to opt-in to future behavior for downcasting ---
# This should address the FutureWarning from the .replace() method.
pd.set_option('future.no_silent_downcasting', True)


# --- Your Model2 class (Unchanged) ---
class Model2(nn.Module):
    def __init__(self, in_features, h1, h2, h3, h4,
                 drop1_rate, drop2_rate, drop3_rate, drop4_rate,
                 out_features=11): # out_features is fixed
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

# --- Data Loading and Preprocessing ---
try:
    trainDF_orig = pd.read_csv("train10(11000)(1)(extractedALT9).csv")
    testDF_orig = pd.read_csv("test90(11000)(1)(extractedALT9).csv")
except FileNotFoundError:
    print("Error: One or both data files are not found. Please check file paths.")
    print("Attempted to load: train10(11000)(1)(extractedALT3).csv and test90(11000)(1)(extractedALT2).csv")
    exit()


if trainDF_orig.empty:
    print("Error: train10(11000)(1)(extractedALT2).csv is empty. Exiting.")
    exit()
if testDF_orig.empty:
    print("Error: test90(11000)(1)(extractedALT2).csv is empty. Exiting.")
    exit()


trainDF = trainDF_orig.copy()
testDF = testDF_orig.copy()

variant_to_float = {
    'Alpha': 0.0, 'Beta': 1.0, 'Gamma': 2.0, 'Delta': 3.0, 'Epsilon': 4.0,
    'Zeta': 5.0, 'Eta': 6.0, 'Iota': 7.0, 'Lambda': 8.0, 'Mu': 9.0, 'Omicron': 10.0
}
float_to_variant = {v: k for k, v in variant_to_float.items()}


trainDF['Variant'] = trainDF['Variant'].replace(variant_to_float)
testDF['Variant'] = testDF['Variant'].replace(variant_to_float)

# After replacement, ensure the 'Variant' column is of a numeric type (e.g., float)
# for downstream processing (like stratify in train_test_split).
# The pd.to_numeric call will convert to float if possible, or raise an error if not.
try:
    trainDF['Variant'] = pd.to_numeric(trainDF['Variant'], errors='raise')
    testDF['Variant'] = pd.to_numeric(testDF['Variant'], errors='raise')
except ValueError as e:
    print(f"Error converting 'Variant' column to numeric after replacement: {e}")
    print("This might happen if some variants were not mapped in variant_to_float and remained as strings.")
    print("Train 'Variant' unique values before error:", trainDF['Variant'].unique())
    print("Test 'Variant' unique values before error:", testDF['Variant'].unique())
    exit()


if trainDF['Variant'].isnull().any() or testDF['Variant'].isnull().any():
    print("Warning: NaNs found in 'Variant' column after replacement and to_numeric conversion.")
    print("Train NaNs:", trainDF['Variant'].isnull().sum())
    trainDF.dropna(subset=['Variant'], inplace=True)
    print("Dropped NaNs from trainDF. New shape:", trainDF.shape)
    print("Test NaNs:", testDF['Variant'].isnull().sum())
    testDF.dropna(subset=['Variant'], inplace=True)
    print("Dropped NaNs from testDF. New shape:", testDF.shape)
    if trainDF.empty or testDF.empty:
        print("Error: DataFrames became empty after dropping NaNs. Check variant mapping and input data. Exiting.")
        exit()


X_train_df_full = trainDF.iloc[:, 1:-1]
y_train_df_full = trainDF.iloc[:, -1]

X_test_df = testDF.iloc[:, 1:-1]
y_test_df = testDF.iloc[:, -1]


if X_train_df_full.empty:
    print("Error: X_train_df_full (feature set for training) is empty. Check input CSV structure and iloc slicing.")
    exit()
if X_test_df.empty:
    print("Error: X_test_df (feature set for testing) is empty. Check input CSV structure and iloc slicing.")
    exit()


in_features = X_train_df_full.shape[1]
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
y_train_tensor = torch.LongTensor(y_train_np) # CrossEntropyLoss expects LongTensor for targets
X_val_tensor = torch.FloatTensor(X_val_np)
y_val_tensor = torch.LongTensor(y_val_np)
X_test_tensor = torch.FloatTensor(X_test_np)
y_test_tensor = torch.LongTensor(y_test_np)

batch_size = 32
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


models = []
optimizers = []
schedulers = []
model_configs = []
train_loaders_ensemble = []

modelsUsed = 1
# random.seed(42)
# torch.manual_seed(42)

h1_range = (1024, 2048)
h2_range = (512, 1024)
h3_range = (256, 512)
h4_range = (128,256)
dropout_rate_range = (0.1, 0.4)
lr_range = (0.0001, 0.001)
weight_decay_range = (0.0001, 0.005)
l1_lambda_range = (0.0, 0.0005)

print(f"\n--- Configuring {modelsUsed} Ensemble Models ---")
for i in range(modelsUsed):
    h1_rand = random.randint(h1_range[0], h1_range[1])

    h2_max_allowable = min(h2_range[1], h1_rand - 1)
    h2_rand = random.randint(h2_range[0], max(h2_range[0], h2_max_allowable))
    
    h3_max_allowable = min(h3_range[1], h2_rand - 1)
    h3_rand = random.randint(h3_range[0], max(h3_range[0], h3_max_allowable))
    
    h4_max_allowable = min(h4_range[1], h3_rand - 1)
    h4_rand = random.randint(h4_range[0], max(h4_range[0], h4_max_allowable))

    h1_rand = max(10, h1_rand)
    h2_rand = max(10, h2_rand)
    h3_rand = max(10, h3_rand)
    h4_rand = max(10, h4_rand)

    drop1_rand = round(random.uniform(dropout_rate_range[0], dropout_rate_range[1]), 2)
    drop2_rand = round(random.uniform(dropout_rate_range[0], dropout_rate_range[1]), 2)
    drop3_rand = round(random.uniform(dropout_rate_range[0], max(0.0, dropout_rate_range[1] - 0.05) ), 2)
    drop4_rand = round(random.uniform(dropout_rate_range[0], max(0.0, dropout_rate_range[1] - 0.05) ), 2)

    current_lr = round(random.uniform(lr_range[0], lr_range[1]), 5)
    current_wd = round(random.uniform(weight_decay_range[0], weight_decay_range[1]), 5)
    current_l1_lambda = round(random.uniform(l1_lambda_range[0], l1_lambda_range[1]), 6)

    config = {
        'h1': h1_rand, 'h2': h2_rand, 'h3': h3_rand, 'h4': h4_rand,
        'drop1_rate': drop1_rand, 'drop2_rate': drop2_rand, 'drop3_rate': drop3_rand, 'drop4_rate': drop4_rand,
        'lr': current_lr,
        'weight_decay': current_wd,
        'l1_lambda': current_l1_lambda
    }
    model_configs.append(config)

    model = Model2(in_features=in_features,
                   h1=config['h1'], h2=config['h2'], h3=config['h3'], h4=config['h4'],
                   drop1_rate=config['drop1_rate'],
                   drop2_rate=config['drop2_rate'],
                   drop3_rate=config['drop3_rate'],
                   drop4_rate=config['drop4_rate'],
                   out_features=out_features)
    models.append(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    optimizers.append(optimizer)

    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=15, min_lr=1e-7
    ))

    n_samples_train = X_train_tensor.shape[0]
    if n_samples_train == 0:
        print("Error: No training samples available for bootstrap after potential filtering. Exiting.")
        exit()
    bootstrap_indices = torch.randint(0, n_samples_train, (n_samples_train,))
    X_train_bootstrap = X_train_tensor[bootstrap_indices]
    y_train_bootstrap = y_train_tensor[bootstrap_indices]

    train_dataset_bootstrap = TensorDataset(X_train_bootstrap, y_train_bootstrap)
    train_loader_bootstrap = DataLoader(train_dataset_bootstrap, batch_size=batch_size, shuffle=True)
    train_loaders_ensemble.append(train_loader_bootstrap)

criterion = nn.CrossEntropyLoss()

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
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
            if self.verbose and self.counter % 10 == 0 :
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

epochs = 1000

print("\n--- Training Ensemble Models ---")
for model_idx in range(modelsUsed):
    print(f"\n--- Training Model {model_idx+1}/{modelsUsed} ---")
    print(f"Config: h1={model_configs[model_idx]['h1']}, h2={model_configs[model_idx]['h2']}, h3={model_configs[model_idx]['h3']}, h4={model_configs[model_idx]['h4']}, "
          f"d1={model_configs[model_idx]['drop1_rate']:.2f}, d2={model_configs[model_idx]['drop2_rate']:.2f}, d3={model_configs[model_idx]['drop3_rate']:.2f}, d4={model_configs[model_idx]['drop4_rate']:.2f}, "
          f"lr={model_configs[model_idx]['lr']:.5f}, wd={model_configs[model_idx]['weight_decay']:.5f}, l1={model_configs[model_idx]['l1_lambda']:.6f}")

    model = models[model_idx]
    optimizer = optimizers[model_idx]
    scheduler = schedulers[model_idx]
    current_train_loader = train_loaders_ensemble[model_idx]
    current_l1_lambda_val = model_configs[model_idx]['l1_lambda']
    
    early_stopper = EarlyStopping(patience=30, verbose=False, path=f'model_{model_idx}_checkpoint.pt')

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in current_train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            if current_l1_lambda_val > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + current_l1_lambda_val * l1_norm

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(current_train_loader) if len(current_train_loader) > 0 else 0


        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                y_pred_val = model(batch_X_val)
                val_loss_item = criterion(y_pred_val, batch_y_val)
                epoch_val_loss += val_loss_item.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader) if len(val_loader) > 0 else 0

        
        scheduler.step(avg_val_loss)

        if epoch % 50 == 0:
            print(f'M{model_idx+1} Ep:{epoch:03d} | TrL:{avg_train_loss:.4f} | VaL:{avg_val_loss:.4f} | LR:{optimizer.param_groups[0]["lr"]:.2e}')

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f"Model {model_idx+1} early stopping at epoch {epoch}.")
            model.load_state_dict(torch.load(early_stopper.path))
            break
    
    if not early_stopper.early_stop: # Ensure best model is loaded if not early stopped
        try:
            model.load_state_dict(torch.load(early_stopper.path))
        except FileNotFoundError:
            print(f"Warning: Checkpoint file {early_stopper.path} not found for model {model_idx+1}. Model may not be at its best state.")
            # This can happen if validation loss never improved from np.inf and no checkpoint was saved.

    print(f"Model {model_idx+1} Training Completed. Best Val Loss: {early_stopper.val_loss_min:.4f}")

print("\n--- Evaluating Ensemble ---")
all_model_predictions = []
y_true_for_report = None

for model_idx in range(modelsUsed):
    model = models[model_idx]
    try:
        model.load_state_dict(torch.load(f'model_{model_idx}_checkpoint.pt'))
    except FileNotFoundError:
        print(f"Warning: Checkpoint file model_{model_idx}_checkpoint.pt not found for evaluation of model {model_idx+1}. Skipping this model in ensemble.")
        all_model_predictions.append([]) # Add empty list to maintain structure if needed, or skip appending
        continue # Skip to next model
    except Exception as e:
        print(f"Error loading checkpoint for model {model_idx+1}: {e}. Skipping this model.")
        all_model_predictions.append([])
        continue

    model.eval()
    
    preds_for_this_model = []
    targets_for_this_model = [] # Will be the same for all, but good to collect per model pass

    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            y_pred_logits = model(batch_X_test)
            _, predicted_indices = torch.max(y_pred_logits, 1)
            preds_for_this_model.extend(predicted_indices.cpu().numpy())
            targets_for_this_model.extend(batch_y_test.cpu().numpy())
            
    all_model_predictions.append(preds_for_this_model)
    if y_true_for_report is None and targets_for_this_model: # Ensure targets_for_this_model is not empty
        y_true_for_report = np.array(targets_for_this_model)

# Filter out any models that failed to load (resulted in empty predictions)
valid_model_predictions = [p for p in all_model_predictions if p] # Only keep non-empty prediction lists

predicted_np_eval = None
if not valid_model_predictions:
    print("Error: No valid model predictions available for ensemble or single model evaluation.")
elif len(valid_model_predictions) > 1 : # modelsUsed > 1 and at least two models gave predictions
    print("\n--- Ensemble Performance (Majority Vote) ---")
    stacked_predictions_np = np.array(valid_model_predictions)
    from scipy import stats
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
    print("\n--- Single Model Performance (as only one model's predictions are valid) ---")
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
        for idx in mismatched_indices:
            if idx < len(testDF_orig): # Ensure index is within bounds of original test dataframe
                 virus_id = testDF_orig.iloc[idx, 0] # Assumes ID is the first column
            else:
                virus_id = f"Unknown ID (index {idx} out of bounds for original test data)"

            predicted_label_num = predicted_np_eval[idx]
            actual_label_num = y_test_np_eval[idx]

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