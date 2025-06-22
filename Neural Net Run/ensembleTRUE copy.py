import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
import random
import optuna # For hyperparameter optimization
from collections import Counter

# --- Set Pandas option to opt-in to future behavior for downcasting ---
pd.set_option('future.no_silent_downcasting', True)

# --- Set random seeds for reproducibility (of the final ensemble training part) ---
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
seed_everything(SEED)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Your Model2 class (Unchanged) ---
class Model2(nn.Module):
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

# --- Data Loading and Preprocessing ---
try:
    trainDF_orig = pd.read_csv("train10(11000)(1)(extractedALT6).csv")
    testDF_orig = pd.read_csv("test90(11000)(1)(extractedALT6).csv")
except FileNotFoundError:
    print("Error: One or both data files are not found. Please check file paths.")
    print("Attempted to load: train10(11000)(1)(extractedALT6).csv and test90(11000)(1)(extractedALT6).csv")
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

trainDF['Variant'] = trainDF['Variant'].replace(variant_to_float)
testDF['Variant'] = testDF['Variant'].replace(variant_to_float)

try:
    trainDF['Variant'] = pd.to_numeric(trainDF['Variant'], errors='raise')
    testDF['Variant'] = pd.to_numeric(testDF['Variant'], errors='raise')
except ValueError as e:
    print(f"Error converting 'Variant' column to numeric: {e}")
    exit()

trainDF.dropna(subset=['Variant'], inplace=True)
testDF.dropna(subset=['Variant'], inplace=True)
if trainDF.empty or testDF.empty:
    print("Error: DataFrames became empty after dropping NaNs. Exiting.")
    exit()

X_df_full = trainDF.iloc[:, 1:-1]
y_df_full = trainDF.iloc[:, -1]
X_test_df = testDF.iloc[:, 1:-1]
y_test_df = testDF.iloc[:, -1]

if X_df_full.empty or X_test_df.empty:
    print("Error: Feature sets are empty. Check CSV structure and slicing.")
    exit()

in_features = X_df_full.shape[1]
out_features = len(variant_to_float)

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_df_full)
X_test_scaled = scaler.transform(X_test_df)

# We will split into train/val for HPO inside the Optuna objective function using K-Fold
# For final model training, we can use the full X_scaled_full or split a dedicated validation set
# Let's keep a small final validation set for the ensemble training phase
X_train_final_np, X_val_final_np, y_train_final_np, y_val_final_np = train_test_split(
    X_scaled_full, y_df_full.values,
    test_size=0.1, random_state=SEED, stratify=y_df_full.values # Smaller val set for final model training
)

X_test_np = X_test_scaled
y_test_np = y_test_df.values

# Convert final validation and test sets to tensors
X_val_final_tensor = torch.FloatTensor(X_val_final_np).to(device)
y_val_final_tensor = torch.LongTensor(y_val_final_np).to(device)
X_test_tensor = torch.FloatTensor(X_test_np).to(device)
y_test_tensor = torch.LongTensor(y_test_np).to(device)

BATCH_SIZE = 64 # Can also be a hyperparameter in Optuna

val_final_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor)
val_final_loader = DataLoader(val_final_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
            if self.verbose and self.counter % 5 == 0 :
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

# --- Optuna Objective Function ---
N_EPOCHS_HPO = 150 # Max epochs for each HPO trial fold
N_CV_FOLDS_HPO = 3 # Number of K-folds for HPO
ES_PATIENCE_HPO = 15 # Early stopping patience for HPO

def objective(trial):
    # Suggest Hyperparameters
    h1 = trial.suggest_int('h1', 512, 2048, step=128)
    h2 = trial.suggest_int('h2', 256, h1, step=64) # h2 <= h1
    h3 = trial.suggest_int('h3', 128, h2, step=32) # h3 <= h2
    h4 = trial.suggest_int('h4', 64, h3, step=16)   # h4 <= h3

    drop1 = trial.suggest_float('drop1', 0.1, 0.5, step=0.05)
    drop2 = trial.suggest_float('drop2', 0.1, 0.5, step=0.05)
    drop3 = trial.suggest_float('drop3', 0.05, 0.4, step=0.05)
    drop4 = trial.suggest_float('drop4', 0.05, 0.4, step=0.05)

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    l1_lambda = trial.suggest_float('l1_lambda', 1e-7, 1e-3, log=True)
    
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])
    # scheduler_name = trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'CosineAnnealingLR'])
    scheduler_name = 'ReduceLROnPlateau' # Keep it simple for now, CosineAnnealingLR needs T_max

    skf = StratifiedKFold(n_splits=N_CV_FOLDS_HPO, shuffle=True, random_state=SEED + trial.number)
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled_full, y_df_full.values)):
        X_train_fold_np, X_val_fold_np = X_scaled_full[train_idx], X_scaled_full[val_idx]
        y_train_fold_np, y_val_fold_np = y_df_full.values[train_idx], y_df_full.values[val_idx]

        X_train_fold_tensor = torch.FloatTensor(X_train_fold_np).to(device)
        y_train_fold_tensor = torch.LongTensor(y_train_fold_np).to(device)
        X_val_fold_tensor = torch.FloatTensor(X_val_fold_np).to(device)
        y_val_fold_tensor = torch.LongTensor(y_val_fold_np).to(device)

        train_fold_dataset = TensorDataset(X_train_fold_tensor, y_train_fold_tensor)
        train_fold_loader = DataLoader(train_fold_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_fold_dataset = TensorDataset(X_val_fold_tensor, y_val_fold_tensor)
        val_fold_loader = DataLoader(val_fold_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = Model2(in_features, h1, h2, h3, h4, drop1, drop2, drop3, drop4, out_features).to(device)
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else: # Adam
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # Note: weight_decay in Adam is L2 reg

        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=7, min_lr=1e-7)
        # elif scheduler_name == 'CosineAnnealingLR':
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_HPO, eta_min=1e-7)


        early_stopper_hpo = EarlyStopping(patience=ES_PATIENCE_HPO, verbose=False, path=f'temp_hpo_model_trial{trial.number}_fold{fold}.pt')

        for epoch in range(N_EPOCHS_HPO):
            model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_fold_loader:
                optimizer.zero_grad()
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)
                if l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in model.parameters())
                    loss = loss + l1_lambda * l1_norm
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_fold_loader) if len(train_fold_loader) > 0 else 0

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_fold_loader:
                    y_pred_val = model(batch_X_val)
                    val_loss_item = criterion(y_pred_val, batch_y_val)
                    epoch_val_loss += val_loss_item.item()
            avg_val_loss = epoch_val_loss / len(val_fold_loader) if len(val_fold_loader) > 0 else 0

            if scheduler_name == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            # elif scheduler_name == 'CosineAnnealingLR':
            #     scheduler.step()

            early_stopper_hpo(avg_val_loss, model)
            if early_stopper_hpo.early_stop:
                # print(f"Trial {trial.number} Fold {fold} Early stopping at epoch {epoch}")
                break
        
        fold_val_losses.append(early_stopper_hpo.val_loss_min)
        # Optuna pruning
        trial.report(early_stopper_hpo.val_loss_min, fold) # Report intermediate value for this fold
        if trial.should_prune(): # Check if this trial should be pruned based on intermediate results
            # print(f"Trial {trial.number} pruned at fold {fold}")
            raise optuna.exceptions.TrialPruned()


    avg_kfold_val_loss = np.mean(fold_val_losses)
    return avg_kfold_val_loss


# --- Run Optuna Study ---
N_TRIALS_OPTUNA = 50 # Number of HPO trials to run
# N_TRIALS_OPTUNA = 3 # For quick test
print(f"\n--- Starting Hyperparameter Optimization with Optuna ({N_TRIALS_OPTUNA} trials, {N_CV_FOLDS_HPO}-fold CV) ---")
study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=N_CV_FOLDS_HPO // 2)) # Prune if not promising
study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)

best_hyperparams = study.best_trial.params
print("\nBest Hyperparameters found by Optuna:")
for key, value in best_hyperparams.items():
    print(f"{key}: {value}")

# --- Final Ensemble Model Training with Best Hyperparameters ---
N_ENSEMBLE_MODELS = 3 # Number of models in the final ensemble
EPOCHS_FINAL = 300   # Max epochs for final model training
ES_PATIENCE_FINAL = 30 # Early stopping patience for final models

models_ensemble = []
# model_configs_ensemble = [] # We'll use the same best_hyperparams for all

print(f"\n--- Training {N_ENSEMBLE_MODELS} Ensemble Models with Best Hyperparameters ---")
for model_idx in range(N_ENSEMBLE_MODELS):
    print(f"\n--- Training Final Model {model_idx+1}/{N_ENSEMBLE_MODELS} ---")
    current_seed = SEED + model_idx # Vary seed for initialization and bootstrap
    seed_everything(current_seed)

    model = Model2(in_features=in_features,
                   h1=best_hyperparams['h1'], h2=best_hyperparams['h2'],
                   h3=best_hyperparams['h3'], h4=best_hyperparams['h4'],
                   drop1_rate=best_hyperparams['drop1'],
                   drop2_rate=best_hyperparams['drop2'],
                   drop3_rate=best_hyperparams['drop3'],
                   drop4_rate=best_hyperparams['drop4'],
                   out_features=out_features).to(device)

    if best_hyperparams['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_hyperparams['lr'], weight_decay=best_hyperparams['weight_decay'])
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['lr'], weight_decay=best_hyperparams['weight_decay'])

    # if best_hyperparams['scheduler'] == 'ReduceLROnPlateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, min_lr=1e-7)
    # elif best_hyperparams['scheduler'] == 'CosineAnnealingLR':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_FINAL, eta_min=1e-7)


    # Bootstrap sampling for each ensemble member from the X_train_final_np
    n_samples_train_final = X_train_final_np.shape[0]
    bootstrap_indices = torch.randint(0, n_samples_train_final, (n_samples_train_final,), generator=torch.Generator().manual_seed(current_seed))
    
    X_train_bootstrap_np = X_train_final_np[bootstrap_indices.numpy()]
    y_train_bootstrap_np = y_train_final_np[bootstrap_indices.numpy()]

    X_train_bootstrap_tensor = torch.FloatTensor(X_train_bootstrap_np).to(device)
    y_train_bootstrap_tensor = torch.LongTensor(y_train_bootstrap_np).to(device)

    train_dataset_bootstrap = TensorDataset(X_train_bootstrap_tensor, y_train_bootstrap_tensor)
    train_loader_bootstrap = DataLoader(train_dataset_bootstrap, batch_size=BATCH_SIZE, shuffle=True)

    early_stopper_final = EarlyStopping(patience=ES_PATIENCE_FINAL, verbose=False, path=f'final_model_{model_idx}_checkpoint.pt')

    for epoch in range(EPOCHS_FINAL):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader_bootstrap:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            if best_hyperparams['l1_lambda'] > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + best_hyperparams['l1_lambda'] * l1_norm
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader_bootstrap) if len(train_loader_bootstrap) > 0 else 0

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_final_loader: # Use the dedicated final validation set
                y_pred_val = model(batch_X_val)
                val_loss_item = criterion(y_pred_val, batch_y_val)
                epoch_val_loss += val_loss_item.item()
        avg_val_loss = epoch_val_loss / len(val_final_loader) if len(val_final_loader) > 0 else 0
        
        # if best_hyperparams['scheduler'] == 'ReduceLROnPlateau':
        scheduler.step(avg_val_loss)
        # elif best_hyperparams['scheduler'] == 'CosineAnnealingLR':
        #     scheduler.step()


        if epoch % 20 == 0 or epoch == EPOCHS_FINAL -1 :
            print(f'M{model_idx+1} Ep:{epoch:03d} | TrL:{avg_train_loss:.4f} | VaL:{avg_val_loss:.4f} | LR:{optimizer.param_groups[0]["lr"]:.2e}')

        early_stopper_final(avg_val_loss, model)
        if early_stopper_final.early_stop:
            print(f"Final Model {model_idx+1} early stopping at epoch {epoch}.")
            break
    
    # Load best model state for this ensemble member
    try:
        model.load_state_dict(torch.load(early_stopper_final.path, map_location=device))
        models_ensemble.append(model)
        print(f"Final Model {model_idx+1} Training Completed. Best Val Loss: {early_stopper_final.val_loss_min:.4f}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint file {early_stopper_final.path} not found for final model {model_idx+1}. Model may not be at its best state.")
        # Optionally, decide if you want to append the model anyway or skip it
        # models_ensemble.append(model) # Appending current state if checkpoint not found

# --- Evaluating Ensemble ---
print("\n--- Evaluating Ensemble on Test Set ---")
all_model_predictions_logits = [] # Store logits for potential soft voting or calibration

if not models_ensemble:
    print("No models were successfully trained for the ensemble. Exiting evaluation.")
    exit()

for model_idx, model in enumerate(models_ensemble):
    model.eval()
    model_preds_logits = []
    with torch.no_grad():
        for batch_X_test, _ in test_loader: # We only need X for predictions
            batch_X_test = batch_X_test.to(device)
            y_pred_logits = model(batch_X_test)
            model_preds_logits.append(y_pred_logits.cpu())
    all_model_predictions_logits.append(torch.cat(model_preds_logits))

if not all_model_predictions_logits:
    print("No predictions generated by ensemble models. Exiting.")
    exit()

# Averaging Logits (Soft Voting equivalent for classification)
# For N models, M samples, C classes: stack has shape (N, M, C)
stacked_logits = torch.stack(all_model_predictions_logits)
mean_logits = torch.mean(stacked_logits, dim=0)
_, ensemble_predictions_indices = torch.max(mean_logits, 1)
predicted_np_eval = ensemble_predictions_indices.numpy()

# Majority Vote (Hard Voting) - Alternative
# all_model_predictions_indices = []
# for logits_tensor in all_model_predictions_logits:
#    _, indices = torch.max(logits_tensor, 1)
#    all_model_predictions_indices.append(indices.numpy())
#
# if all_model_predictions_indices:
#    stacked_indices_np = np.array(all_model_predictions_indices) # Shape (N_models, N_samples)
#    from scipy import stats
#    majority_vote_predictions, _ = stats.mode(stacked_indices_np, axis=0, keepdims=False)
#    # predicted_np_eval = majority_vote_predictions # Uncomment to use majority vote
# else:
#    print("No individual model predictions for majority vote.")
#    exit()


y_true_for_report = y_test_np # Ground truth

if y_true_for_report is not None and predicted_np_eval is not None:
    print("\n--- Ensemble Performance (Soft Voting from Averaged Logits) ---")
    print("Ensemble Accuracy:", accuracy_score(y_true_for_report, predicted_np_eval))
    target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
    print("\nEnsemble Classification Report:")
    print(classification_report(y_true_for_report, predicted_np_eval, target_names=target_names_report, zero_division=0))

    mismatched_indices = np.where(predicted_np_eval != y_true_for_report)[0]
    misclassified_data = []

    if len(mismatched_indices) > 0:
        for idx_in_test_set in mismatched_indices:
            # The mismatched_indices are indices *within the test set*.
            # We need to map them back to original testDF_orig if its order was preserved.
            # Assuming testDF_orig was not re-indexed after initial load and X_test_df was derived preserving order.
            
            # Get the original index if testDF was filtered (e.g. by dropna)
            # This is tricky if testDF_orig had NaNs removed, as indices might change.
            # Safest is to use the ID from testDF_orig corresponding to the row in y_test_np/predicted_np_eval
            # For simplicity, if testDF had no NaNs removed from Variant column or we are careful with indexing:
            original_df_idx = testDF.index[idx_in_test_set] # Get original index from testDF
            
            if original_df_idx < len(testDF_orig):
                 virus_id = testDF_orig.iloc[original_df_idx, 0] # Assumes ID is the first column
            else:
                virus_id = f"Unknown ID (index {original_df_idx} out of bounds for original test data)"

            predicted_label_num = predicted_np_eval[idx_in_test_set]
            actual_label_num = y_true_for_report[idx_in_test_set]

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
        print("\nNo misclassified samples in the test set with the final ensemble! Great job!")
else:
    print("\nSkipping misclassification analysis: Predictions or true labels not available.")

print("\n--- Script Finished ---")

# Clean up temporary HPO model files (optional)
import glob
import os
for f_path in glob.glob("temp_hpo_model_trial*_fold*.pt"):
    try:
        os.remove(f_path)
    except OSError as e:
        print(f"Error deleting HPO temp file {f_path}: {e}")