import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score

# --- Your Model2 class ---
class Model2(nn.Module):
    # Consider a slightly smaller default if training time is still an issue
    # def __init__(self, in_features=116, h1=128, h2=64, out_features=11): # Example smaller
    def __init__(self, in_features=121, h1=512, h2=256, h3=128, out_features=11): # Your current good one
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(0.2) # Slightly increased dropout as example
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.drop3 = nn.Dropout(0.1) # Dropout after last hidden
        self.out = nn.Linear(h3, out_features)
        # self.leaky_relu = nn.LeakyReLU(0.01) # Not needed if using F.leaky_relu

    def forward(self, x):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), 0.01))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), 0.01))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x)), 0.01))
        x = self.out(x)
        return x

# --- Data Loading and Preprocessing (Your existing code is good) ---
# ... (keep your variant mapping, df loading, scaling) ...
# ... (make sure X_train_df, y_train_df etc. are defined before split)

trainDF_orig = pd.read_csv("train10(11000)(1)(extractedALT3).csv")
testDF_orig = pd.read_csv("test90(11000)(1)(extractedALT3).csv")

trainDF = trainDF_orig.copy()
testDF = testDF_orig.copy()

variant_to_float = {
    'Alpha': 0.0, 'Beta': 1.0, 'Gamma': 2.0, 'Delta': 3.0, 'Epsilon': 4.0,
    'Zeta': 5.0, 'Eta': 6.0, 'Iota': 7.0, 'Lambda': 8.0, 'Mu': 9.0, 'Omicron': 10.0
}
float_to_variant = {v: k for k, v in variant_to_float.items()} # For later use

trainDF['Variant'] = trainDF['Variant'].replace(variant_to_float)
testDF['Variant'] = testDF['Variant'].replace(variant_to_float)

X_train_df = trainDF.iloc[:, 1:-1]
y_train_df = trainDF.iloc[:, -1:]
X_test_df = testDF.iloc[:, 1:-1]
y_test_df = testDF.iloc[:, -1:]
# --- Create Validation Split ---
X_train_df_full = trainDF.iloc[:, 1:-1]
y_train_df_full = trainDF.iloc[:, -1:]['Variant'] # Make sure y_train is a Series for stratify

# Scale features
scaler = StandardScaler()
X_train_scaled_full = scaler.fit_transform(X_train_df_full)
X_test_scaled = scaler.transform(testDF.iloc[:, 1:-1]) # Use same scaler

# Split training data into train and validation
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_train_scaled_full, y_train_df_full.values,
    test_size=0.15, random_state=42, stratify=y_train_df_full.values
)

X_test_np = X_test_scaled
y_test_np = testDF.iloc[:, -1:]['Variant'].values


X_train_tensor = torch.FloatTensor(X_train_np)
y_train_tensor = torch.LongTensor(y_train_np)
X_val_tensor = torch.FloatTensor(X_val_np)
y_val_tensor = torch.LongTensor(y_val_np)
X_test_tensor = torch.FloatTensor(X_test_np)
y_test_tensor = torch.LongTensor(y_test_np)

# --- DataLoader ---
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- Model, Optimizer, Criterion, Scheduler ---
out_features = 11
models = [] # Store models in a list
optimizers = []
schedulers = []

modelsUsed = 512 # Start with 1, fix ensemble later if needed. For ensemble, set to e.g. 3 or 5
# If using ensemble, ensure models have different initial weights (default)
# or slightly different architectures / training variations.

for i in range(modelsUsed):
    model = Model2(out_features=out_features)
    models.append(model)
    optimizers.append(torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001))
    schedulers.append(torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizers[i], mode='min', factor=0.2, patience=15, verbose=True, min_lr=1e-7 # Increased patience
    ))

criterion = nn.CrossEntropyLoss()

# --- Early Stopping Class ---
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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- Training Loop ---
epochs = 1000 # Max epochs, early stopping will likely trigger sooner
l1_lambda = 0.0001 # Reduced L1, or rely on AdamW weight_decay primarily

print("\n--- Training ---")
for model_idx in range(modelsUsed):
    print(f"\n--- Training Model {model_idx+1} ---")
    model = models[model_idx]
    optimizer = optimizers[model_idx]
    scheduler = schedulers[model_idx]
    early_stopper = EarlyStopping(patience=30, verbose=True, path=f'model_{model_idx}_checkpoint.pt') # Longer patience for scheduler + early stopping

    model_losses = [] # Track losses for this model

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # Optional: L1 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        model_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                y_pred_val = model(batch_X_val)
                val_loss = criterion(y_pred_val, batch_y_val)
                epoch_val_loss += val_loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss) # Step scheduler on validation loss

        if epoch % 50 == 0: # Print less frequently
            print(f'Model {model_idx+1} Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]["lr"]:.7f}')

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load(early_stopper.path)) # Load best model
            break
    
    # Load best model state for this model after training
    if not early_stopper.early_stop: # if it completed all epochs
        model.load_state_dict(torch.load(early_stopper.path)) # ensure best model is loaded

    print(f"\nModel {model_idx+1} Training Completed. Best Val Loss: {early_stopper.val_loss_min:.4f}")


# --- Evaluation ---
print("\n--- Evaluation ---")
all_model_predictions = []

for model_idx in range(modelsUsed):
    model = models[model_idx]
    model.load_state_dict(torch.load(f'model_{model_idx}_checkpoint.pt')) # Ensure best model loaded
    model.eval()
    
    all_preds_for_model = []
    all_targets_for_model = [] # Should be same for all models if y_test_tensor is same

    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            y_pred_logits = model(batch_X_test)
            _, predicted_indices = torch.max(y_pred_logits, 1)
            all_preds_for_model.extend(predicted_indices.cpu().numpy())
            all_targets_for_model.extend(batch_y_test.cpu().numpy())
            
    # Store all predictions for this model (flattened list)
    all_model_predictions.append(all_preds_for_model)

    print(f"\n--- Model {model_idx+1} Individual Performance ---")
    target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
    print("Accuracy:", accuracy_score(all_targets_for_model, all_preds_for_model))
    print(classification_report(all_targets_for_model, all_preds_for_model, target_names=target_names_report, zero_division=0))
    # Your misclassified samples analysis can go here for individual models

# --- Ensemble (if modelsUsed > 1) ---
if modelsUsed > 1:
    print("\n--- Ensemble Performance ---")
    # Stack predictions: rows are models, columns are samples
    stacked_predictions_np = np.array(all_model_predictions)
    
    # Majority vote: scipy.stats.mode
    from scipy import stats
    majority_vote_predictions, _ = stats.mode(stacked_predictions_np, axis=0, keepdims=False) # keepdims=False for newer scipy

    print("Ensemble Accuracy:", accuracy_score(all_targets_for_model, majority_vote_predictions)) # all_targets_for_model should be y_test_np
    print("\nEnsemble Classification Report:")
    print(classification_report(all_targets_for_model, majority_vote_predictions, target_names=target_names_report, zero_division=0))
    # Your misclassified samples analysis for ensemble

# --- For a single model, the ensemble part is skipped, or you just use the results from model_idx=0 ---
elif modelsUsed == 1:
    print("\n--- Single Model Performance (same as Model 1 above) ---")
    # The individual performance of Model 1 is your final result.
    # You can re-print or just refer to the Model 1 section.

# ... (Your misclassified samples analysis for the final chosen predictions (single or ensemble)) ...
# Ensure to use the correct predicted_np_eval and y_test_np_eval
# For ensemble, predicted_np_eval would be majority_vote_predictions
# For single model, it would be all_model_predictions[0]

y_test_np_eval = y_test_tensor.cpu().numpy()
predicted_np_eval = majority_vote_predictions
#print("\n--- Misclassified Samples Analysis ---")

    # Find indices where prediction is not equal to actual
mismatched_indices = np.where(predicted_np_eval != y_test_np_eval)[0]
misclassified_data = []
if len(mismatched_indices) > 0:
    for idx in mismatched_indices:
        # Get the original Virus ID from the original test dataframe (testDF_orig)
        # testDF_orig.iloc[idx, 0] assumes 'Virus ID' is the first column
        # and row order is preserved.
        virus_id = testDF_orig.iloc[idx, 0]
        predicted_label_num = predicted_np_eval[idx]
        actual_label_num = y_test_np_eval[idx]

        # Convert numerical labels back to variant names using the float_to_variant map
        # Ensure keys are float for float_to_variant as defined earlier
        predicted_variant_name = float_to_variant.get(float(predicted_label_num), f"Unknown Label {predicted_label_num}")
        actual_variant_name = float_to_variant.get(float(actual_label_num), f"Unknown Label {actual_label_num}")

        misclassified_data.append({
            'Virus ID': virus_id,
            'Predicted Variant': predicted_variant_name,
            'Actual Variant': actual_variant_name
        })

    misclassified_df = pd.DataFrame(misclassified_data)
    print("\nTable of Misclassified Samples:")
    print(misclassified_df.to_string()) # .to_string() to ensure full display if many rows
else:
    print("\nNo misclassified samples in the test set! Great job!")