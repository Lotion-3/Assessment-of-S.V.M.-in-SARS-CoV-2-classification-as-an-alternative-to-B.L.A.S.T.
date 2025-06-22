import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Model2(nn.Module):
    # For 2 hidden use 19 -> 14
    # For 1 hidden use 17 or 16
    def __init__(self, in_features=116, h1=512, h2=256, h3=128, h4=64, h5=32, h6=50, h7=50, h8=50, out_features=11):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, h4)
        #self.fc5 = nn.Linear(h4, h5)
        #self.fc6 = nn.Linear(h5, h6)
        #self.fc7 = nn.Linear(h6, h7)
        #self.fc8 = nn.Linear(h7, h8)
        self.out = nn.Linear(h3, out_features)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        #x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        #x = F.tanh(self.fc4(x))
        #x = F.tanh(self.fc5(x))
        #x = F.tanh(self.fc6(x))
        #x = F.tanh(self.fc7(x))
        #x = F.tanh(self.fc8(x))
        x = self.out(x)
        return x

out_features=11
modelsUsed = 2

for i in range(1, modelsUsed):
    globals()[f'model{i}'] = Model2()

trainDF_orig = pd.read_csv("train10(11000)(1)(extractedALT).csv")
testDF_orig = pd.read_csv("test90(11000)(1)(extractedALT).csv")

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
scaler1 = StandardScaler()
columnNames=X_train_df.columns
X_train_scaled = scaler1.fit_transform(X_train_df)
X_test_scaled = scaler1.transform(X_test_df)
X_train = pd.DataFrame(scaler1.fit_transform(trainDF.iloc[:, 1:-1]), columns=columnNames)
X_test = pd.DataFrame(scaler1.fit_transform(testDF.iloc[:, 1:-1]), columns=columnNames)

y_train = y_train_df['Variant']
y_test = y_test_df['Variant']
X_train_np = (X_train.values)
X_test_np = (X_test.values)
y_train_np = (y_train.values)
y_test_np = (y_test.values)

print(X_train)
print(y_train)
X_train_tensor = torch.FloatTensor(X_train_np)
X_test_tensor = torch.FloatTensor(X_test_np)
y_train_tensor = torch.LongTensor(y_train_np) 
y_test_tensor = torch.LongTensor(y_test_np)

criterion = nn.CrossEntropyLoss()
#print(model2.parameters)
print("\nModel Parameters:")

for i in range(1, modelsUsed):
    model = globals()[f'model{i}']
    globals()[f'optimizer{i}'] = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)

# Add a learning rate scheduler
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',         # Minimize the loss
    factor=0.9,         # Reduce LR by half when triggered
    patience=2,        # Number of epochs with no improvement
    verbose=True,       # Print LR changes
    min_lr=1e-8         # Minimum learning rate
)
"""
losses = []
def trainyTime(model, epochs, optimizer, xTrainTensor, yTrainTensor, l1_lambda, ):
    for i in range(epochs):
        model.train()
        yPred = model.forward(xTrainTensor)
        #print(yPred)
        #print(y_train)
        loss = criterion(yPred, yTrainTensor)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm
        losses.append(loss.detach().numpy())
        if i % 1000 == 0:
            print(f'Epoch:  {i} and loss: {loss}')
        #print(f'Epoch: {i} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        optimizer.zero_grad()
        loss.backward()
        ##for name, param in model2.named_parameters():
            ##print(name, param.grad)
        optimizer.step()
        #scheduler.step(loss)
        
        #if loss < 0.179:
        #    break

print("\n--- Training ---")

for i in range(1, modelsUsed):
    model = globals()[f'model{i}']
    optimizer = globals()[f'optimizer{i}']  # Cycles through optimizers 1-20
    trainyTime(model, 100000, optimizer, X_train_tensor, y_train_tensor, 0.0035)
    print("\nModel " + str(i) + " Completed")

print("\n--- Evaluation ---")
from sklearn.metrics import classification_report, accuracy_score
def testyTime(model, xTestTensor, yTestTensor):
    with torch.no_grad():
        model.eval()
        y_pred_logits = model.forward(xTestTensor)
        test_loss = criterion(y_pred_logits, yTestTensor)
        _, predicted_indices = torch.max(y_pred_logits, 1)
        y_test_np_eval = y_test_tensor.cpu().numpy()
        predicted_np_eval = predicted_indices.cpu().numpy()

        #print("Accuracy:", accuracy_score(y_test_np_eval, predicted_np_eval))
        #print("Loss:", criterion(y_pred_logits, y_test_tensor))
        #print("\nClassification Report:")
        print("Accuracy:", accuracy_score(yTestTensor, predicted_indices))
        #print("Loss:", criterion(y_pred_logits, yTestTensor))
        #print("\nClassification Report:")
        #print(classification_report(y_test_tensor, predicted_indices))
        target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
       # print(classification_report(y_test_np_eval, predicted_np_eval, target_names=target_names_report, zero_division=0))
        # --- New code for misclassified samples table ---
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
            #print("\nTable of Misclassified Samples:")
            #print(misclassified_df.to_string()) # .to_string() to ensure full display if many rows
        else:
            print("\nNo misclassified samples in the test set! Great job!")
        return predicted_indices
    

for i in range(1, modelsUsed):
    model = globals()[f'model{i}']
    optimizer = globals()[f'optimizer{i}']  # Cycles through optimizers 1-20
    globals()[f'tensor{i}'] = testyTime(model, X_test_tensor, y_test_tensor)

print("\nEnsemble Attempt")

#stacked_tensors = torch.stack([tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8, tensor9, tensor10, tensor11, tensor12, tensor13, tensor14, tensor15, tensor16, tensor17, tensor18, tensor19, tensor20], dim=0)
stacked_tensors = torch.stack([
    globals()[f'tensor{i}'] 
    for i in range(1, modelsUsed)
], dim=0)
print("Stacked Tensors (shape {}):\n{}".format(stacked_tensors.shape, stacked_tensors))
majority_vote_tensor, _ = torch.mode(stacked_tensors, dim=0)

print("\nMajority Vote Tensor:\n", majority_vote_tensor)

y_test_np_eval = y_test_tensor.cpu().numpy()
predicted_np_eval = majority_vote_tensor.cpu().numpy()

#print("Accuracy:", accuracy_score(y_test_np_eval, predicted_np_eval))
#print("Loss:", criterion(y_pred_logits, y_test_tensor))
#print("\nClassification Report:")
print("Accuracy:", accuracy_score(y_test_tensor, majority_vote_tensor))
#print("Loss:", criterion(majority_vote_tensor, y_test_tensor))
print("\nClassification Report:")
#print(classification_report(y_test_tensor, predicted_indices))
target_names_report = [float_to_variant.get(float(i), f"Class {i}") for i in range(out_features)]
print(classification_report(y_test_np_eval, predicted_np_eval, target_names=target_names_report, zero_division=0))
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