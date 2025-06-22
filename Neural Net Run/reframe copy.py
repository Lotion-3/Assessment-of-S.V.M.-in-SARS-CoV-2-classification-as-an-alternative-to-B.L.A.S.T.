import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PhysicsInformedModel(nn.Module):
    def __init__(self, in_features=4, h1=512, h2=256, h3=128, h4=64, h5=32, h6=16, h7=8, h8=4, out_features=1):  # Changed in_features to 4
        super().__init__()
        # Main network architecture (removed redundant physics layer)
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        #self.fc4 = nn.Linear(h3, h4)
        #self.bn4 = nn.BatchNorm1d(h4)
        #self.fc5 = nn.Linear(h4, h5)
        #self.bn5 = nn.BatchNorm1d(h5)
        #self.fc6 = nn.Linear(h5, h6)
        #self.bn6 = nn.BatchNorm1d(h6)
        #self.fc7 = nn.Linear(h6, h7)
        #self.bn7 = nn.BatchNorm1d(h7)
        #self.fc8 = nn.Linear(h7, h8)
        #self.bn8 = nn.BatchNorm1d(h8)
        self.out = nn.Linear(h3, out_features)
        self.dropout = nn.Dropout(0.6)
        
        # Initialize weights properly
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.fc7.weight, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.fc8.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        # Directly process the pre-engineered features
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        #x = F.leaky_relu(self.bn4(self.fc4(x)))
        #x = self.dropout(x)
        #x = F.leaky_relu(self.bn5(self.fc5(x)))
        #x = self.dropout(x)
        #x = F.leaky_relu(self.bn6(self.fc6(x)))
        #x = self.dropout(x)
        #x = F.leaky_relu(self.bn7(self.fc7(x)))
        #x = self.dropout(x)
        #x = F.leaky_relu(self.bn8(self.fc8(x)))
        return self.out(x)

# Load data
trainDF = pd.read_csv("bungeeVerifTraining.csv")
testDF = pd.read_csv("bungeeVerifTesting.csv")

# Feature engineering
def add_physics_features(df):
    df = df.copy()
    df['sqrt_mh'] = np.sqrt(df['mass'] * df['dropHeight'])
    df['m/h_ratio'] = ((df['mass'] / df['dropHeight']))
    return df

# Data processing
X_train = add_physics_features(trainDF.iloc[:, 0:-1])
X_test = add_physics_features(testDF.iloc[:, 0:-1])
y_train = trainDF.iloc[:, -1]
y_test = testDF.iloc[:, -1]

# Verify feature count
print(f"Number of features: {X_train.shape[1]}")  # Should be 4

# Preserve original features for output
X_train_orig = trainDF.iloc[:, 0:-1].copy()
X_test_orig = testDF.iloc[:, 0:-1].copy()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))

# Create validation split
#X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
#    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42
#)

# Initialize model
model = PhysicsInformedModel(in_features=4)  # 2 original + 2 engineered features

# Loss and optimizer
criterion = nn.HuberLoss()  # More robust to outliers
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)

# Training parameters
epochs = 10000
batch_size = 12
best_loss = float('inf')
patience = 100
no_improvement = 0

# Training loop with early stopping
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Mini-batch training
    permutation = torch.randperm(X_train_tensor.size()[0])
    for i in range(0, X_train_tensor.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor)
    
    scheduler.step(val_loss)
    
    # Early stopping check
    
    if val_loss < best_loss + (best_loss*0.00001):
        best_loss = val_loss
        no_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}: Train Loss: {epoch_loss/len(X_train_tensor):.4f}, Val Loss: {val_loss:.4f}')

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    test_loss = criterion(y_pred, y_test_tensor)
    
    # Convert to numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test_tensor.numpy()
    
    # Metrics
    mse = mean_squared_error(y_test_np, y_pred_np)
    mae = mean_absolute_error(y_test_np, y_pred_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_np, y_pred_np)
    
    print(f'\nTest Loss: {test_loss.item():.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R² Score: {r2:.4f}')

    # Save results
    results_df = X_test_orig.copy()
    results_df['Prediction'] = y_pred_np
    results_df.to_csv("improved_results.csv", index=False)
    print("\nSample predictions:")
    print(results_df.head())