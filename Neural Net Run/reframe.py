import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array
import numpy as np
from sklearn.model_selection import train_test_split

class Model2(nn.Module):
    # For 2 hidden use 19 -> 14
    # For 1 hidden use 17 or 16
    def __init__(self, in_features=25, h1=256, h2=128, h3=64, h4=50, h5=50, h6=50, h7=50, h8=50, out_features=11):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        #self.drop1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        #self.drop2 = nn.Dropout(0.1)
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
        #x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        #x = self.drop2(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        #x = F.tanh(self.fc4(x))
        #x = F.tanh(self.fc5(x))
        #x = F.tanh(self.fc6(x))
        #x = F.tanh(self.fc7(x))
        #x = F.tanh(self.fc8(x))
        x = self.out(x)
        return x

model2 = Model2()
trainDF = pd.read_csv("train90(11000)(1)(extracted).csv")
testDF = pd.read_csv("test10(11000)(1)(extracted).csv")
trainDF['Variant'] = trainDF['Variant'].replace('Alpha', 0.0)
trainDF['Variant'] = trainDF['Variant'].replace('Beta', 1.0)
trainDF['Variant'] = trainDF['Variant'].replace('Gamma', 2.0)
trainDF['Variant'] = trainDF['Variant'].replace('Delta', 3.0)
trainDF['Variant'] = trainDF['Variant'].replace('Epsilon', 4.0)
trainDF['Variant'] = trainDF['Variant'].replace('Zeta', 5.0)
trainDF['Variant'] = trainDF['Variant'].replace('Eta', 6.0)
trainDF['Variant'] = trainDF['Variant'].replace('Iota', 7.0)
trainDF['Variant'] = trainDF['Variant'].replace('Lambda', 8.0)
trainDF['Variant'] = trainDF['Variant'].replace('Mu', 9.0)
trainDF['Variant'] = trainDF['Variant'].replace('Omicron', 10.0)

testDF['Variant'] = testDF['Variant'].replace('Alpha', 0.0)
testDF['Variant'] = testDF['Variant'].replace('Beta', 1.0)
testDF['Variant'] = testDF['Variant'].replace('Gamma', 2.0)
testDF['Variant'] = testDF['Variant'].replace('Delta', 3.0)
testDF['Variant'] = testDF['Variant'].replace('Epsilon', 4.0)
testDF['Variant'] = testDF['Variant'].replace('Zeta', 5.0)
testDF['Variant'] = testDF['Variant'].replace('Eta', 6.0)
testDF['Variant'] = testDF['Variant'].replace('Iota', 7.0)
testDF['Variant'] = testDF['Variant'].replace('Lambda', 8.0)
testDF['Variant'] = testDF['Variant'].replace('Mu', 9.0)
testDF['Variant'] = testDF['Variant'].replace('Omicron', 10.0)

X_train = trainDF.iloc[:, 1:-1]
y_train = trainDF.iloc[:, -1:]
X_test = testDF.iloc[:, 1:-1]
y_test = testDF.iloc[:, -1:]
scaler1 = StandardScaler()
columnNames=X_train.columns
X_train = pd.DataFrame(scaler1.fit_transform(trainDF.iloc[:, 1:-1]), columns=columnNames)
X_test = pd.DataFrame(scaler1.fit_transform(testDF.iloc[:, 1:-1]), columns=columnNames)
"""
X_train["Histidine perc"]=X_train["Histidine perc"]*25
X_train["Proline perc"]=X_train["Proline perc"]*24
X_train["Glutamine perc"]=X_train["Glutamine perc"]*23
X_train["Phenylalanine perc"]=X_train["Phenylalanine perc"]*22
X_train["Threonine perc"]=X_train["Threonine perc"]*21
X_train["Glutamic Acid perc"]=X_train["Glutamic Acid perc"]*20
X_train["Tryptophan perc"]=X_train["Tryptophan perc"]*19
X_train["Arginine perc"]=X_train["Arginine perc"]*18
X_train["Serine perc"]=X_train["Serine perc"]*17
X_train["Valine perc"]=X_train["Valine perc"]*16
X_train["Aspartic Acid perc"]=X_train["Aspartic Acid perc"]*15
X_train["A perc"]=X_train["A perc"]*14
X_train["Cysteine perc"]=X_train["Cysteine perc"]*13
X_train["Asparagine perc"]=X_train["Asparagine perc"]*12
X_train["Glycine perc"]=X_train["Glycine perc"]*11
X_train["T perc"]=X_train["T perc"]*10
X_train["Lysine perc"]=X_train["Lysine perc"]*9
X_train["Isoleucine perc"]=X_train["Isoleucine perc"]*8
X_train["Alanine perc"]=X_train["Alanine perc"]*7
X_train["Tyrosine perc"]=X_train["Tyrosine perc"]*6
X_train["GC perc"]=X_train["GC perc"]*5
X_train["Methionine perc"]=X_train["Methionine perc"]*4
X_train["Leucine perc"]=X_train["Leucine perc"]*3
X_train["C perc"]=X_train["C perc"]*2
X_train["G perc"]=X_train["G perc"]*1

X_test["Histidine perc"]=X_test["Histidine perc"]*25
X_test["Proline perc"]=X_test["Proline perc"]*24
X_test["Glutamine perc"]=X_test["Glutamine perc"]*23
X_test["Phenylalanine perc"]=X_test["Phenylalanine perc"]*22
X_test["Threonine perc"]=X_test["Threonine perc"]*21
X_test["Glutamic Acid perc"]=X_test["Glutamic Acid perc"]*20
X_test["Tryptophan perc"]=X_test["Tryptophan perc"]*19
X_test["Arginine perc"]=X_test["Arginine perc"]*18
X_test["Serine perc"]=X_test["Serine perc"]*17
X_test["Valine perc"]=X_test["Valine perc"]*16
X_test["Aspartic Acid perc"]=X_test["Aspartic Acid perc"]*15
X_test["A perc"]=X_test["A perc"]*14
X_test["Cysteine perc"]=X_test["Cysteine perc"]*13
X_test["Asparagine perc"]=X_test["Asparagine perc"]*12
X_test["Glycine perc"]=X_test["Glycine perc"]*11
X_test["T perc"]=X_test["T perc"]*10
X_test["Lysine perc"]=X_test["Lysine perc"]*9
X_test["Isoleucine perc"]=X_test["Isoleucine perc"]*8
X_test["Alanine perc"]=X_test["Alanine perc"]*7
X_test["Tyrosine perc"]=X_test["Tyrosine perc"]*6
X_test["GC perc"]=X_test["GC perc"]*5
X_test["Methionine perc"]=X_test["Methionine perc"]*4
X_test["Leucine perc"]=X_test["Leucine perc"]*3
X_test["C perc"]=X_test["C perc"]*2
X_test["G perc"]=X_test["G perc"]*1
"""
y_train = y_train['Variant']
y_test = y_test['Variant']
X_train = (X_train.values)
X_test = (X_test.values)
y_train = (y_train.values)
y_test = (y_test.values)

print(X_train)
print(y_train)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train) 
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
print(model2.parameters)
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.00001, weight_decay=0.001)
l1_lambda = 0.0035
##epochs = 1000000
epochs = 100000
losses = []
for i in range(epochs):
    yPred = model2.forward(X_train)
    #print(yPred)
    #print(y_train)
    loss = criterion(yPred, y_train)
    l1_norm = sum(p.abs().sum() for p in model2.parameters())
    loss = loss + l1_lambda * l1_norm
    losses.append(loss.detach().numpy())
    print(f'Epoch:  {i} and loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    ##for name, param in model2.named_parameters():
        ##print(name, param.grad)
    optimizer.step()
    if loss < 0.179:
        break
from sklearn.metrics import classification_report, accuracy_score
with torch.no_grad():
    """
    y_eval = model2.forward(X_test)
    loss = criterion(y_eval, y_test)
    predicted = torch.max(y_eval, 1)
    print(y_test)
    print(y_test.shape)
    print(predicted)
    print(predicted.shape)
    print("Accuracy:", accuracy_score(y_test, predicted))
    print(classification_report(y_test, predicted))
    """
    model2.eval()
    y_pred = model2.forward(X_test)
    _, predicted = torch.max(y_pred, 1)
    
    print("Accuracy:", accuracy_score(y_test, predicted))
    print("Loss:", criterion(y_pred, y_test))
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))
print(y_pred)