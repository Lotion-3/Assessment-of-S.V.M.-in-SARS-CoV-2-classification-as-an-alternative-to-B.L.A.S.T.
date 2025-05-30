import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import array
import numpy as np
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, in_features=25, h1=20, h2=10, out_features=11):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
model = Model()

"""
trainDF = pd.read_csv("train90(11000)(1)(extracted).csv")
testDF = pd.read_csv("test10(11000)(1)(extracted).csv")
trainDF['Variant'] = trainDF['Variant'].replace('Alpha', 1.0)
trainDF['Variant'] = trainDF['Variant'].replace('Beta', 2.0)
trainDF['Variant'] = trainDF['Variant'].replace('Gamma', 3.0)
trainDF['Variant'] = trainDF['Variant'].replace('Delta', 4.0)
trainDF['Variant'] = trainDF['Variant'].replace('Epsilon', 5.0)
trainDF['Variant'] = trainDF['Variant'].replace('Zeta', 6.0)
trainDF['Variant'] = trainDF['Variant'].replace('Eta', 7.0)
trainDF['Variant'] = trainDF['Variant'].replace('Iota', 8.0)
trainDF['Variant'] = trainDF['Variant'].replace('Lambda', 9.0)
trainDF['Variant'] = trainDF['Variant'].replace('Mu', 10.0)
trainDF['Variant'] = trainDF['Variant'].replace('Omicron', 11.0)

testDF['Variant'] = testDF['Variant'].replace('Alpha', 1.0)
testDF['Variant'] = testDF['Variant'].replace('Beta', 2.0)
testDF['Variant'] = testDF['Variant'].replace('Gamma', 3.0)
testDF['Variant'] = testDF['Variant'].replace('Delta', 4.0)
testDF['Variant'] = testDF['Variant'].replace('Epsilon', 5.0)
testDF['Variant'] = testDF['Variant'].replace('Zeta', 6.0)
testDF['Variant'] = testDF['Variant'].replace('Eta', 7.0)
testDF['Variant'] = testDF['Variant'].replace('Iota', 8.0)
testDF['Variant'] = testDF['Variant'].replace('Lambda', 9.0)
testDF['Variant'] = testDF['Variant'].replace('Mu', 10.0)
testDF['Variant'] = testDF['Variant'].replace('Omicron', 11.0)
columnNames=trainDF.columns

xTrain = trainDF.iloc[:, 1:-1]
yTrain = trainDF.iloc[:, -1:]
xTest = testDF.iloc[:, 1:-1]
yTest = testDF.iloc[:, -1:]
print(yTest)
columnNames=xTrain.columns
print(len(columnNames))
scaler1=StandardScaler()
xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)

xTrSA = xTrainScaled.values
xTeSA = xTestScaled.values
yTrSA = yTrain.values
yTeSA = yTest.values

xTrainTensor = torch.FloatTensor(xTrSA)
xTestTensor = torch.FloatTensor(xTeSA)
yTrainTensor = torch.FloatTensor(yTrSA)
yTestTensor = torch.FloatTensor(yTeSA)

criterion = torch.nn.CrossEntropyLoss()
print(model.parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
epochs = 1000
losses = []
rawPredictions = np.array([1])
realPredictions = np.array([1])
for i in range(epochs):
    yPred = model.forward(xTrainTensor)
    predictions = yPred.detach().numpy()
    for row in predictions:
        largest = -100000
        k=1
        for element in row:
            if element > largest:
                largest = element
                colVal=k
            k=k+1
        rawPredictions = np.vstack([rawPredictions, [largest]])
        realPredictions = np.vstack((realPredictions, [colVal]))
    rawPredictions = np.delete(rawPredictions, 0, axis = 0)
    print(rawPredictions)
    realPredictions = np.delete(realPredictions, 0, axis = 0)
    print(realPredictions)
    #yPred.unsqueeze()
    print(yPred)
    print(predictions)
    print(yTrainTensor)
    realPredTensor = torch.FloatTensor(realPredictions)
    loss = criterion(realPredTensor, yTrainTensor)
    losses.append(loss.detach().numpy())
    print(f'Epoch:  {i} and loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
class Model2(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=4, h3=6, h4=7, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x
    
model2 = Model2()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)
print(my_df)
X=my_df.drop('species', axis=1)
y = my_df['species']
X=X.values
y=y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""
X_train = (X_train.values)
X_test = (X_test.values)
y_train = (y_train.values)
y_test = (y_test.values)
"""
print(X_train)
print(y_train)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train) 
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
print(model2.parameters)
optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
epochs = 300
losses = []
for i in range(epochs):
    yPred = model2.forward(X_train)
    print(yPred)
    print(y_train)
    loss = criterion(yPred, y_train)

    losses.append(loss.detach().numpy())
    print(f'Epoch:  {i} and loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    ##for name, param in model2.named_parameters():
        ##print(name, param.grad)
    optimizer.step()