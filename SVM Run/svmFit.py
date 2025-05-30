from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance

testDF = pd.read_csv("test10(11000)(1)(extracted).csv", header=0)
xTest = testDF.iloc[:, 1:-1]
columnNames=xTest.columns

featuresList= ['G perc', 'C perc', 'A perc', 'T perc', 'GC perc', 'Alanine perc', 'Arginine perc', 'Asparagine perc', 'Aspartic Acid perc', 'Cysteine perc', 'Glutamine perc', 'Glutamic Acid perc', 'Glycine perc', 'Histidine perc', 'Isoleucine perc', 'Leucine perc', 'Lysine perc', 'Methionine perc', 'Phenylalanine perc', 'Proline perc', 'Serine perc', 'Threonine perc', 'Tryptophan perc', 'Tyrosine perc', 'Valine perc']
def timer(operation):
    start_time = time.time()
    operation
    end_time = time.time()
    timeTaken = (start_time-end_time)
    return timeTaken

def addColumns(resultsDictionary):
    for i in range(len(columnNames)):
        resultsDictionary[str(i+1)+" Features Model Time"]=[]
        resultsDictionary[str(i+1)+" Features Predict Time"]=[]
        resultsDictionary[str(i+1)+" Features Accuracy"]=[]

def svmPredict(trainCSV, testCSV, resultsDictionary):
    start_time = time.time()
    i=0
    
    trainDF = pd.read_csv(trainCSV)
    testDF = pd.read_csv(testCSV)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    
    for i in range(len(columnNames)):
        start_time = time.time()
        
        xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
        xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]
        start_time = time.time()
        mod = SVC(C=1000, kernel='rbf', decision_function_shape='ovr')
        mod.fit(xTrainFeatureScaled, yTrain.values.ravel())
        end_time = time.time()
        modelTime = set_time+end_time-start_time
        start_time = time.time()
        predicted=mod.predict(xTestFeatureScaled)
        end_time = time.time()
        predictTime = end_time-start_time
        
        k=0
        amountCorrect = 0
        for c in range(0, (len(predicted))):
            if yTest.iloc[c, 0] == predicted[c]:
                amountCorrect=amountCorrect+1
            k = k+1
        accuracy = (amountCorrect/k)
        #rawDF = pd.read_csv(resultsFile)
        #rawDic = rawDF.to_dict(orient='list')
        
        resultsDictionary[str(i+1)+" Features Model Time"].append(modelTime)
        resultsDictionary[str(i+1)+" Features Predict Time"].append(predictTime)
        resultsDictionary[str(i+1)+" Features Accuracy"].append(accuracy)

def accuracyScore(resultsFile):
    k=0
    amountCorrect = 0
    print(k)
    for sequence in SeqIO.parse(resultsFile, "fasta"):
        for keySeq in SeqIO.parse("key.fasta", "fasta"):
            if sequence.id == keySeq.id:
                if sequence.seq==keySeq.seq:
                    amountCorrect=amountCorrect+1
        k = k+1
    print(k)
    accuracy=((amountCorrect/k)*100)
    return accuracy

rawDF = pd.read_csv('rawSvmRuns5x10percInc.csv')
rawDic = rawDF.to_dict(orient='list')

addColumns(rawDic)

#rawDic["Prediction Time Taken"] = []
#rawDic["Accuracy"] = []
#predictionTimes=[]
#accuracyList=[]

print("fit start")

svmPredict("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv", rawDic)
svmPredict("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv", rawDic)
svmPredict("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv", rawDic)
svmPredict("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv", rawDic)
svmPredict("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv", rawDic)

svmPredict("train20(11000)(1)(extracted).csv", "test80(11000)(1)(extracted).csv", rawDic)
svmPredict("train20(11000)(2)(extracted).csv", "test80(11000)(2)(extracted).csv", rawDic)
svmPredict("train20(11000)(3)(extracted).csv", "test80(11000)(3)(extracted).csv", rawDic)
svmPredict("train20(11000)(4)(extracted).csv", "test80(11000)(4)(extracted).csv", rawDic)
svmPredict("train20(11000)(5)(extracted).csv", "test80(11000)(5)(extracted).csv", rawDic)

svmPredict("train30(11000)(1)(extracted).csv", "test70(11000)(1)(extracted).csv", rawDic)
svmPredict("train30(11000)(2)(extracted).csv", "test70(11000)(2)(extracted).csv", rawDic)
svmPredict("train30(11000)(3)(extracted).csv", "test70(11000)(3)(extracted).csv", rawDic)
svmPredict("train30(11000)(4)(extracted).csv", "test70(11000)(4)(extracted).csv", rawDic)
svmPredict("train30(11000)(5)(extracted).csv", "test70(11000)(5)(extracted).csv", rawDic)

svmPredict("train40(11000)(1)(extracted).csv", "test60(11000)(1)(extracted).csv", rawDic)
svmPredict("train40(11000)(2)(extracted).csv", "test60(11000)(2)(extracted).csv", rawDic)
svmPredict("train40(11000)(3)(extracted).csv", "test60(11000)(3)(extracted).csv", rawDic)
svmPredict("train40(11000)(4)(extracted).csv", "test60(11000)(4)(extracted).csv", rawDic)
svmPredict("train40(11000)(5)(extracted).csv", "test60(11000)(5)(extracted).csv", rawDic)

svmPredict("train50(11000)(1)(extracted).csv", "test50(11000)(1)(extracted).csv", rawDic)
svmPredict("train50(11000)(2)(extracted).csv", "test50(11000)(2)(extracted).csv", rawDic)
svmPredict("train50(11000)(3)(extracted).csv", "test50(11000)(3)(extracted).csv", rawDic)
svmPredict("train50(11000)(4)(extracted).csv", "test50(11000)(4)(extracted).csv", rawDic)
svmPredict("train50(11000)(5)(extracted).csv", "test50(11000)(5)(extracted).csv", rawDic)

svmPredict("train60(11000)(1)(extracted).csv", "test40(11000)(1)(extracted).csv", rawDic)
svmPredict("train60(11000)(2)(extracted).csv", "test40(11000)(2)(extracted).csv", rawDic)
svmPredict("train60(11000)(3)(extracted).csv", "test40(11000)(3)(extracted).csv", rawDic)
svmPredict("train60(11000)(4)(extracted).csv", "test40(11000)(4)(extracted).csv", rawDic)
svmPredict("train60(11000)(5)(extracted).csv", "test40(11000)(5)(extracted).csv", rawDic)

svmPredict("train70(11000)(1)(extracted).csv", "test30(11000)(1)(extracted).csv", rawDic)
svmPredict("train70(11000)(2)(extracted).csv", "test30(11000)(2)(extracted).csv", rawDic)
svmPredict("train70(11000)(3)(extracted).csv", "test30(11000)(3)(extracted).csv", rawDic)
svmPredict("train70(11000)(4)(extracted).csv", "test30(11000)(4)(extracted).csv", rawDic)
svmPredict("train70(11000)(5)(extracted).csv", "test30(11000)(5)(extracted).csv", rawDic)

svmPredict("train80(11000)(1)(extracted).csv", "test20(11000)(1)(extracted).csv", rawDic)
svmPredict("train80(11000)(2)(extracted).csv", "test20(11000)(2)(extracted).csv", rawDic)
svmPredict("train80(11000)(3)(extracted).csv", "test20(11000)(3)(extracted).csv", rawDic)
svmPredict("train80(11000)(4)(extracted).csv", "test20(11000)(4)(extracted).csv", rawDic)
svmPredict("train80(11000)(5)(extracted).csv", "test20(11000)(5)(extracted).csv", rawDic)

svmPredict("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv", rawDic)
svmPredict("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv", rawDic)
svmPredict("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv", rawDic)
svmPredict("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv", rawDic)
svmPredict("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv", rawDic)

"""
accuracyList.append((accuracyScore("shuffled11000(10-90)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(10-90)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(10-90)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(10-90)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(10-90)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(20-80)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(20-80)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(20-80)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(20-80)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(20-80)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(30-70)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(30-70)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(30-70)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(30-70)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(30-70)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(40-60)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(40-60)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(40-60)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(40-60)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(40-60)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(50-50)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(50-50)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(50-50)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(50-50)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(50-50)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(60-40)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(60-40)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(60-40)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(60-40)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(60-40)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(70-30)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(70-30)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(70-30)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(70-30)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(70-30)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(80-20)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(80-20)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(80-20)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(80-20)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(80-20)(5)(SVMresults).fasta")))

accuracyList.append((accuracyScore("shuffled11000(90-10)(1)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(90-10)(2)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(90-10)(3)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(90-10)(4)(SVMresults).fasta")))
accuracyList.append((accuracyScore("shuffled11000(90-10)(5)(SVMresults).fasta")))

rawDic['Prediction Time Taken']=predictionTimes
rawDic['Accuracy']=accuracyList
print(rawDic)

rawDF = pd.DataFrame.from_dict(rawDic)
rawDF.to_csv("rawSvmRuns5x10percInc.csv", index=False)
"""

"""
trainDF = pd.read_csv("train10(11000)(1)(extracted).csv", header=0)
testDF = pd.read_csv("test90(11000)(1)(extracted).csv", header=0)
columnNames=trainDF.columns
print(trainDF)
print(testDF)

xTrain = trainDF.iloc[:, 1:-1]
yTrain = trainDF.iloc[:, -1:]
xTest = testDF.iloc[:, 1:-1]
yTest = testDF.iloc[:, -1:]
columnNames=xTrain.columns
print(columnNames)

scaler1=StandardScaler()
xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)

combX = np.concatenate([xTrainScaled, xTestScaled])

print(xTrainScaled)
print(xTestScaled)
mod = SVC(C=100000000, kernel='rbf', decision_function_shape='ovr')
mod.fit(xTrainScaled, yTrain.values.ravel())
results = permutation_importance(mod, xTest, yTest, n_repeats=10)
for i, importance in enumerate(results.importances_mean):
    print(f"Feature {i+1}: {importance}")
"""
print(rawDic)
rawDF = pd.DataFrame.from_dict(rawDic)
rawDF.to_csv("abcdefghij.csv", index=False)
print("fit end")