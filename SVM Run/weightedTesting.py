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

def svmWeightedPredict(trainCSV, testCSV, resultsDictionary):
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
"""
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
"""
print(rawDic)
rawDF = pd.DataFrame.from_dict(rawDic)
rawDF.to_csv("abcdefghij.csv", index=False)
"""
def unweightedPred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
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
    print(accuracy)
    return accuracy

def linearWeightPred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    start_time = time.time()
            
    xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
    xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]


    xTrainFeatureScaled["Histidine perc"]=xTrainFeatureScaled["Histidine perc"]*25
    xTrainFeatureScaled["Proline perc"]=xTrainFeatureScaled["Proline perc"]*24
    xTrainFeatureScaled["Glutamine perc"]=xTrainFeatureScaled["Glutamine perc"]*23
    xTrainFeatureScaled["Phenylalanine perc"]=xTrainFeatureScaled["Phenylalanine perc"]*22
    xTrainFeatureScaled["Threonine perc"]=xTrainFeatureScaled["Threonine perc"]*21
    xTrainFeatureScaled["Glutamic Acid perc"]=xTrainFeatureScaled["Glutamic Acid perc"]*20
    xTrainFeatureScaled["Tryptophan perc"]=xTrainFeatureScaled["Tryptophan perc"]*19
    xTrainFeatureScaled["Arginine perc"]=xTrainFeatureScaled["Arginine perc"]*18
    xTrainFeatureScaled["Serine perc"]=xTrainFeatureScaled["Serine perc"]*17
    xTrainFeatureScaled["Valine perc"]=xTrainFeatureScaled["Valine perc"]*16
    xTrainFeatureScaled["Aspartic Acid perc"]=xTrainFeatureScaled["Aspartic Acid perc"]*15
    xTrainFeatureScaled["A perc"]=xTrainFeatureScaled["A perc"]*14
    xTrainFeatureScaled["Cysteine perc"]=xTrainFeatureScaled["Cysteine perc"]*13
    xTrainFeatureScaled["Asparagine perc"]=xTrainFeatureScaled["Asparagine perc"]*12
    xTrainFeatureScaled["Glycine perc"]=xTrainFeatureScaled["Glycine perc"]*11
    xTrainFeatureScaled["T perc"]=xTrainFeatureScaled["T perc"]*10
    xTrainFeatureScaled["Lysine perc"]=xTrainFeatureScaled["Lysine perc"]*9
    xTrainFeatureScaled["Isoleucine perc"]=xTrainFeatureScaled["Isoleucine perc"]*8
    xTrainFeatureScaled["Alanine perc"]=xTrainFeatureScaled["Alanine perc"]*0
    xTrainFeatureScaled["Tyrosine perc"]=xTrainFeatureScaled["Tyrosine perc"]*0
    xTrainFeatureScaled["GC perc"]=xTrainFeatureScaled["GC perc"]*0
    xTrainFeatureScaled["Methionine perc"]=xTrainFeatureScaled["Methionine perc"]*0
    xTrainFeatureScaled["Leucine perc"]=xTrainFeatureScaled["Leucine perc"]*0
    xTrainFeatureScaled["C perc"]=xTrainFeatureScaled["C perc"]*0
    xTrainFeatureScaled["G perc"]=xTrainFeatureScaled["G perc"]*0

    xTestFeatureScaled["Histidine perc"]=xTestFeatureScaled["Histidine perc"]*25
    xTestFeatureScaled["Proline perc"]=xTestFeatureScaled["Proline perc"]*24
    xTestFeatureScaled["Glutamine perc"]=xTestFeatureScaled["Glutamine perc"]*23
    xTestFeatureScaled["Phenylalanine perc"]=xTestFeatureScaled["Phenylalanine perc"]*22
    xTestFeatureScaled["Threonine perc"]=xTestFeatureScaled["Threonine perc"]*21
    xTestFeatureScaled["Glutamic Acid perc"]=xTestFeatureScaled["Glutamic Acid perc"]*20
    xTestFeatureScaled["Tryptophan perc"]=xTestFeatureScaled["Tryptophan perc"]*19
    xTestFeatureScaled["Arginine perc"]=xTestFeatureScaled["Arginine perc"]*18
    xTestFeatureScaled["Serine perc"]=xTestFeatureScaled["Serine perc"]*17
    xTestFeatureScaled["Valine perc"]=xTestFeatureScaled["Valine perc"]*16
    xTestFeatureScaled["Aspartic Acid perc"]=xTestFeatureScaled["Aspartic Acid perc"]*15
    xTestFeatureScaled["A perc"]=xTestFeatureScaled["A perc"]*14
    xTestFeatureScaled["Cysteine perc"]=xTestFeatureScaled["Cysteine perc"]*13
    xTestFeatureScaled["Asparagine perc"]=xTestFeatureScaled["Asparagine perc"]*12
    xTestFeatureScaled["Glycine perc"]=xTestFeatureScaled["Glycine perc"]*11
    xTestFeatureScaled["T perc"]=xTestFeatureScaled["T perc"]*10
    xTestFeatureScaled["Lysine perc"]=xTestFeatureScaled["Lysine perc"]*9
    xTestFeatureScaled["Isoleucine perc"]=xTestFeatureScaled["Isoleucine perc"]*8
    xTestFeatureScaled["Alanine perc"]=xTestFeatureScaled["Alanine perc"]*0
    xTestFeatureScaled["Tyrosine perc"]=xTestFeatureScaled["Tyrosine perc"]*0
    xTestFeatureScaled["GC perc"]=xTestFeatureScaled["GC perc"]*0
    xTestFeatureScaled["Methionine perc"]=xTestFeatureScaled["Methionine perc"]*0
    xTestFeatureScaled["Leucine perc"]=xTestFeatureScaled["Leucine perc"]*0
    xTestFeatureScaled["C perc"]=xTestFeatureScaled["C perc"]*0
    xTestFeatureScaled["G perc"]=xTestFeatureScaled["G perc"]*0

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
    print(accuracy)

    return accuracy

def add1Pred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    start_time = time.time()
            
    xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
    xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]


    xTrainFeatureScaled["Histidine perc"]=xTrainFeatureScaled["Histidine perc"]*1.36
    xTrainFeatureScaled["Proline perc"]=xTrainFeatureScaled["Proline perc"]*1.35
    xTrainFeatureScaled["Glutamine perc"]=xTrainFeatureScaled["Glutamine perc"]*1.35
    xTrainFeatureScaled["Phenylalanine perc"]=xTrainFeatureScaled["Phenylalanine perc"]*1.34
    xTrainFeatureScaled["Threonine perc"]=xTrainFeatureScaled["Threonine perc"]*1.32
    xTrainFeatureScaled["Glutamic Acid perc"]=xTrainFeatureScaled["Glutamic Acid perc"]*1.31
    xTrainFeatureScaled["Tryptophan perc"]=xTrainFeatureScaled["Tryptophan perc"]*1.29
    xTrainFeatureScaled["Arginine perc"]=xTrainFeatureScaled["Arginine perc"]*1.28
    xTrainFeatureScaled["Serine perc"]=xTrainFeatureScaled["Serine perc"]*1.28
    xTrainFeatureScaled["Valine perc"]=xTrainFeatureScaled["Valine perc"]*1.28
    xTrainFeatureScaled["Aspartic Acid perc"]=xTrainFeatureScaled["Aspartic Acid perc"]*1.28
    xTrainFeatureScaled["A perc"]=xTrainFeatureScaled["A perc"]*1.27
    xTrainFeatureScaled["Cysteine perc"]=xTrainFeatureScaled["Cysteine perc"]*1.27
    xTrainFeatureScaled["Asparagine perc"]=xTrainFeatureScaled["Asparagine perc"]*1.27
    xTrainFeatureScaled["Glycine perc"]=xTrainFeatureScaled["Glycine perc"]*1.26
    xTrainFeatureScaled["T perc"]=xTrainFeatureScaled["T perc"]*1.25
    xTrainFeatureScaled["Lysine perc"]=xTrainFeatureScaled["Lysine perc"]*1.25
    xTrainFeatureScaled["Isoleucine perc"]=xTrainFeatureScaled["Isoleucine perc"]*1.25
    xTrainFeatureScaled["Alanine perc"]=xTrainFeatureScaled["Alanine perc"]*1.25
    xTrainFeatureScaled["Tyrosine perc"]=xTrainFeatureScaled["Tyrosine perc"]*1.23
    xTrainFeatureScaled["GC perc"]=xTrainFeatureScaled["GC perc"]*1.23
    xTrainFeatureScaled["Methionine perc"]=xTrainFeatureScaled["Methionine perc"]*1.23
    xTrainFeatureScaled["Leucine perc"]=xTrainFeatureScaled["Leucine perc"]*1.23
    xTrainFeatureScaled["C perc"]=xTrainFeatureScaled["C perc"]*1.21
    xTrainFeatureScaled["G perc"]=xTrainFeatureScaled["G perc"]*1.20

    xTestFeatureScaled["Histidine perc"]=xTestFeatureScaled["Histidine perc"]*1.36
    xTestFeatureScaled["Proline perc"]=xTestFeatureScaled["Proline perc"]*1.35
    xTestFeatureScaled["Glutamine perc"]=xTestFeatureScaled["Glutamine perc"]*1.35
    xTestFeatureScaled["Phenylalanine perc"]=xTestFeatureScaled["Phenylalanine perc"]*1.34
    xTestFeatureScaled["Threonine perc"]=xTestFeatureScaled["Threonine perc"]*1.32
    xTestFeatureScaled["Glutamic Acid perc"]=xTestFeatureScaled["Glutamic Acid perc"]*1.31
    xTestFeatureScaled["Tryptophan perc"]=xTestFeatureScaled["Tryptophan perc"]*1.29
    xTestFeatureScaled["Arginine perc"]=xTestFeatureScaled["Arginine perc"]*1.28
    xTestFeatureScaled["Serine perc"]=xTestFeatureScaled["Serine perc"]*1.28
    xTestFeatureScaled["Valine perc"]=xTestFeatureScaled["Valine perc"]*1.28
    xTestFeatureScaled["Aspartic Acid perc"]=xTestFeatureScaled["Aspartic Acid perc"]*1.28
    xTestFeatureScaled["A perc"]=xTestFeatureScaled["A perc"]*1.27
    xTestFeatureScaled["Cysteine perc"]=xTestFeatureScaled["Cysteine perc"]*1.27
    xTestFeatureScaled["Asparagine perc"]=xTestFeatureScaled["Asparagine perc"]*1.27
    xTestFeatureScaled["Glycine perc"]=xTestFeatureScaled["Glycine perc"]*1.26
    xTestFeatureScaled["T perc"]=xTestFeatureScaled["T perc"]*1.25
    xTestFeatureScaled["Lysine perc"]=xTestFeatureScaled["Lysine perc"]*1.25
    xTestFeatureScaled["Isoleucine perc"]=xTestFeatureScaled["Isoleucine perc"]*1.25
    xTestFeatureScaled["Alanine perc"]=xTestFeatureScaled["Alanine perc"]*1.25
    xTestFeatureScaled["Tyrosine perc"]=xTestFeatureScaled["Tyrosine perc"]*1.23
    xTestFeatureScaled["GC perc"]=xTestFeatureScaled["GC perc"]*1.23
    xTestFeatureScaled["Methionine perc"]=xTestFeatureScaled["Methionine perc"]*1.23
    xTestFeatureScaled["Leucine perc"]=xTestFeatureScaled["Leucine perc"]*1.23
    xTestFeatureScaled["C perc"]=xTestFeatureScaled["C perc"]*1.21
    xTestFeatureScaled["G perc"]=xTestFeatureScaled["G perc"]*1.20

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
    print(accuracy)

    return accuracy

def proportionalWeightPred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    start_time = time.time()
            
    xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
    xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]


    xTrainFeatureScaled["Histidine perc"]=xTrainFeatureScaled["Histidine perc"]*1.8
    xTrainFeatureScaled["Proline perc"]=xTrainFeatureScaled["Proline perc"]*1.75
    xTrainFeatureScaled["Glutamine perc"]=xTrainFeatureScaled["Glutamine perc"]*1.75
    xTrainFeatureScaled["Phenylalanine perc"]=xTrainFeatureScaled["Phenylalanine perc"]*1.7
    xTrainFeatureScaled["Threonine perc"]=xTrainFeatureScaled["Threonine perc"]*1.6
    xTrainFeatureScaled["Glutamic Acid perc"]=xTrainFeatureScaled["Glutamic Acid perc"]*1.55
    xTrainFeatureScaled["Tryptophan perc"]=xTrainFeatureScaled["Tryptophan perc"]*1.45
    xTrainFeatureScaled["Arginine perc"]=xTrainFeatureScaled["Arginine perc"]*1.4
    xTrainFeatureScaled["Serine perc"]=xTrainFeatureScaled["Serine perc"]*1.4
    xTrainFeatureScaled["Valine perc"]=xTrainFeatureScaled["Valine perc"]*1.4
    xTrainFeatureScaled["Aspartic Acid perc"]=xTrainFeatureScaled["Aspartic Acid perc"]*1.4
    xTrainFeatureScaled["A perc"]=xTrainFeatureScaled["A perc"]*1.35
    xTrainFeatureScaled["Cysteine perc"]=xTrainFeatureScaled["Cysteine perc"]*1.35
    xTrainFeatureScaled["Asparagine perc"]=xTrainFeatureScaled["Asparagine perc"]*1.35
    xTrainFeatureScaled["Glycine perc"]=xTrainFeatureScaled["Glycine perc"]*1.3
    xTrainFeatureScaled["T perc"]=xTrainFeatureScaled["T perc"]*1.25
    xTrainFeatureScaled["Lysine perc"]=xTrainFeatureScaled["Lysine perc"]*1.25
    xTrainFeatureScaled["Isoleucine perc"]=xTrainFeatureScaled["Isoleucine perc"]*1.25
    xTrainFeatureScaled["Alanine perc"]=xTrainFeatureScaled["Alanine perc"]*1.25
    xTrainFeatureScaled["Tyrosine perc"]=xTrainFeatureScaled["Tyrosine perc"]*1.15
    xTrainFeatureScaled["GC perc"]=xTrainFeatureScaled["GC perc"]*1.15
    xTrainFeatureScaled["Methionine perc"]=xTrainFeatureScaled["Methionine perc"]*1.15
    xTrainFeatureScaled["Leucine perc"]=xTrainFeatureScaled["Leucine perc"]*1.15
    xTrainFeatureScaled["C perc"]=xTrainFeatureScaled["C perc"]*1.05
    xTrainFeatureScaled["G perc"]=xTrainFeatureScaled["G perc"]*1.00

    xTestFeatureScaled["Histidine perc"]=xTestFeatureScaled["Histidine perc"]*1.8
    xTestFeatureScaled["Proline perc"]=xTestFeatureScaled["Proline perc"]*1.75
    xTestFeatureScaled["Glutamine perc"]=xTestFeatureScaled["Glutamine perc"]*1.75
    xTestFeatureScaled["Phenylalanine perc"]=xTestFeatureScaled["Phenylalanine perc"]*1.7
    xTestFeatureScaled["Threonine perc"]=xTestFeatureScaled["Threonine perc"]*1.6
    xTestFeatureScaled["Glutamic Acid perc"]=xTestFeatureScaled["Glutamic Acid perc"]*1.55
    xTestFeatureScaled["Tryptophan perc"]=xTestFeatureScaled["Tryptophan perc"]*1.45
    xTestFeatureScaled["Arginine perc"]=xTestFeatureScaled["Arginine perc"]*1.4
    xTestFeatureScaled["Serine perc"]=xTestFeatureScaled["Serine perc"]*1.4
    xTestFeatureScaled["Valine perc"]=xTestFeatureScaled["Valine perc"]*1.4
    xTestFeatureScaled["Aspartic Acid perc"]=xTestFeatureScaled["Aspartic Acid perc"]*1.4
    xTestFeatureScaled["A perc"]=xTestFeatureScaled["A perc"]*1.35
    xTestFeatureScaled["Cysteine perc"]=xTestFeatureScaled["Cysteine perc"]*1.35
    xTestFeatureScaled["Asparagine perc"]=xTestFeatureScaled["Asparagine perc"]*1.35
    xTestFeatureScaled["Glycine perc"]=xTestFeatureScaled["Glycine perc"]*1.3
    xTestFeatureScaled["T perc"]=xTestFeatureScaled["T perc"]*1.25
    xTestFeatureScaled["Lysine perc"]=xTestFeatureScaled["Lysine perc"]*1.25
    xTestFeatureScaled["Isoleucine perc"]=xTestFeatureScaled["Isoleucine perc"]*1.25
    xTestFeatureScaled["Alanine perc"]=xTestFeatureScaled["Alanine perc"]*1.25
    xTestFeatureScaled["Tyrosine perc"]=xTestFeatureScaled["Tyrosine perc"]*1.15
    xTestFeatureScaled["GC perc"]=xTestFeatureScaled["GC perc"]*1.15
    xTestFeatureScaled["Methionine perc"]=xTestFeatureScaled["Methionine perc"]*1.15
    xTestFeatureScaled["Leucine perc"]=xTestFeatureScaled["Leucine perc"]*1.15
    xTestFeatureScaled["C perc"]=xTestFeatureScaled["C perc"]*1.05
    xTestFeatureScaled["G perc"]=xTestFeatureScaled["G perc"]*1.00

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
    print(accuracy)

    return accuracy

def reducedWeightPred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    start_time = time.time()
            
    xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
    xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]


    xTrainFeatureScaled["Histidine perc"]=xTrainFeatureScaled["Histidine perc"]*0.36
    xTrainFeatureScaled["Proline perc"]=xTrainFeatureScaled["Proline perc"]*0.35
    xTrainFeatureScaled["Glutamine perc"]=xTrainFeatureScaled["Glutamine perc"]*0.35
    xTrainFeatureScaled["Phenylalanine perc"]=xTrainFeatureScaled["Phenylalanine perc"]*0.34
    xTrainFeatureScaled["Threonine perc"]=xTrainFeatureScaled["Threonine perc"]*0.32
    xTrainFeatureScaled["Glutamic Acid perc"]=xTrainFeatureScaled["Glutamic Acid perc"]*0.31
    xTrainFeatureScaled["Tryptophan perc"]=xTrainFeatureScaled["Tryptophan perc"]*0.29
    xTrainFeatureScaled["Arginine perc"]=xTrainFeatureScaled["Arginine perc"]*0.28
    xTrainFeatureScaled["Serine perc"]=xTrainFeatureScaled["Serine perc"]*0.28
    xTrainFeatureScaled["Valine perc"]=xTrainFeatureScaled["Valine perc"]*0.28
    xTrainFeatureScaled["Aspartic Acid perc"]=xTrainFeatureScaled["Aspartic Acid perc"]*0.28
    xTrainFeatureScaled["A perc"]=xTrainFeatureScaled["A perc"]*0.27
    xTrainFeatureScaled["Cysteine perc"]=xTrainFeatureScaled["Cysteine perc"]*0.27
    xTrainFeatureScaled["Asparagine perc"]=xTrainFeatureScaled["Asparagine perc"]*0.27
    xTrainFeatureScaled["Glycine perc"]=xTrainFeatureScaled["Glycine perc"]*0.26
    xTrainFeatureScaled["T perc"]=xTrainFeatureScaled["T perc"]*0.25
    xTrainFeatureScaled["Lysine perc"]=xTrainFeatureScaled["Lysine perc"]*0.25
    xTrainFeatureScaled["Isoleucine perc"]=xTrainFeatureScaled["Isoleucine perc"]*0.25
    xTrainFeatureScaled["Alanine perc"]=xTrainFeatureScaled["Alanine perc"]*0.25
    xTrainFeatureScaled["Tyrosine perc"]=xTrainFeatureScaled["Tyrosine perc"]*0.23
    xTrainFeatureScaled["GC perc"]=xTrainFeatureScaled["GC perc"]*0.23
    xTrainFeatureScaled["Methionine perc"]=xTrainFeatureScaled["Methionine perc"]*0.23
    xTrainFeatureScaled["Leucine perc"]=xTrainFeatureScaled["Leucine perc"]*0.23
    xTrainFeatureScaled["C perc"]=xTrainFeatureScaled["C perc"]*0.21
    xTrainFeatureScaled["G perc"]=xTrainFeatureScaled["G perc"]*0.20

    xTestFeatureScaled["Histidine perc"]=xTestFeatureScaled["Histidine perc"]*0.36
    xTestFeatureScaled["Proline perc"]=xTestFeatureScaled["Proline perc"]*0.35
    xTestFeatureScaled["Glutamine perc"]=xTestFeatureScaled["Glutamine perc"]*0.35
    xTestFeatureScaled["Phenylalanine perc"]=xTestFeatureScaled["Phenylalanine perc"]*0.34
    xTestFeatureScaled["Threonine perc"]=xTestFeatureScaled["Threonine perc"]*0.32
    xTestFeatureScaled["Glutamic Acid perc"]=xTestFeatureScaled["Glutamic Acid perc"]*0.31
    xTestFeatureScaled["Tryptophan perc"]=xTestFeatureScaled["Tryptophan perc"]*0.29
    xTestFeatureScaled["Arginine perc"]=xTestFeatureScaled["Arginine perc"]*0.28
    xTestFeatureScaled["Serine perc"]=xTestFeatureScaled["Serine perc"]*0.28
    xTestFeatureScaled["Valine perc"]=xTestFeatureScaled["Valine perc"]*0.28
    xTestFeatureScaled["Aspartic Acid perc"]=xTestFeatureScaled["Aspartic Acid perc"]*0.28
    xTestFeatureScaled["A perc"]=xTestFeatureScaled["A perc"]*0.27
    xTestFeatureScaled["Cysteine perc"]=xTestFeatureScaled["Cysteine perc"]*0.27
    xTestFeatureScaled["Asparagine perc"]=xTestFeatureScaled["Asparagine perc"]*0.27
    xTestFeatureScaled["Glycine perc"]=xTestFeatureScaled["Glycine perc"]*0.26
    xTestFeatureScaled["T perc"]=xTestFeatureScaled["T perc"]*0.25
    xTestFeatureScaled["Lysine perc"]=xTestFeatureScaled["Lysine perc"]*0.25
    xTestFeatureScaled["Isoleucine perc"]=xTestFeatureScaled["Isoleucine perc"]*0.25
    xTestFeatureScaled["Alanine perc"]=xTestFeatureScaled["Alanine perc"]*0.25
    xTestFeatureScaled["Tyrosine perc"]=xTestFeatureScaled["Tyrosine perc"]*0.23
    xTestFeatureScaled["GC perc"]=xTestFeatureScaled["GC perc"]*0.23
    xTestFeatureScaled["Methionine perc"]=xTestFeatureScaled["Methionine perc"]*0.23
    xTestFeatureScaled["Leucine perc"]=xTestFeatureScaled["Leucine perc"]*0.23
    xTestFeatureScaled["C perc"]=xTestFeatureScaled["C perc"]*0.21
    xTestFeatureScaled["G perc"]=xTestFeatureScaled["G perc"]*0.20

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
    print(accuracy)

    return accuracy

def invLinearWeightPred(trainFile, testFile):
    start_time = time.time()
    i=25
    trainDF = pd.read_csv(trainFile)
    testDF = pd.read_csv(testFile)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    end_time = time.time()
    set_time = end_time-start_time
    start_time = time.time()
            
    xTrainFeatureScaled = xTrainScaled.iloc[:, 0:i+1]
    xTestFeatureScaled = xTestScaled.iloc[:, 0:i+1]


    xTrainFeatureScaled["Histidine perc"]=xTrainFeatureScaled["Histidine perc"]*1
    xTrainFeatureScaled["Proline perc"]=xTrainFeatureScaled["Proline perc"]*2
    xTrainFeatureScaled["Glutamine perc"]=xTrainFeatureScaled["Glutamine perc"]*3
    xTrainFeatureScaled["Phenylalanine perc"]=xTrainFeatureScaled["Phenylalanine perc"]*4
    xTrainFeatureScaled["Threonine perc"]=xTrainFeatureScaled["Threonine perc"]*5
    xTrainFeatureScaled["Glutamic Acid perc"]=xTrainFeatureScaled["Glutamic Acid perc"]*6
    xTrainFeatureScaled["Tryptophan perc"]=xTrainFeatureScaled["Tryptophan perc"]*7
    xTrainFeatureScaled["Arginine perc"]=xTrainFeatureScaled["Arginine perc"]*8
    xTrainFeatureScaled["Serine perc"]=xTrainFeatureScaled["Serine perc"]*9
    xTrainFeatureScaled["Valine perc"]=xTrainFeatureScaled["Valine perc"]*10
    xTrainFeatureScaled["Aspartic Acid perc"]=xTrainFeatureScaled["Aspartic Acid perc"]*11
    xTrainFeatureScaled["A perc"]=xTrainFeatureScaled["A perc"]*12
    xTrainFeatureScaled["Cysteine perc"]=xTrainFeatureScaled["Cysteine perc"]*13
    xTrainFeatureScaled["Asparagine perc"]=xTrainFeatureScaled["Asparagine perc"]*14
    xTrainFeatureScaled["Glycine perc"]=xTrainFeatureScaled["Glycine perc"]*15
    xTrainFeatureScaled["T perc"]=xTrainFeatureScaled["T perc"]*16
    xTrainFeatureScaled["Lysine perc"]=xTrainFeatureScaled["Lysine perc"]*17
    xTrainFeatureScaled["Isoleucine perc"]=xTrainFeatureScaled["Isoleucine perc"]*18
    xTrainFeatureScaled["Alanine perc"]=xTrainFeatureScaled["Alanine perc"]*19
    xTrainFeatureScaled["Tyrosine perc"]=xTrainFeatureScaled["Tyrosine perc"]*20
    xTrainFeatureScaled["GC perc"]=xTrainFeatureScaled["GC perc"]*21
    xTrainFeatureScaled["Methionine perc"]=xTrainFeatureScaled["Methionine perc"]*22
    xTrainFeatureScaled["Leucine perc"]=xTrainFeatureScaled["Leucine perc"]*23
    xTrainFeatureScaled["C perc"]=xTrainFeatureScaled["C perc"]*24
    xTrainFeatureScaled["G perc"]=xTrainFeatureScaled["G perc"]*25

    xTestFeatureScaled["Histidine perc"]=xTestFeatureScaled["Histidine perc"]*1
    xTestFeatureScaled["Proline perc"]=xTestFeatureScaled["Proline perc"]*2
    xTestFeatureScaled["Glutamine perc"]=xTestFeatureScaled["Glutamine perc"]*3
    xTestFeatureScaled["Phenylalanine perc"]=xTestFeatureScaled["Phenylalanine perc"]*4
    xTestFeatureScaled["Threonine perc"]=xTestFeatureScaled["Threonine perc"]*5
    xTestFeatureScaled["Glutamic Acid perc"]=xTestFeatureScaled["Glutamic Acid perc"]*6
    xTestFeatureScaled["Tryptophan perc"]=xTestFeatureScaled["Tryptophan perc"]*7
    xTestFeatureScaled["Arginine perc"]=xTestFeatureScaled["Arginine perc"]*8
    xTestFeatureScaled["Serine perc"]=xTestFeatureScaled["Serine perc"]*9
    xTestFeatureScaled["Valine perc"]=xTestFeatureScaled["Valine perc"]*10
    xTestFeatureScaled["Aspartic Acid perc"]=xTestFeatureScaled["Aspartic Acid perc"]*11
    xTestFeatureScaled["A perc"]=xTestFeatureScaled["A perc"]*12
    xTestFeatureScaled["Cysteine perc"]=xTestFeatureScaled["Cysteine perc"]*13
    xTestFeatureScaled["Asparagine perc"]=xTestFeatureScaled["Asparagine perc"]*14
    xTestFeatureScaled["Glycine perc"]=xTestFeatureScaled["Glycine perc"]*15
    xTestFeatureScaled["T perc"]=xTestFeatureScaled["T perc"]*16
    xTestFeatureScaled["Lysine perc"]=xTestFeatureScaled["Lysine perc"]*17
    xTestFeatureScaled["Isoleucine perc"]=xTestFeatureScaled["Isoleucine perc"]*18
    xTestFeatureScaled["Alanine perc"]=xTestFeatureScaled["Alanine perc"]*19
    xTestFeatureScaled["Tyrosine perc"]=xTestFeatureScaled["Tyrosine perc"]*20
    xTestFeatureScaled["GC perc"]=xTestFeatureScaled["GC perc"]*21
    xTestFeatureScaled["Methionine perc"]=xTestFeatureScaled["Methionine perc"]*22
    xTestFeatureScaled["Leucine perc"]=xTestFeatureScaled["Leucine perc"]*23
    xTestFeatureScaled["C perc"]=xTestFeatureScaled["C perc"]*24
    xTestFeatureScaled["G perc"]=xTestFeatureScaled["G perc"]*25

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
    print(accuracy)

    return accuracy

print("fit end")
unweighted10list=[]
unweighted10list.append(unweightedPred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
unweighted10list.append(unweightedPred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
unweighted10list.append(unweightedPred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
unweighted10list.append(unweightedPred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
unweighted10list.append(unweightedPred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("Unweight 10 Avg " + str(((sum(unweighted10list))/(len(unweighted10list)))))
linear10list=[]
linear10list.append(linearWeightPred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
linear10list.append(linearWeightPred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
linear10list.append(linearWeightPred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
linear10list.append(linearWeightPred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
linear10list.append(linearWeightPred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("Linear 10 Avg " + str(((sum(linear10list))/(len(linear10list)))))
add110list=[]
add110list.append(add1Pred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
add110list.append(add1Pred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
add110list.append(add1Pred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
add110list.append(add1Pred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
add110list.append(add1Pred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("+1 10 Avg " + str(((sum(add110list))/(len(add110list)))))
prop10list=[]
prop10list.append(proportionalWeightPred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
prop10list.append(proportionalWeightPred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
prop10list.append(proportionalWeightPred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
prop10list.append(proportionalWeightPred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
prop10list.append(proportionalWeightPred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("Proportional 10 Avg " + str(((sum(prop10list))/(len(prop10list)))))

red10list=[]
red10list.append(reducedWeightPred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
red10list.append(reducedWeightPred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
red10list.append(reducedWeightPred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
red10list.append(reducedWeightPred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
red10list.append(reducedWeightPred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("Reduced 10 Avg " + str(((sum(red10list))/(len(red10list)))))
invLinear10list=[]
invLinear10list.append(invLinearWeightPred("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv"))
invLinear10list.append(invLinearWeightPred("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv"))
invLinear10list.append(invLinearWeightPred("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv"))
invLinear10list.append(invLinearWeightPred("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv"))
invLinear10list.append(invLinearWeightPred("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv"))
print("Inverted Linear 10 Avg " + str(((sum(invLinear10list))/(len(invLinear10list)))))


unweighted90list=[]
unweighted90list.append(unweightedPred("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv"))
unweighted90list.append(unweightedPred("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv"))
unweighted90list.append(unweightedPred("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv"))
unweighted90list.append(unweightedPred("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv"))
unweighted90list.append(unweightedPred("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv"))
print("Unweight 90 Avg " + str(((sum(unweighted90list))/(len(unweighted90list)))))
linear90list=[]
linear90list.append(linearWeightPred("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv"))
linear90list.append(linearWeightPred("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv"))
linear90list.append(linearWeightPred("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv"))
linear90list.append(linearWeightPred("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv"))
linear90list.append(linearWeightPred("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv"))
print("Linear 90 Avg " + str(((sum(linear90list))/(len(linear90list)))))
add190list=[]
add190list.append(add1Pred("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv"))
add190list.append(add1Pred("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv"))
add190list.append(add1Pred("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv"))
add190list.append(add1Pred("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv"))
add190list.append(add1Pred("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv"))
print("+1 90 Avg " + str(((sum(add190list))/(len(add190list)))))
prop90list=[]
prop90list.append(proportionalWeightPred("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv"))
prop90list.append(proportionalWeightPred("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv"))
prop90list.append(proportionalWeightPred("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv"))
prop90list.append(proportionalWeightPred("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv"))
prop90list.append(proportionalWeightPred("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv"))
print("Proportional 90 Avg " + str(((sum(prop90list))/(len(prop90list)))))

red90list=[]
red90list.append(reducedWeightPred("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv"))
red90list.append(reducedWeightPred("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv"))
red90list.append(reducedWeightPred("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv"))
red90list.append(reducedWeightPred("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv"))
red90list.append(reducedWeightPred("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv"))
print("Reduced 90 Avg " + str(((sum(red90list))/(len(red90list)))))