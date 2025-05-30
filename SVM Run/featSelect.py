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
featuresList=xTest.columns
featuresAccDictionary = {'G perc':[], 'C perc':[], 'A perc':[], 'T perc':[], 'GC perc':[], 'Alanine perc': [], 'Arginine perc': [], 'Asparagine perc': [], 'Aspartic Acid perc': [], 'Cysteine perc': [], 'Glutamine perc': [], 'Glutamic Acid perc': [], 'Glycine perc': [], 'Histidine perc': [], 'Isoleucine perc': [], 'Leucine perc': [], 'Lysine perc': [], 'Methionine perc': [], 'Phenylalanine perc': [], 'Proline perc': [], 'Serine perc': [], 'Threonine perc': [], 'Tryptophan perc': [], 'Tyrosine perc': [], 'Valine perc': []}
featuresTimeDictionary = {'G perc':[], 'C perc':[], 'A perc':[], 'T perc':[], 'GC perc':[], 'Alanine perc': [], 'Arginine perc': [], 'Asparagine perc': [], 'Aspartic Acid perc': [], 'Cysteine perc': [], 'Glutamine perc': [], 'Glutamic Acid perc': [], 'Glycine perc': [], 'Histidine perc': [], 'Isoleucine perc': [], 'Leucine perc': [], 'Lysine perc': [], 'Methionine perc': [], 'Phenylalanine perc': [], 'Proline perc': [], 'Serine perc': [], 'Threonine perc': [], 'Tryptophan perc': [], 'Tyrosine perc': [], 'Valine perc': []}
def timer(operation):
    start_time = time.time()
    operation
    end_time = time.time()
    timeTaken = (start_time-end_time)
    return timeTaken

def svmPredict(trainCSV, testCSV, resultsFile):
    featPerf = {"Feature" : [], "Accuracy" : [], "Time" : []}
    accuracyList = []
    timesList = []

    trainDF = pd.read_csv(trainCSV)
    testDF = pd.read_csv(testCSV)
    columnNames=trainDF.columns

    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    columnNames=xTrain.columns
    print(len(columnNames))
    scaler1=StandardScaler()
    xTrainScaled = pd.DataFrame(scaler1.fit_transform(xTrain), columns=columnNames)
    xTestScaled = pd.DataFrame(scaler1.fit_transform(xTest), columns=columnNames)
    combX = np.concatenate([xTrainScaled, xTestScaled])

    for i in range(len(columnNames)):
        start_time = time.time()
        xTrainFeatureScaled = xTrainScaled.iloc[:, i:i+1]
        xTestFeatureScaled = xTestScaled.iloc[:, i:i+1]
        mod = SVC(C=1, kernel='rbf', decision_function_shape='ovr')
        mod.fit(xTrainFeatureScaled, yTrain.values.ravel())
        predicted=mod.predict(xTestFeatureScaled)
        end_time = time.time()
        totalTime = end_time-start_time
        timesList.append(totalTime)

        k=0
        amountCorrect = 0
        for c in range(0, (len(predicted))):
            if yTest.iloc[c, 0] == predicted[c]:
                amountCorrect=amountCorrect+1
            k = k+1
        accuracy = (amountCorrect/k)
        accuracyList.append(accuracy)
        featuresAccDictionary[columnNames[i]].append(accuracy)
        featuresTimeDictionary[columnNames[i]].append(totalTime)
        

    featPerf['Feature'] = columnNames
    featPerf['Accuracy'] = accuracyList
    featPerf['Time'] = timesList
    featPerfDF = pd.DataFrame.from_dict(featPerf)
    svmResults = open(resultsFile, "w")
    featPerfDF.to_csv(svmResults, index=False)
    print("predicted")

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


svmPredict("train10(11000)(1)(extracted).csv", "test90(11000)(1)(extracted).csv", "featurePerformance(10-90)(1).csv")
svmPredict("train10(11000)(2)(extracted).csv", "test90(11000)(2)(extracted).csv", "featurePerformance(10-90)(2).csv")
svmPredict("train10(11000)(3)(extracted).csv", "test90(11000)(3)(extracted).csv", "featurePerformance(10-90)(3).csv")
svmPredict("train10(11000)(4)(extracted).csv", "test90(11000)(4)(extracted).csv", "featurePerformance(10-90)(4).csv")
svmPredict("train10(11000)(5)(extracted).csv", "test90(11000)(5)(extracted).csv", "featurePerformance(10-90)(5).csv")

svmPredict("train20(11000)(1)(extracted).csv", "test80(11000)(1)(extracted).csv", "featurePerformance(20-80)(1).csv")
svmPredict("train20(11000)(2)(extracted).csv", "test80(11000)(2)(extracted).csv", "featurePerformance(20-80)(2).csv")
svmPredict("train20(11000)(3)(extracted).csv", "test80(11000)(3)(extracted).csv", "featurePerformance(20-80)(3).csv")
svmPredict("train20(11000)(4)(extracted).csv", "test80(11000)(4)(extracted).csv", "featurePerformance(20-80)(4).csv")
svmPredict("train20(11000)(5)(extracted).csv", "test80(11000)(5)(extracted).csv", "featurePerformance(20-80)(5).csv")

svmPredict("train30(11000)(1)(extracted).csv", "test70(11000)(1)(extracted).csv", "featurePerformance(30-70)(1).csv")
svmPredict("train30(11000)(2)(extracted).csv", "test70(11000)(2)(extracted).csv", "featurePerformance(30-70)(2).csv")
svmPredict("train30(11000)(3)(extracted).csv", "test70(11000)(3)(extracted).csv", "featurePerformance(30-70)(3).csv")
svmPredict("train30(11000)(4)(extracted).csv", "test70(11000)(4)(extracted).csv", "featurePerformance(30-70)(4).csv")
svmPredict("train30(11000)(5)(extracted).csv", "test70(11000)(5)(extracted).csv", "featurePerformance(30-70)(5).csv")

svmPredict("train40(11000)(1)(extracted).csv", "test60(11000)(1)(extracted).csv", "featurePerformance(40-60)(1).csv")
svmPredict("train40(11000)(2)(extracted).csv", "test60(11000)(2)(extracted).csv", "featurePerformance(40-60)(2).csv")
svmPredict("train40(11000)(3)(extracted).csv", "test60(11000)(3)(extracted).csv", "featurePerformance(40-60)(3).csv")
svmPredict("train40(11000)(4)(extracted).csv", "test60(11000)(4)(extracted).csv", "featurePerformance(40-60)(4).csv")
svmPredict("train40(11000)(5)(extracted).csv", "test60(11000)(5)(extracted).csv", "featurePerformance(40-60)(5).csv")

svmPredict("train50(11000)(1)(extracted).csv", "test50(11000)(1)(extracted).csv", "featurePerformance(50-50)(1).csv")
svmPredict("train50(11000)(2)(extracted).csv", "test50(11000)(2)(extracted).csv", "featurePerformance(50-50)(2).csv")
svmPredict("train50(11000)(3)(extracted).csv", "test50(11000)(3)(extracted).csv", "featurePerformance(50-50)(3).csv")
svmPredict("train50(11000)(4)(extracted).csv", "test50(11000)(4)(extracted).csv", "featurePerformance(50-50)(4).csv")
svmPredict("train50(11000)(5)(extracted).csv", "test50(11000)(5)(extracted).csv", "featurePerformance(50-50)(5).csv")

svmPredict("train60(11000)(1)(extracted).csv", "test40(11000)(1)(extracted).csv", "featurePerformance(60-40)(1).csv")
svmPredict("train60(11000)(2)(extracted).csv", "test40(11000)(2)(extracted).csv", "featurePerformance(60-40)(2).csv")
svmPredict("train60(11000)(3)(extracted).csv", "test40(11000)(3)(extracted).csv", "featurePerformance(60-40)(3).csv")
svmPredict("train60(11000)(4)(extracted).csv", "test40(11000)(4)(extracted).csv", "featurePerformance(60-40)(4).csv")
svmPredict("train60(11000)(5)(extracted).csv", "test40(11000)(5)(extracted).csv", "featurePerformance(60-40)(5).csv")

svmPredict("train70(11000)(1)(extracted).csv", "test30(11000)(1)(extracted).csv", "featurePerformance(70-30)(1).csv")
svmPredict("train70(11000)(2)(extracted).csv", "test30(11000)(2)(extracted).csv", "featurePerformance(70-30)(2).csv")
svmPredict("train70(11000)(3)(extracted).csv", "test30(11000)(3)(extracted).csv", "featurePerformance(70-30)(3).csv")
svmPredict("train70(11000)(4)(extracted).csv", "test30(11000)(4)(extracted).csv", "featurePerformance(70-30)(4).csv")
svmPredict("train70(11000)(5)(extracted).csv", "test30(11000)(5)(extracted).csv", "featurePerformance(70-30)(5).csv")

svmPredict("train80(11000)(1)(extracted).csv", "test20(11000)(1)(extracted).csv", "featurePerformance(80-20)(1).csv")
svmPredict("train80(11000)(2)(extracted).csv", "test20(11000)(2)(extracted).csv", "featurePerformance(80-20)(2).csv")
svmPredict("train80(11000)(3)(extracted).csv", "test20(11000)(3)(extracted).csv", "featurePerformance(80-20)(3).csv")
svmPredict("train80(11000)(4)(extracted).csv", "test20(11000)(4)(extracted).csv", "featurePerformance(80-20)(4).csv")
svmPredict("train80(11000)(5)(extracted).csv", "test20(11000)(5)(extracted).csv", "featurePerformance(80-20)(5).csv")

svmPredict("train90(11000)(1)(extracted).csv", "test10(11000)(1)(extracted).csv", "featurePerformance(90-10)(1).csv")
svmPredict("train90(11000)(2)(extracted).csv", "test10(11000)(2)(extracted).csv", "featurePerformance(90-10)(2).csv")
svmPredict("train90(11000)(3)(extracted).csv", "test10(11000)(3)(extracted).csv", "featurePerformance(90-10)(3).csv")
svmPredict("train90(11000)(4)(extracted).csv", "test10(11000)(4)(extracted).csv", "featurePerformance(90-10)(4).csv")
svmPredict("train90(11000)(5)(extracted).csv", "test10(11000)(5)(extracted).csv", "featurePerformance(90-10)(5).csv")

print(len(featuresList))
featOverallPerf = {"Feature" : [], "Accuracy" : [], "Time" : []}
featOverallPerf['Feature'] = featuresList
i=0
for key in featuresAccDictionary:
    i=i+1
    featOverallPerf['Accuracy'].append((sum(featuresAccDictionary[key]))/(len(featuresAccDictionary[key])+1))
    featOverallPerf['Time'].append((sum(featuresTimeDictionary[key]))/(len(featuresTimeDictionary[key])+1))
print(i)
svmResults = open("featureOverallPerformance.csv", "w")
featOverallDF = pd.DataFrame.from_dict(featOverallPerf)
featOverallDF.to_csv(svmResults, index=False)

print("fit end")