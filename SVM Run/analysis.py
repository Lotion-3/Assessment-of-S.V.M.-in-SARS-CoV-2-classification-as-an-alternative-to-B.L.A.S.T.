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

def timer(operation):
    start_time = time.time()
    operation
    end_time = time.time()
    timeTaken = (start_time-end_time)
    return timeTaken

def svmPredict(trainCSV, testCSV, resultsFile):
    start_time = time.time()
    trainDF = pd.read_csv(trainCSV)
    testDF = pd.read_csv(testCSV)
    xTrain = trainDF.iloc[:, 1:-1]
    yTrain = trainDF.iloc[:, -1:]
    xTest = testDF.iloc[:, 1:-1]
    yTest = testDF.iloc[:, -1:]
    scaler1=StandardScaler()
    xTrainScaled = scaler1.fit_transform(xTrain)
    xTestScaled = scaler1.fit_transform(xTest)
    print(xTrainScaled)
    print(xTestScaled)
    mod = SVC(C=100000000, kernel='rbf', decision_function_shape='ovr')
    mod.fit(xTrainScaled, yTrain.values.ravel())
    svmResults = open(resultsFile, "w")
    predicted = mod.predict(xTestScaled)
    i=0
    print(len(predicted))
    end_time = time.time()
    for i in range(0, (len(predicted))):
        svmResults.write(">"+testDF.iloc[i, 0]+"\n")
        svmResults.write(predicted[i]+"\n")
    return(end_time-start_time)

def accuracyScore(resultsFile):
    alphaList=[]
    betaList=[]
    deltaList=[]
    epsilonList=[]
    etaList=[]
    gammaList=[]
    iotaList=[]
    lambdaList=[]
    muList=[]
    omicronList=[]
    zetaError=[]
    alphaError = {'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    betaError = {'alpha':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    deltaError = {'alpha':[], 'beta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    epsilonError = {'alpha':[],'beta':[], 'delta':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    etaError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    gammaError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    iotaError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'lambda': [], 'mu': [], 'omicron': [], 'zeta':[]}
    lambdaError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'mu': [], 'omicron': [], 'zeta':[]}
    muError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'omicron': [], 'zeta':[]}
    omicronError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'zeta':[]}
    zetaError = {'alpha':[],'beta':[], 'delta':[], 'epsilon':[], 'eta':[], 'gamma':[], 'iota':[], 'lambda': [], 'mu': [], 'omicron': []}
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



"""
rawDF = pd.read_csv('rawSvmRuns5x10percInc.csv')
print(rawDF)
rawDic = rawDF.to_dict(orient='list')
rawDic["Prediction Time Taken"] = []
rawDic["Accuracy"] = []
predictionTimes=[]
accuracyList=[]


print("fit start")

rawDic['Prediction Time Taken']=predictionTimes
rawDic['Accuracy']=accuracyList
print(rawDic)

rawDF = pd.DataFrame.from_dict(rawDic)
rawDF.to_csv("rawSvmRuns5x10percInc.csv", index=False)

print("fit end")
"""