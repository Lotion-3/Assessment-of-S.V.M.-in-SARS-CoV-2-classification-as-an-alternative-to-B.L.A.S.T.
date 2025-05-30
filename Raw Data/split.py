import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

def split(file, percTrain, percTest, trainFile, testFile):
    testCreate=open(testFile, "w")
    trainCreate=open(trainFile, "w")
    k=0
    for sequence in SeqIO.parse(file, "fasta"):
        k=k+1
    k=round(k*percTrain)
    print(k)
    h=0
    for sequence in SeqIO.parse(file, "fasta"):
        h=h+1
        if h<=k:
            trainCreate.write(">"+sequence.id + "\n")
            trainCreate.write(str(sequence.seq)+"\n")
            print(h)
        else:
            testCreate.write(">"+sequence.id + "\n")
            testCreate.write(str(sequence.seq)+"\n")

print("split start")

split("shuffled11000(10-90)(1).fasta", 0.1, 0.9, "train10(11000)(1).fasta", "test90(11000)(1).fasta")
split("shuffled11000(10-90)(2).fasta", 0.1, 0.9, "train10(11000)(2).fasta", "test90(11000)(2).fasta")
split("shuffled11000(10-90)(3).fasta", 0.1, 0.9, "train10(11000)(3).fasta", "test90(11000)(3).fasta")
split("shuffled11000(10-90)(4).fasta", 0.1, 0.9, "train10(11000)(4).fasta", "test90(11000)(4).fasta")
split("shuffled11000(10-90)(5).fasta", 0.1, 0.9, "train10(11000)(5).fasta", "test90(11000)(5).fasta")

split("shuffled11000(20-80)(1).fasta", 0.2, 0.8, "train20(11000)(1).fasta", "test80(11000)(1).fasta")
split("shuffled11000(20-80)(2).fasta", 0.2, 0.8, "train20(11000)(2).fasta", "test80(11000)(2).fasta")
split("shuffled11000(20-80)(3).fasta", 0.2, 0.8, "train20(11000)(3).fasta", "test80(11000)(3).fasta")
split("shuffled11000(20-80)(4).fasta", 0.2, 0.8, "train20(11000)(4).fasta", "test80(11000)(4).fasta")
split("shuffled11000(20-80)(5).fasta", 0.2, 0.8, "train20(11000)(5).fasta", "test80(11000)(5).fasta")

split("shuffled11000(30-70)(1).fasta", 0.3, 0.7, "train30(11000)(1).fasta", "test70(11000)(1).fasta")
split("shuffled11000(30-70)(2).fasta", 0.3, 0.7, "train30(11000)(2).fasta", "test70(11000)(2).fasta")
split("shuffled11000(30-70)(3).fasta", 0.3, 0.7, "train30(11000)(3).fasta", "test70(11000)(3).fasta")
split("shuffled11000(30-70)(4).fasta", 0.3, 0.7, "train30(11000)(4).fasta", "test70(11000)(4).fasta")
split("shuffled11000(30-70)(5).fasta", 0.3, 0.7, "train30(11000)(5).fasta", "test70(11000)(5).fasta")

split("shuffled11000(40-60)(1).fasta", 0.4, 0.6, "train40(11000)(1).fasta", "test60(11000)(1).fasta")
split("shuffled11000(40-60)(2).fasta", 0.4, 0.6, "train40(11000)(2).fasta", "test60(11000)(2).fasta")
split("shuffled11000(40-60)(3).fasta", 0.4, 0.6, "train40(11000)(3).fasta", "test60(11000)(3).fasta")
split("shuffled11000(40-60)(4).fasta", 0.4, 0.6, "train40(11000)(4).fasta", "test60(11000)(4).fasta")
split("shuffled11000(40-60)(5).fasta", 0.4, 0.6, "train40(11000)(5).fasta", "test60(11000)(5).fasta")

split("shuffled11000(50-50)(1).fasta", 0.5, 0.5, "train50(11000)(1).fasta", "test50(11000)(1).fasta")
split("shuffled11000(50-50)(2).fasta", 0.5, 0.5, "train50(11000)(2).fasta", "test50(11000)(2).fasta")
split("shuffled11000(50-50)(3).fasta", 0.5, 0.5, "train50(11000)(3).fasta", "test50(11000)(3).fasta")
split("shuffled11000(50-50)(4).fasta", 0.5, 0.5, "train50(11000)(4).fasta", "test50(11000)(4).fasta")
split("shuffled11000(50-50)(5).fasta", 0.5, 0.5, "train50(11000)(5).fasta", "test50(11000)(5).fasta")

split("shuffled11000(60-40)(1).fasta", 0.6, 0.4, "train60(11000)(1).fasta", "test40(11000)(1).fasta")
split("shuffled11000(60-40)(2).fasta", 0.6, 0.4, "train60(11000)(2).fasta", "test40(11000)(2).fasta")
split("shuffled11000(60-40)(3).fasta", 0.6, 0.4, "train60(11000)(3).fasta", "test40(11000)(3).fasta")
split("shuffled11000(60-40)(4).fasta", 0.6, 0.4, "train60(11000)(4).fasta", "test40(11000)(4).fasta")
split("shuffled11000(60-40)(5).fasta", 0.6, 0.4, "train60(11000)(5).fasta", "test40(11000)(5).fasta")

split("shuffled11000(70-30)(1).fasta", 0.7, 0.3, "train70(11000)(1).fasta", "test30(11000)(1).fasta")
split("shuffled11000(70-30)(2).fasta", 0.7, 0.3, "train70(11000)(2).fasta", "test30(11000)(2).fasta")
split("shuffled11000(70-30)(3).fasta", 0.7, 0.3, "train70(11000)(3).fasta", "test30(11000)(3).fasta")
split("shuffled11000(70-30)(4).fasta", 0.7, 0.3, "train70(11000)(4).fasta", "test30(11000)(4).fasta")
split("shuffled11000(70-30)(5).fasta", 0.7, 0.3, "train70(11000)(5).fasta", "test30(11000)(5).fasta")

split("shuffled11000(80-20)(1).fasta", 0.8, 0.2, "train80(11000)(1).fasta", "test20(11000)(1).fasta")
split("shuffled11000(80-20)(2).fasta", 0.8, 0.2, "train80(11000)(2).fasta", "test20(11000)(2).fasta")
split("shuffled11000(80-20)(3).fasta", 0.8, 0.2, "train80(11000)(3).fasta", "test20(11000)(3).fasta")
split("shuffled11000(80-20)(4).fasta", 0.8, 0.2, "train80(11000)(4).fasta", "test20(11000)(4).fasta")
split("shuffled11000(80-20)(5).fasta", 0.8, 0.2, "train80(11000)(5).fasta", "test20(11000)(5).fasta")

split("shuffled11000(90-10)(1).fasta", 0.9, 0.1, "train90(11000)(1).fasta", "test10(11000)(1).fasta")
split("shuffled11000(90-10)(2).fasta", 0.9, 0.1, "train90(11000)(2).fasta", "test10(11000)(2).fasta")
split("shuffled11000(90-10)(3).fasta", 0.9, 0.1, "train90(11000)(3).fasta", "test10(11000)(3).fasta")
split("shuffled11000(90-10)(4).fasta", 0.9, 0.1, "train90(11000)(4).fasta", "test10(11000)(4).fasta")
split("shuffled11000(90-10)(5).fasta", 0.9, 0.1, "train90(11000)(5).fasta", "test10(11000)(5).fasta")

rawDic = {"File Name":[]}
filesNamesList=[]

filesNamesList.append("shuffled11000(10-90)(1).fasta")
filesNamesList.append("shuffled11000(10-90)(2).fasta")
filesNamesList.append("shuffled11000(10-90)(3).fasta")
filesNamesList.append("shuffled11000(10-90)(4).fasta")
filesNamesList.append("shuffled11000(10-90)(5).fasta")

filesNamesList.append("shuffled11000(20-80)(1).fasta")
filesNamesList.append("shuffled11000(20-80)(2).fasta")
filesNamesList.append("shuffled11000(20-80)(3).fasta")
filesNamesList.append("shuffled11000(20-80)(4).fasta")
filesNamesList.append("shuffled11000(20-80)(5).fasta")

filesNamesList.append("shuffled11000(30-70)(1).fasta")
filesNamesList.append("shuffled11000(30-70)(2).fasta")
filesNamesList.append("shuffled11000(30-70)(3).fasta")
filesNamesList.append("shuffled11000(30-70)(4).fasta")
filesNamesList.append("shuffled11000(30-70)(5).fasta")

filesNamesList.append("shuffled11000(40-60)(1).fasta")
filesNamesList.append("shuffled11000(40-60)(2).fasta")
filesNamesList.append("shuffled11000(40-60)(3).fasta")
filesNamesList.append("shuffled11000(40-60)(4).fasta")
filesNamesList.append("shuffled11000(40-60)(5).fasta")

filesNamesList.append("shuffled11000(50-50)(1).fasta")
filesNamesList.append("shuffled11000(50-50)(2).fasta")
filesNamesList.append("shuffled11000(50-50)(3).fasta")
filesNamesList.append("shuffled11000(50-50)(4).fasta")
filesNamesList.append("shuffled11000(50-50)(5).fasta")

filesNamesList.append("shuffled11000(60-40)(1).fasta")
filesNamesList.append("shuffled11000(60-40)(2).fasta")
filesNamesList.append("shuffled11000(60-40)(3).fasta")
filesNamesList.append("shuffled11000(60-40)(4).fasta")
filesNamesList.append("shuffled11000(60-40)(5).fasta")

filesNamesList.append("shuffled11000(70-30)(1).fasta")
filesNamesList.append("shuffled11000(70-30)(2).fasta")
filesNamesList.append("shuffled11000(70-30)(3).fasta")
filesNamesList.append("shuffled11000(70-30)(4).fasta")
filesNamesList.append("shuffled11000(70-30)(5).fasta")

filesNamesList.append("shuffled11000(80-20)(1).fasta")
filesNamesList.append("shuffled11000(80-20)(2).fasta")
filesNamesList.append("shuffled11000(80-20)(3).fasta")
filesNamesList.append("shuffled11000(80-20)(4).fasta")
filesNamesList.append("shuffled11000(80-20)(5).fasta")

filesNamesList.append("shuffled11000(90-10)(1).fasta")
filesNamesList.append("shuffled11000(90-10)(2).fasta")
filesNamesList.append("shuffled11000(90-10)(3).fasta")
filesNamesList.append("shuffled11000(90-10)(4).fasta")
filesNamesList.append("shuffled11000(90-10)(5).fasta")

rawDic['File Name']=filesNamesList
rawDF = pd.DataFrame.from_dict(rawDic)
rawDF.to_csv("rawSvmRuns5x10percInc.csv", index=False)
print("split end")