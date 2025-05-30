import pandas as pd
from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO

def createDF(file):
    df = pd.read_csv(file)
    return df.iloc[:, 0:2]

blastResultsDF = createDF("blastResults(11000).csv")
#print(blastResultsDF)
amonutCorrect = 0
total = 0
for index, row in blastResultsDF.iterrows():
    print(index, row['seq1'], row['seq2'])
    seq1Key = ''
    seq2Key = ''
    for sequence in SeqIO.parse('key.fasta', 'fasta'):
        if row['seq1'] == sequence.id:
            seq1Key= str(sequence.seq)
        if row['seq2'] == sequence.id:
            seq2Key= str(sequence.seq)
    if seq1Key == seq2Key:
        amonutCorrect = amonutCorrect+1
    total = total+1
    print(amonutCorrect)
    print(total)

print(amonutCorrect/total)