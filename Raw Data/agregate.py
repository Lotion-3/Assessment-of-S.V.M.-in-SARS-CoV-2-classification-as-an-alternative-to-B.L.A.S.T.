import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("comb11000.fasta", "w")
print("comb11000 start")
for sequence in SeqIO.parse("alpha1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta1000.fasta", "fasta"):
    comb.write(">"+sequence.id + "$")
    comb.write(str(sequence.seq)+"\n")
print("comb11000 end")