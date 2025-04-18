import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("beta200.fasta", "w")
for sequence in SeqIO.parse("beta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("beta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("beta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("beta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("betaKey.fasta", "w")
for sequence in SeqIO.parse("beta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta" + "\n")
for sequence in SeqIO.parse("beta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
for sequence in SeqIO.parse("beta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Beta"+"\n")
"""