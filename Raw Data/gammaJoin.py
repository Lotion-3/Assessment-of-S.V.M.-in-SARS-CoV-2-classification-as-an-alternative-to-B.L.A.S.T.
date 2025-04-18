import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("gamma200.fasta", "w")
for sequence in SeqIO.parse("gamma100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("gamma100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("gamma100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("gamma100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("gammaKey.fasta", "w")
for sequence in SeqIO.parse("gamma100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
for sequence in SeqIO.parse("gamma100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Gamma"+"\n")
"""