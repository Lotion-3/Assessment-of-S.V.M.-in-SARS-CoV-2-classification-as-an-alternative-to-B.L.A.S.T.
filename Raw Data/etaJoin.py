import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("eta200.fasta", "w")
for sequence in SeqIO.parse("eta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("eta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("eta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("eta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("etaKey.fasta", "w")
for sequence in SeqIO.parse("eta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
for sequence in SeqIO.parse("eta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Eta"+"\n")
"""