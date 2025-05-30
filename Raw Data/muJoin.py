import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("mu200.fasta", "w")
for sequence in SeqIO.parse("mu100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("mu100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("mu100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("mu100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("muKey.fasta", "w")
for sequence in SeqIO.parse("mu100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
for sequence in SeqIO.parse("mu100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Mu"+"\n")
"""