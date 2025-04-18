import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("epsilon200.fasta", "w")
for sequence in SeqIO.parse("epsilon100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("epsilon100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("epsilon100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("epsilon100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("epsilonKey.fasta", "w")
for sequence in SeqIO.parse("epsilon100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
for sequence in SeqIO.parse("epsilon100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Epsilon"+"\n")
"""