import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("lambda200.fasta", "w")
for sequence in SeqIO.parse("lambda100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("lambda100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("lambda100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("lambda100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("lambdaKey.fasta", "w")
for sequence in SeqIO.parse("lambda100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
for sequence in SeqIO.parse("lambda100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Lambda"+"\n")
"""