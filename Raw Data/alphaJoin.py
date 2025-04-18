import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("alpha200.fasta", "w")
for sequence in SeqIO.parse("alpha100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("alpha100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("alpha100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("alpha100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""

comb= open("alphaKey.fasta", "w")
for sequence in SeqIO.parse("alpha100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
for sequence in SeqIO.parse("alpha100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Alpha"+"\n")
"""