import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("omicron200.fasta", "w")
for sequence in SeqIO.parse("omicron100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("omicron100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("omicron100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("omicron100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("omicronKey.fasta", "w")
for sequence in SeqIO.parse("omicron100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
for sequence in SeqIO.parse("omicron100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Omicron"+"\n")
"""