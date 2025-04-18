import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("delta200.fasta", "w")
for sequence in SeqIO.parse("delta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("delta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("delta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("delta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("deltaKey.fasta", "w")
for sequence in SeqIO.parse("delta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
for sequence in SeqIO.parse("delta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Delta"+"\n")
"""