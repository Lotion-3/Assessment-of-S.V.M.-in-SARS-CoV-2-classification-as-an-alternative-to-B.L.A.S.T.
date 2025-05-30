import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("iota200.fasta", "w")
for sequence in SeqIO.parse("iota100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("iota100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("iota100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("iota100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("iotaKey.fasta", "w")
for sequence in SeqIO.parse("iota100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
for sequence in SeqIO.parse("iota100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Iota"+"\n")
"""