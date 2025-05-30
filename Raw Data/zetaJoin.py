import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

comb= open("zeta200.fasta", "w")
for sequence in SeqIO.parse("zeta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("zeta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
for sequence in SeqIO.parse("zeta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
for sequence in SeqIO.parse("zeta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write(str(sequence.seq)+"\n")
"""
comb= open("zetaKey.fasta", "w")
for sequence in SeqIO.parse("zeta100a.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100b.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100c.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100d.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100e.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100f.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100g.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100h.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100i.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
for sequence in SeqIO.parse("zeta100j.fasta", "fasta"):
    comb.write(">"+sequence.id + "\n")
    comb.write("Zeta"+"\n")
"""