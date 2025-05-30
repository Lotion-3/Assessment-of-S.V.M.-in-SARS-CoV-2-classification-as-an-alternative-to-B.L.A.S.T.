import pandas as pd
import random
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

def randomizeLines(unshuffledFile, shuffledFileName):
    comb= open(unshuffledFile)
    lines = comb.readlines()
    for i in range(1000):
        random.shuffle(lines)
    master= open(shuffledFileName, "w")
    print("shuffled11000 start")
    for line in lines:
        find=True
        i=0
        while find ==True:
            if line[i] == '$':
                master.write(line[0:i]+"\n")
                master.write(line[i+1:])
                find = False   
            else:
                i=i+1
    master.close()
    print("shuffled11000 end")

randomizeLines("comb11000.fasta", "shuffled11000(90-10)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(90-10)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(90-10)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(90-10)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(90-10)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(80-20)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(80-20)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(80-20)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(80-20)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(80-20)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(70-30)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(70-30)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(70-30)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(70-30)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(70-30)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(60-40)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(60-40)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(60-40)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(60-40)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(60-40)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(50-50)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(50-50)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(50-50)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(50-50)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(50-50)(5).fasta")


randomizeLines("comb11000.fasta", "shuffled11000(40-60)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(30-70)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(40-60)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(40-60)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(30-70)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(30-70)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(20-80)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(20-80)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(20-80)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(20-80)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(20-80)(5).fasta")

randomizeLines("comb11000.fasta", "shuffled11000(10-90)(1).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(10-90)(2).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(10-90)(3).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(10-90)(4).fasta")
randomizeLines("comb11000.fasta", "shuffled11000(10-90)(5).fasta")