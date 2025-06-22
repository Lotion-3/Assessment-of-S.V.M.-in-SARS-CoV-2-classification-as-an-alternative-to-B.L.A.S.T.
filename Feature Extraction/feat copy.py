from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import timeit
from itertools import product
import itertools # For generating k-mer combinations
from collections import Counter
#Feature Extraction
import time
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from itertools import product

import time
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
import itertools

# Define standard amino acids and DNA bases globally
STANDARD_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DNA_BASES = 'ATGC'

# Generate all possible k-mers for consistent column ordering
def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

ALL_DINUCLEOTIDES = generate_kmers(2)
ALL_TRINUCLEOTIDES = generate_kmers(3)

def calculate_kmer_percentages(sequence_str, k):
    """Calculates k-mer percentages for a given sequence string, considering only valid DNA k-mers."""
    seq_len = len(sequence_str)
    # Ensure k-mers are generated with the correct alphabet
    possible_kmers = generate_kmers(k, DNA_BASES)
    
    if seq_len < k: # Sequence too short for any k-mer
        return {kmer: 0.0 for kmer in possible_kmers}

    counts = Counter()
    valid_kmer_starts = 0

    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        # Ensure k-mer contains only standard DNA bases before counting
        if all(base in DNA_BASES for base in kmer):
            counts[kmer] += 1
            valid_kmer_starts += 1
    
    # Initialize all k-mer percentages to 0.0
    percentages = {kmer: 0.0 for kmer in possible_kmers}
    if valid_kmer_starts == 0: # Avoid division by zero if no valid k-mers found
        return percentages

    for kmer_seq, count in counts.items():
        # kmer_seq should already be in possible_kmers if it passed the 'all(base in DNA_BASES)' check
        percentages[kmer_seq] = (count / valid_kmer_starts) * 100
    return percentages


def extractedDF(splitFile, extractedCsvName):
    startTime = time.time()

    try:
        variants = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
        variants = {}
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")
        variants = {}

    columns = [
        'id', 'Sequence Length', 'G perc', 'C perc', 'A perc', 'T perc', 'GC perc',
        'GC Skew', 'AT Skew'
    ]
    columns.extend([f"{dn}_perc" for dn in ALL_DINUCLEOTIDES])
    columns.extend([f"{tn}_perc" for tn in ALL_TRINUCLEOTIDES])
    aa_names_sorted = sorted([
        'Alanine', 'Arginine', 'Asparagine', 'Aspartic Acid', 'Cysteine',
        'Glutamine', 'Glutamic Acid', 'Glycine', 'Histidine', 'Isoleucine',
        'Leucine', 'Lysine', 'Methionine', 'Phenylalanine', 'Proline',
        'Serine', 'Threonine', 'Tryptophan', 'Tyrosine', 'Valine'
    ])
    columns.extend([f"{name} perc" for name in aa_names_sorted])
    columns.extend([
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ])
    columns.append('Variant')

    data = []
    aa_mapping_code_to_name = {
        'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
        'C': 'Cysteine', 'Q': 'Glutamine', 'E': 'Glutamic Acid', 'G': 'Glycine',
        'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
        'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
        'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
    }

    processed_count = 0
    protein_error_count = 0
    empty_protein_after_cleanup_count = 0

    for sequence_record in SeqIO.parse(splitFile, "fasta"):
        seq_id = sequence_record.id
        seq_obj = sequence_record.seq # This is a Bio.Seq.Seq object
        seq_len = len(seq_obj)

        if seq_len == 0:
            print(f"Warning: Sequence {seq_id} is empty. Skipping.")
            continue
        
        # Work with uppercase string for nucleotide analysis
        seq_str_upper = str(seq_obj).upper()

        # --- Nucleotide Features ---
        nuc_counts_abs = {nuc: seq_str_upper.count(nuc) for nuc in DNA_BASES}
        
        # Calculate total valid bases (A, T, G, C) for accurate percentage
        # This prevents 'N's from skewing percentages if seq_len is used as denominator
        total_atgc_bases = sum(nuc_counts_abs.values())

        if total_atgc_bases == 0: # Sequence might be all 'N's or other non-ATGC
            nuc_percs = {nuc: 0.0 for nuc in DNA_BASES}
            # gc_fraction will return 0 if no G or C, or if only Ns.
            # For consistency, if total_atgc_bases is 0, gc_perc should be 0.
            gc_perc_val = 0.0
        else:
            nuc_percs = {nuc: (nuc_counts_abs[nuc] / total_atgc_bases) * 100
                         for nuc in DNA_BASES}
            # gc_fraction correctly handles non-ATGC characters by ignoring them
            gc_perc_val = gc_fraction(seq_obj) * 100


        gc_sum = nuc_counts_abs['G'] + nuc_counts_abs['C']
        at_sum = nuc_counts_abs['A'] + nuc_counts_abs['T']

        gc_skew = ((nuc_counts_abs['G'] - nuc_counts_abs['C']) / gc_sum) if gc_sum > 0 else 0
        at_skew = ((nuc_counts_abs['A'] - nuc_counts_abs['T']) / at_sum) if at_sum > 0 else 0

        # Pass the uppercase string which might contain 'N's.
        # The kmer function is updated to handle this by only counting ATGC k-mers.
        dinuc_percs = calculate_kmer_percentages(seq_str_upper, 2)
        trinuc_percs = calculate_kmer_percentages(seq_str_upper, 3)

        row = {
            'id': seq_id,
            'Sequence Length': seq_len,
            'G perc': nuc_percs.get('G', 0.0),
            'C perc': nuc_percs.get('C', 0.0),
            'A perc': nuc_percs.get('A', 0.0),
            'T perc': nuc_percs.get('T', 0.0),
            'GC perc': gc_perc_val,
            'GC Skew': gc_skew,
            'AT Skew': at_skew,
            'Variant': variants.get(seq_id, 'UNKNOWN')
        }

        for dn, perc in dinuc_percs.items():
            row[f"{dn}_perc"] = perc
        for tn, perc in trinuc_percs.items():
            row[f"{tn}_perc"] = perc

        # Initialize protein features to 0.0
        for name in aa_names_sorted:
            row[f"{name} perc"] = 0.0
        row.update({
            'Protein Molecular Weight': 0.0, 'Protein Isoelectric Point': 0.0,
            'Protein Aromaticity': 0.0, 'Protein Instability Index': 0.0,
            'Protein GRAVY': 0.0, 'Protein Helix Perc': 0.0,
            'Protein Turn Perc': 0.0, 'Protein Sheet Perc': 0.0
        })

        try:
            # Translate the Bio.Seq.Seq object
            # to_stop=True: translates until the first stop codon
            translated_seq_obj = seq_obj.translate(to_stop=True)
            protein_str_raw = str(translated_seq_obj)

            # CRITICAL FIX: Remove 'X' (ambiguous) and '*' (stop, though to_stop should handle)
            protein_str_cleaned = protein_str_raw.replace('X', '').replace('*', '')

            if not protein_str_cleaned: # If string is empty after cleaning
                # This means original translation was all 'X', '*', or very short and became empty.
                # No warning printed here for brevity, summary at the end.
                empty_protein_after_cleanup_count +=1
            else:
                protein_analyzer = ProteinAnalysis(protein_str_cleaned)
                aa_percent_dict = protein_analyzer.get_amino_acids_percent()

                for code, aa_name in aa_mapping_code_to_name.items():
                    row[f"{aa_name} perc"] = aa_percent_dict.get(code, 0.0) * 100

                row['Protein Molecular Weight'] = protein_analyzer.molecular_weight()
                row['Protein Isoelectric Point'] = protein_analyzer.isoelectric_point()
                row['Protein Aromaticity'] = protein_analyzer.aromaticity()
                row['Protein Instability Index'] = protein_analyzer.instability_index()
                row['Protein GRAVY'] = protein_analyzer.gravy()
                
                helix, turn, sheet = protein_analyzer.secondary_structure_fraction()
                row['Protein Helix Perc'] = helix * 100
                row['Protein Turn Perc'] = turn * 100
                row['Protein Sheet Perc'] = sheet * 100
        
        except Exception as e: # Catch any other unexpected error during protein analysis
            # This might catch issues if translate itself fails for some very odd sequences,
            # or if ProteinAnalysis fails for reasons other than 'X' (unlikely with standard AA).
            # print(f"Error processing protein features for {seq_id}: {str(e)}. Protein features will be zero.")
            protein_error_count +=1
        
        data.append(row)
        processed_count += 1

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(extractedCsvName, index=False)

    endTime = time.time()
    print(f"Processed {processed_count} sequences.")
    if protein_error_count > 0 :
        print(f"Encountered unexpected errors during protein analysis for {protein_error_count} sequences. Their protein features are zeroed.")
    if empty_protein_after_cleanup_count > 0:
        print(f"Translation (after X/* removal) resulted in empty protein for {empty_protein_after_cleanup_count} sequences. Their protein features are zeroed.")
    print(f"Feature extraction completed in {endTime-startTime:.2f} seconds. Output: {extractedCsvName}")
    return df

# --- Example Usage (assuming you have 'key.fasta' and an input 'sequences.fasta') ---
"""if __name__ == '__main__':
    # Create dummy files for testing
    with open("key.fasta", "w") as f:
        f.write(">seq1_variantA\nATGC\n")
        f.write(">seq2_variantB\nCGTA\n")

    with open("input_sequences.fasta", "w") as f:
        f.write(">seq1_variantA\nATGCATGCATGCATGCATGC\n") # Should translate to M H M H M
        f.write(">seq2_variantB\nCGTACGTACGTACGTACGTA\n") # R T R T R
        f.write(">seq3_unknown\nAGCTAGCTAGCTAGCTAGCT\n") # S Stop S Stop S (translate(to_stop=True) will give 'S')
        f.write(">seq4_short\nAT\n") # Too short for some protein analysis, will warn
        f.write(">seq5_noncoding_like\nNNNNNNNNNNNNNNNNNNNN\n") # Will have 0 for most things

    # Run the extraction
    output_csv_name = "extracted_features_plus.csv"
    extracted_df = extractedDF("input_sequences.fasta", output_csv_name)
    print(f"\nDataFrame head:\n{extracted_df.head()}")
    print(f"\nDataFrame columns:\n{extracted_df.columns.tolist()}")
"""
def timer(operation):
    start_time = time.time()
    operation
    end_time = time.time()
    timeTaken = (end_time-start_time)
    print(start_time)
    print(end_time)
    print(timeTaken)
    return timeTaken

def counting(num):
    print(num)

print("start")
testExtract = extractedDF("test90(11000)(1).fasta", "test90(11000)(1)(extractedALT).csv")
testExtract = extractedDF("train10(11000)(1).fasta", "train10(11000)(1)(extractedALT).csv")
print("end")