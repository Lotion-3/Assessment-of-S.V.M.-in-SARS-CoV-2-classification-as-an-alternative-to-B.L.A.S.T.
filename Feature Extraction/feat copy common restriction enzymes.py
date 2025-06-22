from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import time
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from itertools import islice # <-- ADD THIS IMPORT

# --- Restriction Enzyme Imports ---
from Bio.Restriction import RestrictionBatch, CommOnly
from Bio.Restriction import EcoRI, HindIII, BamHI

print("Very Start")
# CHOOSE YOUR ENZYME BATCH HERE:
# Option 1: Smaller, custom batch (MUCH FASTER)
#selected_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI])
#enzyme_batch_global = selected_enzymes
# Option 2: Broad set (SLOWER, but as in your original script)
enzyme_batch_global = CommOnly

enzyme_names_global = [str(enz).upper() for enz in enzyme_batch_global]

# CORRECTED PRINT STATEMENT:
first_five_enzyme_names = [str(e) for e in islice(enzyme_batch_global, 5)]
print("Enzyme batch selected:", first_five_enzyme_names, "..." if len(enzyme_batch_global) > 5 else "")


print("Later start")
STANDARD_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DNA_BASES = 'ATGC'

# Generate all possible k-mers for consistent column ordering
def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

print("slightly later")
ALL_DINUCLEOTIDES = generate_kmers(2)
ALL_TRINUCLEOTIDES = generate_kmers(3)

# Global variants dictionary to be populated once
variants_global = {}

# AA mapping
aa_mapping_code_to_name_global = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
    'C': 'Cysteine', 'Q': 'Glutamine', 'E': 'Glutamic Acid', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
}
aa_names_sorted_global = sorted(list(aa_mapping_code_to_name_global.values()))


def calculate_kmer_percentages(sequence_str, k):
    seq_len = len(sequence_str)
    possible_kmers = generate_kmers(k, DNA_BASES)

    if seq_len < k:
        return {kmer: 0.0 for kmer in possible_kmers}

    counts = Counter()
    valid_kmer_starts = 0

    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        if all(base in DNA_BASES for base in kmer):
            counts[kmer] += 1
            valid_kmer_starts += 1

    percentages = {kmer: 0.0 for kmer in possible_kmers}
    if valid_kmer_starts == 0:
        return percentages

    for kmer_seq, count in counts.items():
        percentages[kmer_seq] = (count / valid_kmer_starts) * 100
    return percentages


def process_sequence_record(sequence_record_tuple):
    sequence_record, enzyme_batch_arg, enzyme_names_arg, variants_arg, aa_map_arg, aa_names_sorted_arg = sequence_record_tuple

    seq_id = sequence_record.id
    seq_obj = sequence_record.seq
    seq_len = len(seq_obj)

    row = {'id': seq_id, 'Sequence Length': seq_len}
    
    for col in ['G perc', 'C perc', 'A perc', 'T perc', 'GC perc', 'GC Skew', 'AT Skew',
                'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
                'Protein Instability Index', 'Protein GRAVY',
                'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc',
                'Total_RE_Fragments', 'Avg_RE_Fragment_Length']:
        row[col] = 0.0
    for dn in ALL_DINUCLEOTIDES: row[f"{dn}_perc"] = 0.0
    for tn in ALL_TRINUCLEOTIDES: row[f"{tn}_perc"] = 0.0
    for name in aa_names_sorted_arg: row[f"{name} perc"] = 0.0
    for enz_name in enzyme_names_arg: row[f"{enz_name}_cuts"] = 0

    row['Variant'] = variants_arg.get(seq_id, 'UNKNOWN')

    protein_str_cleaned_exists_and_is_empty = False # For tracking empty protein state

    if seq_len == 0:
        return row, False, protein_str_cleaned_exists_and_is_empty # False for protein_ok, False for protein_empty

    seq_str_upper = str(seq_obj).upper()

    nuc_counts_abs = {nuc: seq_str_upper.count(nuc) for nuc in DNA_BASES}
    total_atgc_bases = sum(nuc_counts_abs.values())

    if total_atgc_bases > 0:
        nuc_percs = {nuc: (nuc_counts_abs[nuc] / total_atgc_bases) * 100
                     for nuc in DNA_BASES}
        row.update({
            'G perc': nuc_percs.get('G', 0.0),
            'C perc': nuc_percs.get('C', 0.0),
            'A perc': nuc_percs.get('A', 0.0),
            'T perc': nuc_percs.get('T', 0.0),
            'GC perc': gc_fraction(seq_obj) * 100
        })

    gc_sum = nuc_counts_abs.get('G', 0) + nuc_counts_abs.get('C', 0)
    at_sum = nuc_counts_abs.get('A', 0) + nuc_counts_abs.get('T', 0)

    row['GC Skew'] = ((nuc_counts_abs.get('G', 0) - nuc_counts_abs.get('C', 0)) / gc_sum) if gc_sum > 0 else 0
    row['AT Skew'] = ((nuc_counts_abs.get('A', 0) - nuc_counts_abs.get('T', 0)) / at_sum) if at_sum > 0 else 0

    dinuc_percs = calculate_kmer_percentages(seq_str_upper, 2)
    trinuc_percs = calculate_kmer_percentages(seq_str_upper, 3)

    for dn, perc in dinuc_percs.items(): row[f"{dn}_perc"] = perc
    for tn, perc in trinuc_percs.items(): row[f"{tn}_perc"] = perc

    protein_analysis_successful = False
    try:
        protein_str_raw = str(seq_obj.translate(to_stop=True))
        protein_str_cleaned = protein_str_raw.replace('X', '').replace('*', '')
        
        if not protein_str_cleaned: # Check if protein string is empty after cleaning
            protein_str_cleaned_exists_and_is_empty = True # Mark that it was empty
        else:
            protein_analyzer = ProteinAnalysis(protein_str_cleaned)
            aa_percent_dict = protein_analyzer.get_amino_acids_percent()

            for code, aa_name in aa_map_arg.items():
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
            protein_analysis_successful = True
    except Exception:
        # protein_error_count +=1 # Cannot increment global like this in worker
        # If an error occurred before protein_str_cleaned was defined or if it was not empty,
        # protein_str_cleaned_exists_and_is_empty remains False.
        # If protein_str_cleaned was defined and empty, it's already True.
        pass 

    analysis = enzyme_batch_arg.search(seq_obj)
    all_cuts_sites_combined = set()

    for i, enz_obj in enumerate(enzyme_batch_arg):
        enz_name = enzyme_names_arg[i]
        cut_sites = analysis.get(enz_obj, [])
        row[f"{enz_name}_cuts"] = len(cut_sites)
        for site in cut_sites:
            all_cuts_sites_combined.add(site)

    sorted_unique_cuts = sorted(list(all_cuts_sites_combined))
    if not sorted_unique_cuts:
        row['Total_RE_Fragments'] = 1 if seq_len > 0 else 0
        row['Avg_RE_Fragment_Length'] = float(seq_len) if seq_len > 0 else 0.0
    else:
        num_fragments = len(sorted_unique_cuts) + 1
        row['Total_RE_Fragments'] = num_fragments
        row['Avg_RE_Fragment_Length'] = float(seq_len) / num_fragments if num_fragments > 0 else 0.0
    
    return row, protein_analysis_successful, protein_str_cleaned_exists_and_is_empty


def extractedDF(splitFile, extractedCsvName, enzyme_batch, enzyme_names, variants, aa_map, aa_names_sorted_list):
    startTime = time.time()

    columns = [
        'id', 'Sequence Length', 'G perc', 'C perc', 'A perc', 'T perc', 'GC perc',
        'GC Skew', 'AT Skew'
    ]
    columns.extend([f"{dn}_perc" for dn in ALL_DINUCLEOTIDES])
    columns.extend([f"{tn}_perc" for tn in ALL_TRINUCLEOTIDES])
    columns.extend([f"{name} perc" for name in aa_names_sorted_list])
    columns.extend([
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ])
    for enz_name in enzyme_names: columns.append(f"{enz_name}_cuts")
    columns.append('Total_RE_Fragments')
    columns.append('Avg_RE_Fragment_Length')
    columns.append('Variant')
    
    sequence_records = list(SeqIO.parse(splitFile, "fasta"))
    tasks = [(record, enzyme_batch, enzyme_names, variants, aa_map, aa_names_sorted_list) for record in sequence_records]

    data_results = []
    protein_error_count = 0 # Counts sequences where protein_analysis_successful is False
    empty_protein_after_cleanup_count = 0 # Counts sequences where protein_str_cleaned_exists_and_is_empty is True
    processed_count = 0

    num_workers = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_workers} worker processes for {splitFile}.")

    with Pool(processes=num_workers) as pool:
        results_from_pool = pool.map(process_sequence_record, tasks)

    for result_tuple in results_from_pool:
        if result_tuple:
            row, protein_ok, protein_empty_flag = result_tuple
            data_results.append(row)
            processed_count +=1
            if not protein_ok :
                protein_error_count +=1 # Increment if analysis failed or was skipped (e.g. due to empty cleaned string)
            if protein_empty_flag: # This specifically flags if the cleaned protein string was empty
                 empty_protein_after_cleanup_count += 1


    df = pd.DataFrame(data_results)
    if not df.empty:
        df = df.reindex(columns=columns, fill_value=0.0)
    else:
        df = pd.DataFrame(columns=columns)

    df.to_csv(extractedCsvName, index=False)

    endTime = time.time()
    print(f"Processed {processed_count} sequences from {splitFile}.")
    if protein_error_count > 0 :
        print(f"Encountered issues during protein analysis for {protein_error_count} sequences (features zeroed).")
    if empty_protein_after_cleanup_count > 0:
         print(f"Translation resulted in an empty protein string (after X/* removal) for {empty_protein_after_cleanup_count} sequences (subset of issues).")
    print(f"Feature extraction for {splitFile} completed in {endTime-startTime:.2f} seconds. Output: {extractedCsvName}")
    return df

# --- Main execution block ---
if __name__ == "__main__":
    print("pre timer")

    try:
        variants_global = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
        variants_global = {}
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")
        variants_global = {}

    print("start")
    try:
        testExtract = extractedDF("test90(11000)(1).fasta", "test90(11000)(1)(extractedALT3).csv",
                                  enzyme_batch_global, enzyme_names_global, variants_global,
                                  aa_mapping_code_to_name_global, aa_names_sorted_global)
        
        trainExtract = extractedDF("train10(11000)(1).fasta", "train10(11000)(1)(extractedALT3).csv", # Output CSV name should be different
                                  enzyme_batch_global, enzyme_names_global, variants_global,
                                  aa_mapping_code_to_name_global, aa_names_sorted_global)
        
        print("Extraction successful for both files.")
        if not testExtract.empty:
            print("Head of testExtract DataFrame:")
            print(testExtract.head())
        if not trainExtract.empty:
            print("Head of trainExtract DataFrame:")
            print(trainExtract.head()) # Careful if this overwrites testExtract due to same output name in your original call
            
    except FileNotFoundError as e:
        print(f"Error: Input FASTA file not found: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred during feature extraction: {e}")
        import traceback
        traceback.print_exc()
    print("end")