import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction, MeltingTemp as mt # Added MeltingTemp
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import time
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from itertools import islice

# --- Restriction Enzyme Imports ---
from Bio.Restriction import RestrictionBatch, CommOnly
from Bio.Restriction import EcoRI, HindIII, BamHI # Example enzymes for smaller batch

print("Very Start")
# CHOOSE YOUR ENZYME BATCH HERE:
# Option 1: Smaller, custom batch (MUCH FASTER)
selected_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI])
enzyme_batch_global = selected_enzymes
# Option 2: Broad set (SLOWER, but as in your original script)
# enzyme_batch_global = CommOnly

enzyme_names_global = [str(enz).upper() for enz in enzyme_batch_global]

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
ALL_TETRANUCLEOTIDES = generate_kmers(4) # <-- NEW: Tetranucleotides

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

# <-- NEW: Amino Acid Groups -->
AA_HYDROPHOBIC_GLOBAL = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
AA_POLAR_UNCHARGED_GLOBAL = ['S', 'T', 'N', 'Q', 'G', 'P', 'C'] # G,P,C often grouped here or separately
AA_POSITIVE_GLOBAL = ['R', 'H', 'K']
AA_NEGATIVE_GLOBAL = ['D', 'E']


def calculate_kmer_percentages(sequence_str, k):
    seq_len = len(sequence_str)
    # Use pre-generated k-mers based on k
    if k == 2: possible_kmers = ALL_DINUCLEOTIDES
    elif k == 3: possible_kmers = ALL_TRINUCLEOTIDES
    elif k == 4: possible_kmers = ALL_TETRANUCLEOTIDES
    else: # Fallback, though we only use 2, 3, 4
        possible_kmers = generate_kmers(k, DNA_BASES)


    if seq_len < k:
        return {kmer: 0.0 for kmer in possible_kmers}

    counts = Counter()
    valid_kmer_count_total = 0 # This will be the denominator, number of valid k-mers found

    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        # Ensure kmer only contains standard DNA bases before counting
        if all(base in DNA_BASES for base in kmer):
            counts[kmer] += 1
            valid_kmer_count_total += 1

    percentages = {kmer: 0.0 for kmer in possible_kmers} # Initialize all possible k-mers to 0
    if valid_kmer_count_total == 0: # Avoid division by zero if no valid k-mers
        return percentages

    for kmer_seq, count in counts.items():
        if kmer_seq in percentages: # Only update if it's a standard DNA k-mer
            percentages[kmer_seq] = (count / valid_kmer_count_total) * 100
    return percentages


def process_sequence_record(sequence_record_tuple):
    # Unpack arguments
    (sequence_record, enzyme_batch_arg, enzyme_names_arg, variants_arg,
     aa_map_arg, aa_names_sorted_arg,
     aa_hydrophobic_arg, aa_polar_arg, aa_positive_arg, aa_negative_arg) = sequence_record_tuple


    seq_id = sequence_record.id
    seq_obj = sequence_record.seq # This is a Bio.Seq object
    seq_len = len(seq_obj)

    row = {'id': seq_id, 'Sequence Length': seq_len}
    
    # Initialize all potential features to 0.0 or 0
    base_cols = ['G perc', 'C perc', 'A perc', 'T perc', 'GC perc', 'GC Skew', 'AT Skew']
    protein_cols = [
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ]
    re_cols = ['Total_RE_Fragments', 'Avg_RE_Fragment_Length']
    # <-- NEW FEATURES INITIALIZATION -->
    orf_cols = ['Longest_ORF_Protein_Length_3f', 'Num_ORFs_gt30aa_3f'] # 3f for 3 forward frames
    tm_cols = ['Tm_Wallace', 'Tm_GC']
    aa_group_cols = [
        'Protein Hydrophobic AA Perc', 'Protein Polar Uncharged AA Perc',
        'Protein Positively Charged AA Perc', 'Protein Negatively Charged AA Perc'
    ]

    for col in base_cols + protein_cols + re_cols + orf_cols + tm_cols + aa_group_cols:
        row[col] = 0.0 # Default to float for numeric features

    for dn in ALL_DINUCLEOTIDES: row[f"{dn}_perc"] = 0.0
    for tn in ALL_TRINUCLEOTIDES: row[f"{tn}_perc"] = 0.0
    for tetn in ALL_TETRANUCLEOTIDES: row[f"{tetn}_perc"] = 0.0 # <-- NEW
    for name in aa_names_sorted_arg: row[f"{name} perc"] = 0.0
    for enz_name in enzyme_names_arg: row[f"{enz_name}_cuts"] = 0 # Default to int

    row['Variant'] = variants_arg.get(seq_id, 'UNKNOWN')

    protein_str_cleaned_exists_and_is_empty = False

    if seq_len == 0:
        return row, False, protein_str_cleaned_exists_and_is_empty

    seq_str_upper = str(seq_obj).upper() # Work with uppercase string version for counts

    # --- Nucleotide Features ---
    nuc_counts_abs = {nuc: seq_str_upper.count(nuc) for nuc in DNA_BASES}
    total_atgc_bases = sum(nuc_counts_abs.values())

    if total_atgc_bases > 0:
        row.update({
            'G perc': (nuc_counts_abs.get('G', 0) / total_atgc_bases) * 100,
            'C perc': (nuc_counts_abs.get('C', 0) / total_atgc_bases) * 100,
            'A perc': (nuc_counts_abs.get('A', 0) / total_atgc_bases) * 100,
            'T perc': (nuc_counts_abs.get('T', 0) / total_atgc_bases) * 100,
        })
        try:
            row['GC perc'] = gc_fraction(seq_obj) * 100 # gc_fraction handles non-ATGC by ignoring them
        except Exception: # Should be robust, but just in case
            row['GC perc'] = 0.0


    gc_sum = nuc_counts_abs.get('G', 0) + nuc_counts_abs.get('C', 0)
    at_sum = nuc_counts_abs.get('A', 0) + nuc_counts_abs.get('T', 0)

    row['GC Skew'] = ((nuc_counts_abs.get('G', 0) - nuc_counts_abs.get('C', 0)) / gc_sum) if gc_sum > 0 else 0.0
    row['AT Skew'] = ((nuc_counts_abs.get('A', 0) - nuc_counts_abs.get('T', 0)) / at_sum) if at_sum > 0 else 0.0

    # k-mer percentages
    dinuc_percs = calculate_kmer_percentages(seq_str_upper, 2)
    trinuc_percs = calculate_kmer_percentages(seq_str_upper, 3)
    tetranuc_percs = calculate_kmer_percentages(seq_str_upper, 4) # <-- NEW

    for dn, perc in dinuc_percs.items(): row[f"{dn}_perc"] = perc
    for tn, perc in trinuc_percs.items(): row[f"{tn}_perc"] = perc
    for tetn, perc in tetranuc_percs.items(): row[f"{tetn}_perc"] = perc # <-- NEW

    # --- Melting Temperature Features ---
    try:
        if seq_len > 0 and all(c in "ATGC" for c in seq_str_upper if c != 'N'): # Tm functions might be sensitive to non-ATGC
            strict_seq_obj_for_tm = Seq("".join(c for c in seq_str_upper if c in "ATGC"))
            if len(strict_seq_obj_for_tm) > 0: # Ensure not empty after stripping
                row['Tm_Wallace'] = mt.Tm_Wallace(strict_seq_obj_for_tm)
                row['Tm_GC'] = mt.Tm_GC(strict_seq_obj_for_tm, strict=False) # strict=False to handle Ns if any remained (they shouldn't here)
    except Exception: # Catch any errors from Tm calculations
        pass # Keep defaults (0.0)

    # --- Simplified ORF Features (3 forward frames) ---
    min_protein_len_aa = 30
    num_orfs_gt_min = 0
    longest_orf_prot_len = 0
    if seq_len >= min_protein_len_aa * 3 : # Only process if potentially long enough
        for frame in range(3):
            try:
                # Translate from this frame, stopping at the first stop codon
                # This gives one "dominant" protein product per frame
                translated_seq_str = str(seq_obj[frame:].translate(table="Standard", to_stop=True))
                current_prot_len = len(translated_seq_str)
                if current_prot_len >= min_protein_len_aa:
                    num_orfs_gt_min += 1
                if current_prot_len > longest_orf_prot_len:
                    longest_orf_prot_len = current_prot_len
            except Exception: # In case of translation errors (e.g. unknown codons before stop)
                pass
    row['Longest_ORF_Protein_Length_3f'] = float(longest_orf_prot_len)
    row['Num_ORFs_gt30aa_3f'] = float(num_orfs_gt_min)


    # --- Protein Features ---
    protein_analysis_successful = False
    try:
        protein_str_raw = str(seq_obj.translate(table="Standard", to_stop=True)) # Translate from frame 0
        protein_str_cleaned = protein_str_raw.replace('X', '').replace('*', '')
        
        if not protein_str_cleaned:
            protein_str_cleaned_exists_and_is_empty = True
        else:
            protein_analyzer = ProteinAnalysis(protein_str_cleaned)
            aa_percent_dict = protein_analyzer.get_amino_acids_percent()

            for code, aa_name in aa_map_arg.items():
                row[f"{aa_name} perc"] = aa_percent_dict.get(code, 0.0) * 100

            # <-- NEW: Amino Acid Group Percentages -->
            row['Protein Hydrophobic AA Perc'] = sum(aa_percent_dict.get(aa, 0.0) for aa in aa_hydrophobic_arg) * 100
            row['Protein Polar Uncharged AA Perc'] = sum(aa_percent_dict.get(aa, 0.0) for aa in aa_polar_arg) * 100
            row['Protein Positively Charged AA Perc'] = sum(aa_percent_dict.get(aa, 0.0) for aa in aa_positive_arg) * 100
            row['Protein Negatively Charged AA Perc'] = sum(aa_percent_dict.get(aa, 0.0) for aa in aa_negative_arg) * 100

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
        pass 

    # --- Restriction Enzyme Features ---
    analysis = enzyme_batch_arg.search(seq_obj) # Use original seq_obj for RE
    all_cuts_sites_combined = set()

    for i, enz_obj in enumerate(enzyme_batch_arg):
        enz_name = enzyme_names_arg[i]
        cut_sites = analysis.get(enz_obj, [])
        row[f"{enz_name}_cuts"] = len(cut_sites)
        for site in cut_sites: # site is an integer position
            all_cuts_sites_combined.add(site)

    sorted_unique_cuts = sorted(list(all_cuts_sites_combined))
    if not sorted_unique_cuts:
        row['Total_RE_Fragments'] = 1.0 if seq_len > 0 else 0.0
        row['Avg_RE_Fragment_Length'] = float(seq_len) if seq_len > 0 else 0.0
    else:
        num_fragments = len(sorted_unique_cuts) + 1
        row['Total_RE_Fragments'] = float(num_fragments)
        row['Avg_RE_Fragment_Length'] = float(seq_len) / num_fragments if num_fragments > 0 else 0.0
    
    return row, protein_analysis_successful, protein_str_cleaned_exists_and_is_empty


def extractedDF(splitFile, extractedCsvName, enzyme_batch, enzyme_names, variants, 
                aa_map, aa_names_sorted_list,
                aa_hydrophobic, aa_polar, aa_positive, aa_negative): # <-- Add new AA group args
    startTime = time.time()

    # Define all column names for the DataFrame for consistent order
    columns = [
        'id', 'Sequence Length', 'G perc', 'C perc', 'A perc', 'T perc', 'GC perc',
        'GC Skew', 'AT Skew'
    ]
    # K-mers
    columns.extend([f"{dn}_perc" for dn in ALL_DINUCLEOTIDES])
    columns.extend([f"{tn}_perc" for tn in ALL_TRINUCLEOTIDES])
    columns.extend([f"{tetn}_perc" for tetn in ALL_TETRANUCLEOTIDES]) # <-- NEW
    # Amino Acids
    columns.extend([f"{name} perc" for name in aa_names_sorted_list])
    # Protein Properties
    columns.extend([
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ])
    # <-- NEW FEATURES COLUMNS -->
    columns.extend(['Tm_Wallace', 'Tm_GC'])
    columns.extend(['Longest_ORF_Protein_Length_3f', 'Num_ORFs_gt30aa_3f'])
    columns.extend([
        'Protein Hydrophobic AA Perc', 'Protein Polar Uncharged AA Perc',
        'Protein Positively Charged AA Perc', 'Protein Negatively Charged AA Perc'
    ])
    # Restriction Enzymes
    for enz_name in enzyme_names: columns.append(f"{enz_name}_cuts")
    columns.append('Total_RE_Fragments')
    columns.append('Avg_RE_Fragment_Length')
    # Variant
    columns.append('Variant')
    
    try:
        sequence_records = list(SeqIO.parse(splitFile, "fasta"))
        if not sequence_records:
            print(f"No sequences found in {splitFile}. Output CSV will be empty.")
            df = pd.DataFrame(columns=columns)
            df.to_csv(extractedCsvName, index=False)
            return df
    except FileNotFoundError:
        print(f"Error: {splitFile} not found. Cannot extract features.")
        return pd.DataFrame(columns=columns) # Return empty DF with correct columns
    except Exception as e:
        print(f"Error reading {splitFile}: {e}. Cannot extract features.")
        return pd.DataFrame(columns=columns)

    # Prepare tasks for multiprocessing, including new AA group lists
    tasks = [(record, enzyme_batch, enzyme_names, variants, aa_map, aa_names_sorted_list,
              aa_hydrophobic, aa_polar, aa_positive, aa_negative) # Pass new args
             for record in sequence_records]

    data_results = []
    protein_error_count = 0
    empty_protein_after_cleanup_count = 0
    processed_count = 0

    num_workers = max(1, int(cpu_count() * 0.8)) # Or cpu_count() -1 or similar
    print(f"Using {num_workers} worker processes for {splitFile}.")

    with Pool(processes=num_workers) as pool:
        # pool.map can be slow for very large iterables of tasks if chunking isn't optimal
        # For very many short tasks, consider chunksize or imap_unordered
        results_from_pool = pool.map(process_sequence_record, tasks, chunksize=max(1, len(tasks) // (num_workers * 4) ))


    for result_tuple in results_from_pool:
        if result_tuple: # Should always be true unless process_sequence_record returns None
            row, protein_ok, protein_empty_flag = result_tuple
            data_results.append(row)
            processed_count +=1
            if not protein_ok :
                protein_error_count +=1
            if protein_empty_flag:
                 empty_protein_after_cleanup_count += 1


    df = pd.DataFrame(data_results)
    # Ensure DataFrame has all columns in the defined order, filling missing with 0.0 or appropriate NA
    if not df.empty:
        df = df.reindex(columns=columns, fill_value=0.0) # Ensure all columns present
    else: # If data_results was empty (e.g. no sequences processed)
        df = pd.DataFrame(columns=columns)


    df.to_csv(extractedCsvName, index=False)

    endTime = time.time()
    print(f"Processed {processed_count} sequences from {splitFile}.")
    if protein_error_count > 0 :
        print(f"Encountered issues during protein analysis for {protein_error_count} sequences (features zeroed or defaulted).")
    if empty_protein_after_cleanup_count > 0:
         print(f"Translation resulted in an empty protein string (after X/* removal) for {empty_protein_after_cleanup_count} sequences (protein features defaulted).")
    print(f"Feature extraction for {splitFile} completed in {endTime-startTime:.2f} seconds. Output: {extractedCsvName}")
    return df

# --- Main execution block ---
if __name__ == "__main__":
    print("pre timer")

    try:
        # Load variants only once
        variants_global = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
        print(f"Loaded {len(variants_global)} variants from key.fasta")
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
        variants_global = {}
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")
        variants_global = {}

    print("start")
    fasta_file_test = "test90(11000)(1).fasta" # Make sure this file exists
    csv_output_test = "test90(11000)(1)(extractedALT5).csv"
    
    fasta_file_train = "train10(11000)(1).fasta" # Make sure this file exists
    csv_output_train = "train10(11000)(1)(extractedALT5).csv"

    try:
        print(f"\n--- Extracting features for {fasta_file_test} ---")
        testExtract = extractedDF(fasta_file_test, csv_output_test,
                                  enzyme_batch_global, enzyme_names_global, variants_global,
                                  aa_mapping_code_to_name_global, aa_names_sorted_global,
                                  AA_HYDROPHOBIC_GLOBAL, AA_POLAR_UNCHARGED_GLOBAL, # Pass AA groups
                                  AA_POSITIVE_GLOBAL, AA_NEGATIVE_GLOBAL)
        
        print(f"\n--- Extracting features for {fasta_file_train} ---")
        trainExtract = extractedDF(fasta_file_train, csv_output_train,
                                   enzyme_batch_global, enzyme_names_global, variants_global,
                                   aa_mapping_code_to_name_global, aa_names_sorted_global,
                                   AA_HYDROPHOBIC_GLOBAL, AA_POLAR_UNCHARGED_GLOBAL, # Pass AA groups
                                   AA_POSITIVE_GLOBAL, AA_NEGATIVE_GLOBAL)
        
        print("\nExtraction successful for both files (if they existed and contained sequences).")
        if not testExtract.empty:
            print(f"\nHead of testExtract DataFrame ({csv_output_test}):")
            print(testExtract.head())
        else:
            print(f"\nTestExtract DataFrame ({csv_output_test}) is empty.")

        if not trainExtract.empty:
            print(f"\nHead of trainExtract DataFrame ({csv_output_train}):")
            print(trainExtract.head())
        else:
            print(f"\nTrainExtract DataFrame ({csv_output_train}) is empty.")
            
    except Exception as e:
        print(f"An unexpected error occurred during the main execution: {e}")
        import traceback
        traceback.print_exc()
    print("end")