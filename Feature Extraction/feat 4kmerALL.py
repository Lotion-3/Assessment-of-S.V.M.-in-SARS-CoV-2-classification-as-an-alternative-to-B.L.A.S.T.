import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO, BiopythonWarning
from Bio.SeqUtils import gc_fraction, MeltingTemp as mt
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import time
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from itertools import islice
import warnings
import os # For checking file existence
import csv # For writing the timing log

# --- Restriction Enzyme Imports ---
from Bio.Restriction import RestrictionBatch, CommOnly
from Bio.Restriction import EcoRI, HindIII, BamHI # Example enzymes

# Suppress specific Biopython warnings that are understood in this context
warnings.filterwarnings("ignore", category=BiopythonWarning, message="Partial codon.*")

print("Very Start")

DNA_BASES = 'ATGC'  # Define DNA_BASES globally for generate_kmers

# Generate all possible k-mers for consistent column ordering
def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

ALL_DINUCLEOTIDES = generate_kmers(2)
ALL_TRINUCLEOTIDES = generate_kmers(3)
ALL_TETRANUCLEOTIDES = generate_kmers(4)

variants_global = {}
aa_mapping_code_to_name_global = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
    'C': 'Cysteine', 'Q': 'Glutamine', 'E': 'Glutamic Acid', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
}
aa_names_sorted_global = sorted(list(aa_mapping_code_to_name_global.values()))
AA_HYDROPHOBIC_GLOBAL = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
AA_POLAR_UNCHARGED_GLOBAL = ['S', 'T', 'N', 'Q', 'G', 'P', 'C']
AA_POSITIVE_GLOBAL = ['R', 'H', 'K']
AA_NEGATIVE_GLOBAL = ['D', 'E']


def calculate_kmer_percentages(sequence_str, k):
    seq_len = len(sequence_str)
    if k == 2: possible_kmers = ALL_DINUCLEOTIDES
    elif k == 3: possible_kmers = ALL_TRINUCLEOTIDES
    elif k == 4: possible_kmers = ALL_TETRANUCLEOTIDES
    else: # Should not be reached with current usage
        possible_kmers = generate_kmers(k, DNA_BASES)

    if seq_len < k:
        return {kmer: 0.0 for kmer in possible_kmers}

    counts = Counter()
    valid_kmer_count_total = 0
    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        if all(base in DNA_BASES for base in kmer): # Ensure kmer is composed of valid DNA bases
            counts[kmer] += 1
            valid_kmer_count_total += 1

    percentages = {kmer: 0.0 for kmer in possible_kmers} # Initialize all with 0
    if valid_kmer_count_total == 0: # Avoid division by zero
        return percentages

    for kmer_seq, count in counts.items():
        if kmer_seq in percentages: # Safety check
            percentages[kmer_seq] = (count / valid_kmer_count_total) * 100
    return percentages


def process_sequence_record(sequence_record_tuple):
    (sequence_record, enzyme_batch_arg, enzyme_names_arg, variants_arg,
     aa_map_arg, aa_names_sorted_arg,
     aa_hydrophobic_arg, aa_polar_arg, aa_positive_arg, aa_negative_arg) = sequence_record_tuple

    seq_id = sequence_record.id
    seq_obj = sequence_record.seq
    seq_len = len(seq_obj)

    row = {'id': seq_id, 'Sequence Length': seq_len}
    
    base_cols = ['G perc', 'C perc', 'A perc', 'T perc', 'GC perc', 'GC Skew', 'AT Skew']
    protein_cols = [
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ]
    re_cols = ['Total_RE_Fragments', 'Avg_RE_Fragment_Length']
    orf_cols = ['Longest_ORF_Protein_Length_3f', 'Num_ORFs_gt30aa_3f']
    tm_cols = ['Tm_Wallace', 'Tm_GC']
    aa_group_cols = [
        'Protein Hydrophobic AA Perc', 'Protein Polar Uncharged AA Perc',
        'Protein Positively Charged AA Perc', 'Protein Negatively Charged AA Perc'
    ]

    for col in base_cols + protein_cols + re_cols + orf_cols + tm_cols + aa_group_cols:
        row[col] = 0.0
    for dn in ALL_DINUCLEOTIDES: row[f"{dn}_perc"] = 0.0
    for tn in ALL_TRINUCLEOTIDES: row[f"{tn}_perc"] = 0.0
    for tetn in ALL_TETRANUCLEOTIDES: row[f"{tetn}_perc"] = 0.0
    for name in aa_names_sorted_arg: row[f"{name} perc"] = 0.0
    for enz_name in enzyme_names_arg: row[f"{enz_name}_cuts"] = 0

    row['Variant'] = variants_arg.get(seq_id, 'UNKNOWN')
    protein_str_cleaned_exists_and_is_empty = False

    if seq_len == 0:
        return row, False, protein_str_cleaned_exists_and_is_empty

    seq_str_upper = str(seq_obj).upper()
    
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
            row['GC perc'] = gc_fraction(seq_obj) * 100
        except Exception:
            row['GC perc'] = 0.0

    gc_sum = nuc_counts_abs.get('G', 0) + nuc_counts_abs.get('C', 0)
    at_sum = nuc_counts_abs.get('A', 0) + nuc_counts_abs.get('T', 0)
    row['GC Skew'] = ((nuc_counts_abs.get('G', 0) - nuc_counts_abs.get('C', 0)) / gc_sum) if gc_sum > 0 else 0.0
    row['AT Skew'] = ((nuc_counts_abs.get('A', 0) - nuc_counts_abs.get('T', 0)) / at_sum) if at_sum > 0 else 0.0

    dinuc_percs = calculate_kmer_percentages(seq_str_upper, 2)
    trinuc_percs = calculate_kmer_percentages(seq_str_upper, 3)
    tetranuc_percs = calculate_kmer_percentages(seq_str_upper, 4)
    for dn, perc in dinuc_percs.items(): row[f"{dn}_perc"] = perc
    for tn, perc in trinuc_percs.items(): row[f"{tn}_perc"] = perc
    for tetn, perc in tetranuc_percs.items(): row[f"{tetn}_perc"] = perc

    try:
        if seq_len > 0:
            strict_seq_str_for_tm = "".join(c for c in seq_str_upper if c in "ATGC")
            if strict_seq_str_for_tm:
                strict_seq_obj_for_tm = Seq(strict_seq_str_for_tm)
                row['Tm_Wallace'] = mt.Tm_Wallace(strict_seq_obj_for_tm)
                row['Tm_GC'] = mt.Tm_GC(strict_seq_obj_for_tm, strict=False)
    except Exception:
        pass

    min_protein_len_aa = 30
    num_orfs_gt_min = 0
    longest_orf_prot_len = 0
    if seq_len >= min_protein_len_aa * 3 :
        for frame in range(3):
            try:
                translated_seq_str = str(seq_obj[frame:].translate(table="Standard", to_stop=True))
                current_prot_len = len(translated_seq_str)
                if current_prot_len >= min_protein_len_aa:
                    num_orfs_gt_min += 1
                if current_prot_len > longest_orf_prot_len:
                    longest_orf_prot_len = current_prot_len
            except Exception:
                pass
    row['Longest_ORF_Protein_Length_3f'] = float(longest_orf_prot_len)
    row['Num_ORFs_gt30aa_3f'] = float(num_orfs_gt_min)

    protein_analysis_successful = False
    try:
        protein_str_raw = str(seq_obj.translate(table="Standard", to_stop=True))
        protein_str_cleaned = protein_str_raw.replace('X', '').replace('*', '')
        
        if not protein_str_cleaned:
            protein_str_cleaned_exists_and_is_empty = True
        else:
            protein_analyzer = ProteinAnalysis(protein_str_cleaned)
            aa_percent_dict = protein_analyzer.amino_acids_percent

            for code, aa_name in aa_map_arg.items():
                row[f"{aa_name} perc"] = aa_percent_dict.get(code, 0.0) * 100

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
        row['Total_RE_Fragments'] = 1.0 if seq_len > 0 else 0.0
        row['Avg_RE_Fragment_Length'] = float(seq_len) if seq_len > 0 else 0.0
    else:
        num_fragments = len(sorted_unique_cuts) + 1
        row['Total_RE_Fragments'] = float(num_fragments)
        row['Avg_RE_Fragment_Length'] = float(seq_len) / num_fragments if num_fragments > 0 else 0.0
    
    return row, protein_analysis_successful, protein_str_cleaned_exists_and_is_empty


def extractedDF(splitFile, extractedCsvName, enzyme_batch, enzyme_names, variants, 
                aa_map, aa_names_sorted_list,
                aa_hydrophobic, aa_polar, aa_positive, aa_negative):
    # This function's internal startTime/endTime is for THIS file, not the whole batch.
    # The return value is extended to include the duration.
    file_processing_startTime = time.time()
    
    columns = [
        'id', 'Sequence Length', 'G perc', 'C perc', 'A perc', 'T perc', 'GC perc',
        'GC Skew', 'AT Skew'
    ]
    columns.extend([f"{dn}_perc" for dn in ALL_DINUCLEOTIDES])
    columns.extend([f"{tn}_perc" for tn in ALL_TRINUCLEOTIDES])
    columns.extend([f"{tetn}_perc" for tetn in ALL_TETRANUCLEOTIDES])
    columns.extend([f"{name} perc" for name in aa_names_sorted_list])
    columns.extend([
        'Protein Molecular Weight', 'Protein Isoelectric Point', 'Protein Aromaticity',
        'Protein Instability Index', 'Protein GRAVY',
        'Protein Helix Perc', 'Protein Turn Perc', 'Protein Sheet Perc'
    ])
    columns.extend(['Tm_Wallace', 'Tm_GC'])
    columns.extend(['Longest_ORF_Protein_Length_3f', 'Num_ORFs_gt30aa_3f'])
    columns.extend([
        'Protein Hydrophobic AA Perc', 'Protein Polar Uncharged AA Perc',
        'Protein Positively Charged AA Perc', 'Protein Negatively Charged AA Perc'
    ])
    for enz_name in enzyme_names: columns.append(f"{enz_name}_cuts")
    columns.append('Total_RE_Fragments')
    columns.append('Avg_RE_Fragment_Length')
    columns.append('Variant')
    
    df_status = "Skipped (File not found)"
    processed_count_this_file = 0
    df_to_return = pd.DataFrame(columns=columns) # Default to empty DF

    try:
        if not os.path.exists(splitFile):
            print(f"Error: Input file {splitFile} not found. Skipping.")
            # df_status remains "Skipped (File not found)"
            file_processing_endTime = time.time()
            duration = file_processing_endTime - file_processing_startTime
            return df_to_return, duration, processed_count_this_file, df_status

        sequence_records = list(SeqIO.parse(splitFile, "fasta"))
        if not sequence_records:
            print(f"No sequences found in {splitFile}. Output CSV will be empty.")
            df_status = "Processed (No sequences)"
            # df_to_return is already an empty DF with correct columns
            df_to_return.to_csv(extractedCsvName, index=False) # Create empty CSV
            file_processing_endTime = time.time()
            duration = file_processing_endTime - file_processing_startTime
            return df_to_return, duration, processed_count_this_file, df_status
            
    except Exception as e: # Catch errors during file reading/parsing phase
        print(f"Error reading or parsing {splitFile}: {e}. Cannot extract features.")
        df_status = f"Error (Reading/Parsing: {e})"
        file_processing_endTime = time.time()
        duration = file_processing_endTime - file_processing_startTime
        return df_to_return, duration, processed_count_this_file, df_status

    # If file reading was successful and sequences exist:
    tasks = [(record, enzyme_batch, enzyme_names, variants, aa_map, aa_names_sorted_list,
              aa_hydrophobic, aa_polar, aa_positive, aa_negative)
             for record in sequence_records]

    data_results = []
    protein_error_count = 0
    empty_protein_after_cleanup_count = 0
    
    num_workers = max(1, int(cpu_count() * 0.8))
    # print(f"Using {num_workers} worker processes for {splitFile}.") # Moved print to main loop

    chunk_size = 1
    if tasks:
        chunk_size = max(1, len(tasks) // (num_workers * 4))
        if len(tasks) < num_workers * 2 :
            chunk_size = 1

    try:
        with Pool(processes=num_workers) as pool:
            results_from_pool = pool.map(process_sequence_record, tasks, chunksize=chunk_size)

        for result_tuple in results_from_pool:
            if result_tuple:
                row, protein_ok, protein_empty_flag = result_tuple
                data_results.append(row)
                processed_count_this_file +=1 # Use local counter for this file
                if not protein_ok :
                    protein_error_count +=1
                if protein_empty_flag:
                     empty_protein_after_cleanup_count += 1
        
        df_to_return = pd.DataFrame(data_results)
        if not df_to_return.empty:
            df_to_return = df_to_return.reindex(columns=columns, fill_value=0.0) 
        else: # If all sequences failed processing or data_results is empty
            df_to_return = pd.DataFrame(columns=columns) # Ensure columns exist

        df_to_return.to_csv(extractedCsvName, index=False)
        df_status = "Success"
        # Print moved to main loop for better context
        # print(f"Processed {processed_count_this_file} sequences from {splitFile}.")
        # if protein_error_count > 0 :
        #     print(f"Encountered issues during protein analysis for {protein_error_count} sequences.")
        # if empty_protein_after_cleanup_count > 0:
        #      print(f"Translation resulted in an empty protein string for {empty_protein_after_cleanup_count} sequences.")
        # print(f"Feature extraction for {splitFile} completed. Output: {extractedCsvName}")

    except Exception as e: # Catch errors during the multiprocessing/DataFrame creation phase
        print(f"Critical error during feature processing for {splitFile}: {e}")
        import traceback
        traceback.print_exc()
        df_status = f"Error (Processing: {e})"
        # df_to_return remains as it was before this try block (empty or from parsing)

    file_processing_endTime = time.time()
    duration = file_processing_endTime - file_processing_startTime
    return df_to_return, duration, processed_count_this_file, df_status


if __name__ == "__main__":
    overall_start_time = time.time()
    
    TIMING_LOG_FILE = "feature_extraction_timing_log.csv"

    # Initialize Timing Log File
    with open(TIMING_LOG_FILE, 'w', newline='') as f_log:
        log_writer = csv.writer(f_log)
        log_writer.writerow(['FASTA_File', 'Processing_Time_Seconds', 'Sequences_Processed', 'Status'])

    # --- Enzyme Batch Selection ---
    # Option 1: Smaller, custom batch (MUCH FASTER)
    # selected_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI])
    # enzyme_batch_global = selected_enzymes
    # Option 2: Broad set (SLOWER)
    enzyme_batch_global = CommOnly
    
    enzyme_names_global = [str(enz).upper() for enz in enzyme_batch_global]
    enzyme_list_for_info = list(enzyme_batch_global) 
    first_five_enzyme_names = [str(e) for e in islice(enzyme_list_for_info, 5)]
    print("Enzyme batch selected:", first_five_enzyme_names, "..." if len(enzyme_list_for_info) > 5 else "")
    print("Total enzymes in batch:", len(enzyme_list_for_info))


    print("pre timer: Loading variants key...")
    try:
        variants_global = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
        print(f"Loaded {len(variants_global)} variants from key.fasta")
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
        variants_global = {}
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")
        variants_global = {}

    print("Starting batch feature extraction...")

    file_types = ["test", "train"]
    percentages = range(10, 91, 10)
    versions = range(1, 6)
    base_name_part = "(11000)"
    output_suffix = "(extracted_ALT5)" 

    files_to_process = []
    for f_type in file_types:
        for perc in percentages:
            for ver in versions:
                fasta_filename = f"{f_type}{perc}{base_name_part}({ver}).fasta"
                csv_filename = f"{f_type}{perc}{base_name_part}({ver}){output_suffix}.csv"
                files_to_process.append({'fasta': fasta_filename, 'csv': csv_filename})

    total_files = len(files_to_process)
    print(f"Found {total_files} files to process based on naming convention.")

    processed_successfully_count = 0 # Counts files where CSV was generated
    critical_error_count = 0       # Counts files with major script-stopping errors for that file

    for i, file_info in enumerate(files_to_process):
        fasta_file = file_info['fasta']
        csv_output_file = file_info['csv']
        
        print(f"\n--- Processing file {i+1}/{total_files}: {fasta_file} ---")
        
        # extractedDF now returns: df, duration, sequences_processed_in_file, status_string
        df_extracted, file_duration, num_seq_processed, status = extractedDF(
                                      fasta_file, csv_output_file,
                                      enzyme_batch_global, enzyme_names_global, variants_global,
                                      aa_mapping_code_to_name_global, aa_names_sorted_global,
                                      AA_HYDROPHOBIC_GLOBAL, AA_POLAR_UNCHARGED_GLOBAL,
                                      AA_POSITIVE_GLOBAL, AA_NEGATIVE_GLOBAL)
        
        print(f"Status for {fasta_file}: {status}")
        print(f"Time taken for {fasta_file}: {file_duration:.2f} seconds.")
        if num_seq_processed > 0 or status == "Processed (No sequences)": # If some processing happened or empty file handled
             print(f"Sequences processed in {fasta_file}: {num_seq_processed}")
        
        # Log timing and status
        with open(TIMING_LOG_FILE, 'a', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow([fasta_file, f"{file_duration:.4f}", num_seq_processed, status])

        if status == "Success" or status == "Processed (No sequences)":
            processed_successfully_count += 1
            # Optionally print head if successful and not empty
            # if status == "Success" and not df_extracted.empty:
            #     print(f"Head of DataFrame for {csv_output_file}:")
            #     print(df_extracted.head())
        elif status.startswith("Skipped") or status.startswith("Error"):
            critical_error_count +=1 # Count these as files that didn't complete fully

    overall_end_time = time.time()
    print("\n--- Batch Processing Summary ---")
    print(f"Total files scheduled for processing: {total_files}")
    print(f"Files resulting in a CSV (Success or Empty): {processed_successfully_count}")
    print(f"Files skipped or with errors: {critical_error_count}")
    print(f"Timing log written to: {TIMING_LOG_FILE}")
    print(f"Total time for batch processing: {(overall_end_time - overall_start_time)/60:.2f} minutes")
    print("Batch extraction finished.")