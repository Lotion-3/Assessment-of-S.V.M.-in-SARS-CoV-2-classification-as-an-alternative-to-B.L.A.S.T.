import pandas as pd
from Bio.Seq import Seq
from Bio import SeqIO, BiopythonWarning
import time
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count, freeze_support
import warnings
import os
import csv

# --- MODIFICATION: Define and generate ONLY 7-mers ---
DNA_BASES = 'ATGC'

def generate_kmers(k, alphabet=DNA_BASES):
    """Generates all possible k-mers for a given alphabet."""
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

# Generate the list of all 16,384 possible 7-mers.
ALL_7MERS = generate_kmers(7)

def calculate_kmer_percentages(sequence_str, k, all_possible_kmers_list):
    """
    Calculates the percentage of each k-mer in a sequence string.
    This function is generic and does not need to be changed.
    """
    seq_len = len(sequence_str)
    if seq_len < k:
        return {kmer: 0.0 for kmer in all_possible_kmers_list}

    counts = Counter()
    valid_kmer_count_total = 0

    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        if all(base in DNA_BASES for base in kmer):
            counts[kmer] += 1
            valid_kmer_count_total += 1

    percentages = {kmer: 0.0 for kmer in all_possible_kmers_list}
    if valid_kmer_count_total == 0:
        return percentages

    for kmer_seq, count in counts.items():
        if kmer_seq in percentages:
            percentages[kmer_seq] = (count / valid_kmer_count_total) * 100
            
    return percentages

def process_sequence_record_kmers(sequence_record_tuple):
    """
    Worker function for multiprocessing. 
    --- MODIFICATION: Simplified to process ONLY 7-mers. ---
    """
    sequence_record, variants_dict = sequence_record_tuple
    seq_id = sequence_record.id
    seq_obj = sequence_record.seq
    seq_len = len(seq_obj)

    # Initialize the row with basic info and all 7-mer columns set to 0.0
    row = {'id': seq_id, 'Sequence Length': seq_len}
    for kmer in ALL_7MERS:
        row[f"{kmer}_perc"] = 0.0
    
    row['Variant'] = variants_dict.get(seq_id, 'UNKNOWN')

    # Optimization: skip calculation if sequence is too short for a 7-mer
    if seq_len < 7:
        return row

    seq_str_upper = str(seq_obj).upper()
    
    # Calculate and update percentages ONLY for 7-mers
    kmer_7_percs = calculate_kmer_percentages(seq_str_upper, 7, ALL_7MERS)
    for kmer, perc in kmer_7_percs.items():
        row[f"{kmer}_perc"] = perc
        
    return row

def extractedDF_kmers(splitFile, extractedCsvName, variants_dict):
    """
    Orchestrates the feature extraction for a single FASTA file.
    --- MODIFICATION: Simplified column definitions for ONLY 7-mers. ---
    """
    file_processing_startTime = time.time()
    
    # Define the columns for the output DataFrame
    columns = ['id', 'Sequence Length']
    columns.extend([f"{kmer}_perc" for kmer in ALL_7MERS])
    columns.append('Variant')
    
    df_status = "Skipped (File not found)"
    processed_count_this_file = 0
    df_to_return = pd.DataFrame(columns=columns)

    # File reading and error handling remains the same
    try:
        if not os.path.exists(splitFile):
            print(f"Error: Input file {splitFile} not found. Skipping.")
            return df_to_return, time.time() - file_processing_startTime, 0, df_status
        sequence_records = list(SeqIO.parse(splitFile, "fasta"))
        if not sequence_records:
            print(f"No sequences found in {splitFile}. Output CSV will be empty.")
            return df_to_return, time.time() - file_processing_startTime, 0, "Processed (No sequences)"
    except Exception as e:
        print(f"Error reading or parsing {splitFile}: {e}. Cannot extract features.")
        return df_to_return, time.time() - file_processing_startTime, 0, f"Error (Reading: {e})"

    tasks = [(record, variants_dict) for record in sequence_records]
    num_workers = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_workers} worker processes for {splitFile}.")
    
    # Multiprocessing and DataFrame creation logic is robust and requires no changes
    chunk_size = max(1, len(tasks) // (num_workers * 4))
    try:
        with Pool(processes=num_workers) as pool:
            results_from_pool = pool.map(process_sequence_record_kmers, tasks, chunksize=chunk_size)
        
        df_to_return = pd.DataFrame(results_from_pool)
        if not df_to_return.empty:
            df_to_return = df_to_return.reindex(columns=columns, fill_value=0.0)
        
        df_to_return.to_csv(extractedCsvName, index=False)
        df_status = "Success"
        processed_count_this_file = len(df_to_return)

    except Exception as e:
        print(f"Critical error during feature processing for {splitFile}: {e}")
        import traceback
        traceback.print_exc()
        df_status = f"Error (Processing: {e})"
        
    file_processing_endTime = time.time()
    duration = file_processing_endTime - file_processing_startTime
    return df_to_return, duration, processed_count_this_file, df_status

# --- Main execution block ---
if __name__ == '__main__':
    freeze_support() 

    warnings.filterwarnings("ignore", category=BiopythonWarning, message="Partial codon.*")
    print("7-mer Feature Extractor - Start") 

    overall_start_time = time.time()
    
    # --- MODIFICATION: Changed log file name for clarity ---
    TIMING_LOG_FILE = "7mer_feature_extraction_timing_log.csv"

    with open(TIMING_LOG_FILE, 'w', newline='') as f_log:
        log_writer = csv.writer(f_log)
        log_writer.writerow(['FASTA_File', 'Processing_Time_Seconds', 'Sequences_Processed', 'Status'])

    print("Loading variants key...")
    variants_main = {} 
    try:
        variants_main = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
        print(f"Loaded {len(variants_main)} variants from key.fasta")
    except FileNotFoundError:
        print("Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")

    print("Starting batch 7-mer feature extraction...")

    # --- MODIFICATION: Updated output CSV filenames to be descriptive for 7-mers ---
    files_to_process = [
        {'fasta': "rawShuffledFastas/train10(11000)(1).fasta", 'csv': "rawShuffledFastas/train10(11000)(1)(extractedALT9).csv"},
        {'fasta': "rawShuffledFastas/test90(11000)(1).fasta",  'csv': "rawShuffledFastas/test90(11000)(1)(extractedALT9).csv"},
    ]

    total_files = len(files_to_process)
    print(f"Found {total_files} files to process for 7-mer extraction.")

    # The main processing loop requires no logical changes
    for i, file_info in enumerate(files_to_process):
        fasta_file = file_info['fasta']
        csv_output_file = file_info['csv']
        
        print(f"\n--- Processing file {i+1}/{total_files}: {fasta_file} for 7-mers ---")
        
        df_extracted, file_duration, num_seq_processed, status = extractedDF_kmers(
                                      fasta_file, 
                                      csv_output_file, 
                                      variants_main
                                      )
        
        print(f"Status for {fasta_file}: {status}")
        print(f"Time taken: {file_duration:.2f} seconds. Sequences processed: {num_seq_processed}")
        
        with open(TIMING_LOG_FILE, 'a', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow([fasta_file, f"{file_duration:.4f}", num_seq_processed, status])

        if status == "Success" and not df_extracted.empty:
            print(f"Head of DataFrame for {csv_output_file}:")
            # Print first 5 rows and first 10 columns
            print(df_extracted.iloc[:, :10].head()) 

    overall_end_time = time.time()
    print("\n--- 7-mer Batch Processing Summary ---")
    print(f"Total files processed: {total_files}")
    print(f"Timing log written to: {TIMING_LOG_FILE}")
    print(f"Total time for 7-mer batch processing: {(overall_end_time - overall_start_time)/60:.2f} minutes")
    print("7-mer batch extraction finished.")