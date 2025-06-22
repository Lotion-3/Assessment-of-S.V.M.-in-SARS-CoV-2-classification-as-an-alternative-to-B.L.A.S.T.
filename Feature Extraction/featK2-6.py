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

# --- All your global variables and function definitions go here ---
DNA_BASES = 'ATGC'

def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

ALL_2MERS = generate_kmers(2)
ALL_3MERS = generate_kmers(3)
ALL_4MERS = generate_kmers(4)
ALL_5MERS = generate_kmers(5)
ALL_6MERS = generate_kmers(6)

# variants_global = {} # This will be loaded in __main__ and passed around

def calculate_kmer_percentages(sequence_str, k, all_possible_kmers_list):
    # ... (function body as before) ...
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
    # ... (function body as before) ...
    sequence_record, variants_arg = sequence_record_tuple
    seq_id = sequence_record.id
    seq_obj = sequence_record.seq
    seq_len = len(seq_obj)
    row = {'id': seq_id, 'Sequence Length': seq_len}
    for kmer in ALL_2MERS: row[f"{kmer}_perc"] = 0.0
    for kmer in ALL_3MERS: row[f"{kmer}_perc"] = 0.0
    for kmer in ALL_4MERS: row[f"{kmer}_perc"] = 0.0
    for kmer in ALL_5MERS: row[f"{kmer}_perc"] = 0.0
    for kmer in ALL_6MERS: row[f"{kmer}_perc"] = 0.0
    row['Variant'] = variants_arg.get(seq_id, 'UNKNOWN')
    if seq_len == 0:
        return row
    seq_str_upper = str(seq_obj).upper()
    kmer_2_percs = calculate_kmer_percentages(seq_str_upper, 2, ALL_2MERS)
    for kmer, perc in kmer_2_percs.items(): row[f"{kmer}_perc"] = perc
    kmer_3_percs = calculate_kmer_percentages(seq_str_upper, 3, ALL_3MERS)
    for kmer, perc in kmer_3_percs.items(): row[f"{kmer}_perc"] = perc
    kmer_4_percs = calculate_kmer_percentages(seq_str_upper, 4, ALL_4MERS)
    for kmer, perc in kmer_4_percs.items(): row[f"{kmer}_perc"] = perc
    kmer_5_percs = calculate_kmer_percentages(seq_str_upper, 5, ALL_5MERS)
    for kmer, perc in kmer_5_percs.items(): row[f"{kmer}_perc"] = perc
    kmer_6_percs = calculate_kmer_percentages(seq_str_upper, 6, ALL_6MERS)
    for kmer, perc in kmer_6_percs.items(): row[f"{kmer}_perc"] = perc
    return row

def extractedDF_kmers(splitFile, extractedCsvName, variants_dict_arg): # Renamed arg for clarity
    # ... (function body as before, ensure it uses variants_dict_arg) ...
    file_processing_startTime = time.time()
    columns = ['id', 'Sequence Length']
    columns.extend([f"{kmer}_perc" for kmer in ALL_2MERS])
    columns.extend([f"{kmer}_perc" for kmer in ALL_3MERS])
    columns.extend([f"{kmer}_perc" for kmer in ALL_4MERS])
    columns.extend([f"{kmer}_perc" for kmer in ALL_5MERS])
    columns.extend([f"{kmer}_perc" for kmer in ALL_6MERS])
    columns.append('Variant')
    df_status = "Skipped (File not found)"
    processed_count_this_file = 0
    df_to_return = pd.DataFrame(columns=columns)
    try:
        if not os.path.exists(splitFile):
            print(f"Error: Input file {splitFile} not found. Skipping.")
            file_processing_endTime = time.time()
            duration = file_processing_endTime - file_processing_startTime
            return df_to_return, duration, processed_count_this_file, df_status
        sequence_records = list(SeqIO.parse(splitFile, "fasta"))
        if not sequence_records:
            print(f"No sequences found in {splitFile}. Output CSV will be empty.")
            df_status = "Processed (No sequences)"
            df_to_return.to_csv(extractedCsvName, index=False)
            file_processing_endTime = time.time()
            duration = file_processing_endTime - file_processing_startTime
            return df_to_return, duration, processed_count_this_file, df_status
    except Exception as e:
        print(f"Error reading or parsing {splitFile}: {e}. Cannot extract features.")
        df_status = f"Error (Reading/Parsing: {e})"
        file_processing_endTime = time.time()
        duration = file_processing_endTime - file_processing_startTime
        return df_to_return, duration, processed_count_this_file, df_status
    tasks = [(record, variants_dict_arg) for record in sequence_records] # Use the passed argument
    data_results = []
    num_workers = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_workers} worker processes for {splitFile}.")
    chunk_size = 1
    if tasks:
        chunk_size = max(1, len(tasks) // (num_workers * 4))
        if len(tasks) < num_workers * 2:
            chunk_size = 1
    try:
        with Pool(processes=num_workers) as pool:
            results_from_pool = pool.map(process_sequence_record_kmers, tasks, chunksize=chunk_size)
        for row_result in results_from_pool:
            if row_result:
                data_results.append(row_result)
                processed_count_this_file += 1
        df_to_return = pd.DataFrame(data_results)
        if not df_to_return.empty:
            df_to_return = df_to_return.reindex(columns=columns, fill_value=0.0)
        else:
            df_to_return = pd.DataFrame(columns=columns)
        df_to_return.to_csv(extractedCsvName, index=False)
        df_status = "Success"
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
    print("K-mer Feature Extractor - Start") 

    overall_start_time = time.time()
    
    TIMING_LOG_FILE = "kmer_feature_extraction_timing_log.csv"

    with open(TIMING_LOG_FILE, 'w', newline='') as f_log:
        log_writer = csv.writer(f_log)
        log_writer.writerow(['FASTA_File', 'Processing_Time_Seconds', 'Sequences_Processed', 'Status'])

    print("Loading variants key...")
    # Define variants_main in the main block
    variants_main = {} 
    try:
        variants_main = {rec.id: str(rec.seq) for rec in SeqIO.parse("key.fasta", "fasta")}
        print(f"Loaded {len(variants_main)} variants from key.fasta")
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")

    print("Starting batch k-mer feature extraction...")

    # For testing, directly use the files that caused issues
    files_to_process = [
        {'fasta': "rawShuffledFastas/train10(11000)(1).fasta", 'csv': "rawShuffledFastas/train10(11000)(1)(extracted_KMERS).csv"},
        {'fasta': "rawShuffledFastas/test90(11000)(1).fasta",  'csv': "rawShuffledFastas/test90(11000)(1)(extracted_KMERS).csv"},
        # Add the two extra calls here if you want them processed as part of the batch
        # Or call them separately after the loop, still within the __main__ block
    ]

    total_files = len(files_to_process)
    print(f"Found {total_files} files to process for k-mer extraction.")

    processed_successfully_count = 0
    critical_error_count = 0

    for i, file_info in enumerate(files_to_process):
        fasta_file = file_info['fasta']
        csv_output_file = file_info['csv']
        
        print(f"\n--- Processing file {i+1}/{total_files}: {fasta_file} for K-mers ---")
        
        df_extracted, file_duration, num_seq_processed, status = extractedDF_kmers(
                                      fasta_file, 
                                      csv_output_file, 
                                      variants_main # Pass the loaded variants
                                      )
        
        print(f"Status for {fasta_file}: {status}")
        print(f"Time taken for {fasta_file}: {file_duration:.2f} seconds.")
        if num_seq_processed > 0 or status == "Processed (No sequences)":
             print(f"Sequences processed in {fasta_file}: {num_seq_processed}")
        
        with open(TIMING_LOG_FILE, 'a', newline='') as f_log:
            log_writer = csv.writer(f_log)
            log_writer.writerow([fasta_file, f"{file_duration:.4f}", num_seq_processed, status])

        if status == "Success" or status == "Processed (No sequences)":
            processed_successfully_count += 1
            if status == "Success" and not df_extracted.empty:
                print(f"Head of DataFrame for {csv_output_file} (K-mers):")
                # print(df_extracted.iloc[:, :10].head()) 
        elif status.startswith("Skipped") or status.startswith("Error"):
            critical_error_count +=1

    # If you had extra calls, process them here:
    print("\n--- Processing additional files (if any) ---")
    
    # Call 1 (Original extra call)
    print("\n--- Processing rawShuffledFastas/test90(11000)(1).fasta for ALT6 ---")
    df_alt6_test, dur_alt6_test, num_alt6_test, stat_alt6_test = extractedDF_kmers(
        "rawShuffledFastas/test90(11000)(1).fasta", 
        "test90(11000)(1)(extractedALT6).csv", 
        variants_main
    )
    print(f"Status: {stat_alt6_test}, Time: {dur_alt6_test:.2f}s, Processed: {num_alt6_test}")
    # (Add to timing log if desired)

    # Call 2 (Original extra call)
    print("\n--- Processing rawShuffledFastas/train10(11000)(1).fasta for ALT6 ---")
    df_alt6_train, dur_alt6_train, num_alt6_train, stat_alt6_train = extractedDF_kmers(
        "rawShuffledFastas/train10(11000)(1).fasta", 
        "train10(11000)(1)(extractedALT6).csv", 
        variants_main
    )
    print(f"Status: {stat_alt6_train}, Time: {dur_alt6_train:.2f}s, Processed: {num_alt6_train}")
    # (Add to timing log if desired)


    overall_end_time = time.time()
    print("\n--- K-mer Batch Processing Summary ---")
    # ... (summary prints as before) ...
    print(f"Total files scheduled for k-mer processing: {total_files + 2}") # +2 for the extra calls if you count them
    print(f"Files resulting in a CSV (Success or Empty): {processed_successfully_count}") # This only counts loop
    print(f"Files skipped or with errors: {critical_error_count}") # This only counts loop
    print(f"Timing log written to: {TIMING_LOG_FILE}")
    print(f"Total time for k-mer batch processing: {(overall_end_time - overall_start_time)/60:.2f} minutes")
    print("K-mer batch extraction finished.")