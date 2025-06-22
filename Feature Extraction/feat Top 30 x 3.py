from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import time
import itertools
from collections import Counter
from multiprocessing import Pool, cpu_count
from itertools import islice

from Bio.Restriction import Restriction, RestrictionBatch

# --- Global Definitions based on the target feature list ---
TARGET_FEATURE_HEADER = [
    'id', 'AAA_perc', 'AAG_perc', 'AAT_perc', 'ACCIII_cuts', 'ACC_perc', 'ACG_perc',
    'AFLIII_cuts', 'AGA_perc', 'AGEI_cuts', 'AGG_perc', 'AHDI_cuts', 'AJUI_cuts',
    'ALEI_cuts', 'ARSI_cuts', 'ASIGI_cuts', 'ATA_perc', 'ATC_perc', 'AT_perc',
    'BCGI_cuts', 'BMERI_cuts', 'BMSI_cuts', 'BPII_cuts', 'BPMI_cuts', 'BPU10I_cuts',
    'BPUEI_cuts', 'BPUMI_cuts', 'BSAWI_cuts', 'BSAXI_cuts', 'BSEAI_cuts',
    'BSEMI_cuts', 'BSERI_cuts', 'BSGI_cuts', 'BSHTI_cuts', 'BSISI_cuts',
    'BSO31I_cuts', 'BSPTI_cuts', 'BSPTNI_cuts', 'BST1107I_cuts', 'BTSI_cuts',
    'CAT_perc', 'CCC_perc', 'CCG_perc', 'CCT_perc', 'CGA_perc', 'CGG_perc',
    'CGT_perc', 'CG_perc', 'CSPAI_cuts', 'ECO57I_cuts', 'FALI_cuts',
    'FAUNDI_cuts', 'GAG_perc', 'GCC_perc', 'GCG_perc', 'GGA_perc', 'GGG_perc',
    'GG_perc', 'GTC_perc', 'HINFI_cuts', 'HPY99I_cuts', 'KPN2I_cuts', 'MMEI_cuts',
    'NCII_cuts', 'NDEI_cuts', 'NMEAIII_cuts', 'NSPI_cuts', 'OLII_cuts',
    'PFEI_cuts', 'SEXAI_cuts', 'STUI_cuts', 'Sequence Length', 'TCC_perc',
    'TCG_perc', 'TGG_perc', 'TTA_perc', 'XCEI_cuts', 'Variant'
]

DNA_BASES = 'ATGC'

def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

ALL_POSSIBLE_DINUCLEOTIDES = generate_kmers(2)
ALL_POSSIBLE_TRINUCLEOTIDES = generate_kmers(3)

def calculate_kmer_percentages(sequence_str, k, all_possible_kmers_list):
    seq_len = len(sequence_str)
    percentages = {kmer: 0.0 for kmer in all_possible_kmers_list}
    if seq_len < k:
        return percentages
    counts = Counter()
    valid_kmer_starts = 0
    for i in range(seq_len - k + 1):
        kmer = sequence_str[i:i+k]
        if all(base in DNA_BASES for base in kmer):
            counts[kmer] += 1
            valid_kmer_starts += 1
    if valid_kmer_starts == 0:
        return percentages
    for kmer_seq, count in counts.items():
        if kmer_seq in percentages:
            percentages[kmer_seq] = (count / valid_kmer_starts) * 100
    return percentages

HEADER_TO_BIOPYTHON_ENZYME_MAP = {
    "ACCIII": "Acc65I", "AFLIII": "AflIII", "AGEI": "AgeI", "AHDI": "AhdI",
    "AJUI": "AjuI", "ALEI": "AleI", "ARSI": "ArsI", "ASIGI": "AsiSI",
    "BCGI": "BcgI", "BMERI": "BmeRI", "BMSI": "BmsI", "BPII": "BpiI",
    "BPMI": "BpmI", "BPU10I": "Bpu10I", "BPUEI": "BpuEI", "BPUMI": "BpuMI",
    "BSAWI": "BsaWI", "BSAXI": "BsaXI", "BSEAI": "BseAI", "BSEMI": "BseMI",
    "BSERI": "BseRI", "BSGI": "BsgI", "BSHTI": "BshTI", "BSISI": "BsiSI",
    "BSO31I": "BsoBI", "BSPTI": "BspTI", "BSPTNI": "BspTNI", "BST1107I": "Bst1107I",
    "BTSI": "BtsI", "CSPAI": "CspAI", "ECO57I": "Eco57I", "FALI": "FalI",
    "FAUNDI": "FauNDI", "HINFI": "HinfI", "HPY99I": "Hpy99I", "KPN2I": "Kpn2I",
    "MMEI": "MmeI", "NCII": "NciI", "NDEI": "NdeI", "NMEAIII": "NmeAIII",
    "NSPI": "NspI", "OLII": "OliI", "PFEI": "PfeI", "SEXAI": "SexAI",
    "STUI": "StuI", "XCEI": "XceI"
}

REQUESTED_DINUCLEOTIDES = []
REQUESTED_TRINUCLEOTIDES = []
REQUESTED_RE_HEADER_NAMES = []

for feature_name in TARGET_FEATURE_HEADER:
    if feature_name.endswith("_perc"):
        kmer = feature_name.split('_')[0]
        if len(kmer) == 2 and all(base in DNA_BASES for base in kmer):
            REQUESTED_DINUCLEOTIDES.append(kmer)
        elif len(kmer) == 3 and all(base in DNA_BASES for base in kmer):
            REQUESTED_TRINUCLEOTIDES.append(kmer)
    elif feature_name.endswith("_cuts") and feature_name != "id_cuts":
        REQUESTED_RE_HEADER_NAMES.append(feature_name.split('_')[0])

enzyme_batch_list_objects_global = []
biopython_name_to_header_name_map_global = {}
processed_enzyme_header_names_global = []

print("\n--- Configuring Global Enzyme Batch for Specific Enzymes ---")
for header_re_name in REQUESTED_RE_HEADER_NAMES:
    biopython_name = HEADER_TO_BIOPYTHON_ENZYME_MAP.get(header_re_name)
    if biopython_name:
        try:
            enzyme_class_obj = None
            if hasattr(Restriction, biopython_name):
                potential_obj = getattr(Restriction, biopython_name)
                if str(potential_obj).replace(" RestrictionType", "") == biopython_name:
                    enzyme_class_obj = potential_obj
            if enzyme_class_obj is None:
                for e_type in Restriction.AllEnzymes:
                    if str(e_type) == biopython_name:
                        enzyme_class_obj = e_type
                        break
            if enzyme_class_obj:
                if enzyme_class_obj not in enzyme_batch_list_objects_global:
                    enzyme_batch_list_objects_global.append(enzyme_class_obj)
                biopython_name_to_header_name_map_global[str(enzyme_class_obj)] = header_re_name
                if header_re_name not in processed_enzyme_header_names_global:
                    processed_enzyme_header_names_global.append(header_re_name)
            else:
                print(f"Warning: Mapped BioPython name '{biopython_name}' for '{header_re_name}' not found as valid enzyme.")
        except Exception as e:
            print(f"Warning: Could not load enzyme for '{biopython_name}' (from '{header_re_name}'): {e}")
    else:
        print(f"Warning: No BioPython mapping for header enzyme '{header_re_name}'. It will have 0 cuts.")
        if header_re_name not in processed_enzyme_header_names_global:
             processed_enzyme_header_names_global.append(header_re_name)

if enzyme_batch_list_objects_global:
    enzyme_batch_global_specific = RestrictionBatch(enzyme_batch_list_objects_global)
    print(f"Successfully created global batch for {len(enzyme_batch_global_specific)} enzymes.")
else:
    enzyme_batch_global_specific = RestrictionBatch([])
    print("Warning: Global enzyme batch is empty. All RE cut features will be 0.")

variants_global = {}

def process_sequence_record(sequence_record_tuple):
    sequence_record, \
    target_header, \
    req_di, req_tri, \
    enzyme_batch_arg, bio_to_header_map_arg, \
    variants_arg = sequence_record_tuple

    seq_id = sequence_record.id
    seq_obj = sequence_record.seq
    seq_len = len(seq_obj)

    row = {col: 0.0 for col in target_header if col not in ['id', 'Variant']}
    row['id'] = seq_id
    row['Variant'] = "UNKNOWN"

    if seq_len == 0:
        row['Variant'] = variants_arg.get(seq_id, "UNKNOWN")
        return row

    seq_str_upper = str(seq_obj).upper()

    if 'Sequence Length' in target_header:
        row['Sequence Length'] = seq_len

    if req_di:
        all_di_percs = calculate_kmer_percentages(seq_str_upper, 2, ALL_POSSIBLE_DINUCLEOTIDES)
        for di_kmer in req_di:
            col_name = f"{di_kmer}_perc"
            if col_name in target_header:
                row[col_name] = all_di_percs.get(di_kmer, 0.0)
    
    if req_tri:
        all_tri_percs = calculate_kmer_percentages(seq_str_upper, 3, ALL_POSSIBLE_TRINUCLEOTIDES)
        for tri_kmer in req_tri:
            col_name = f"{tri_kmer}_perc"
            if col_name in target_header:
                row[col_name] = all_tri_percs.get(tri_kmer, 0.0)

    # --- CORRECTED PART ---
    if enzyme_batch_arg and len(enzyme_batch_arg) > 0: # Check if batch contains any enzymes
    # --- END CORRECTED PART ---
        try:
            analysis = enzyme_batch_arg.search(seq_obj)
            for enz_obj_from_analysis, sites in analysis.items():
                original_header_name = bio_to_header_map_arg.get(str(enz_obj_from_analysis))
                if original_header_name:
                    col_name = f"{original_header_name}_cuts"
                    if col_name in target_header:
                        row[col_name] = len(sites)
        except Exception:
            pass

    row['Variant'] = variants_arg.get(seq_id, 'UNKNOWN')
    return row

def extractedDF_specific(splitFile, extractedCsvName,
                         target_feature_list,
                         requested_dinucleotides_list,
                         requested_trinucleotides_list,
                         enzyme_batch_to_use,
                         enzyme_map_to_use,
                         variants_dict):
    startTime = time.time()
    sequence_records = list(SeqIO.parse(splitFile, "fasta"))
    
    tasks = [(record,
              target_feature_list,
              requested_dinucleotides_list, requested_trinucleotides_list,
              enzyme_batch_to_use, enzyme_map_to_use,
              variants_dict
              ) for record in sequence_records]

    data_results = []
    processed_count = 0
    num_workers = max(1, int(cpu_count() * 0.8))
    print(f"Using {num_workers} worker processes for {splitFile}.")

    with Pool(processes=num_workers) as pool:
        results_from_pool = pool.map(process_sequence_record, tasks)

    for row_dict in results_from_pool:
        if row_dict:
            data_results.append(row_dict)
            processed_count += 1

    df = pd.DataFrame(data_results)
    if not df.empty:
        df = df.reindex(columns=target_feature_list) 
        for col in target_feature_list:
            if col not in ['id', 'Variant']:
                if col not in df.columns or df[col].isnull().any():
                    df[col] = df.get(col, pd.Series(0.0, index=df.index)).fillna(0.0)
        if 'Variant' in target_feature_list:
             df['Variant'] = df.get('Variant', pd.Series("UNKNOWN", index=df.index)).fillna("UNKNOWN")

    df.to_csv(extractedCsvName, index=False)
    endTime = time.time()
    print(f"Processed {processed_count} sequences from {splitFile}.")
    print(f"Feature extraction for {splitFile} completed in {endTime-startTime:.2f} seconds. Output: {extractedCsvName}")
    return df

if __name__ == "__main__":
    print("--- Feature Extractor for Specific Feature List ---")
    try:
        for rec in SeqIO.parse("key.fasta", "fasta"):
            parts = rec.description.split(maxsplit=1)
            variants_global[rec.id] = parts[1] if len(parts) > 1 else "UNKNOWN_KEY_FORMAT"
    except FileNotFoundError:
        print(f"Warning: key.fasta not found. Variant information will be 'UNKNOWN'.")
    except Exception as e:
        print(f"Error loading key.fasta: {e}. Variant information will be 'UNKNOWN'.")

    input_fasta_train = "train10(11000)(1).fasta"
    input_fasta_test = "test90(11000)(1).fasta"
    key_file_path = "key.fasta"

    for dummy_fasta in [input_fasta_train, input_fasta_test, key_file_path]:
        try:
            with open(dummy_fasta, "r") as f:
                if not f.read(1): raise FileNotFoundError
            next(SeqIO.parse(dummy_fasta, "fasta"))
        except (FileNotFoundError, StopIteration):
            print(f"Creating dummy {dummy_fasta} for testing...")
            if dummy_fasta == key_file_path:
                with open(dummy_fasta, "w") as f:
                    f.write(">seqT1 VariantAlpha\n>seqT2 VariantBeta\n")
                    f.write(">seqE1 VariantGamma\n>seqE2 VariantAlpha\n")
            else:
                with open(dummy_fasta, "w") as f:
                    prefix = "seqT" if "train" in dummy_fasta else "seqE"
                    f.write(f">{prefix}1\nAAACCCTTTGGGACGTACGTACGTAGAATTCGGTACC\n")
                    f.write(f">{prefix}2\nCCCCGGGGTTTTAAAACGTACGTACGTACGTAGAATTC\n")

    print("\n--- Starting Extraction for Train Data ---")
    try:
        trainExtract = extractedDF_specific(
            input_fasta_train,
            "train_specific_features.csv",
            TARGET_FEATURE_HEADER,
            REQUESTED_DINUCLEOTIDES,
            REQUESTED_TRINUCLEOTIDES,
            enzyme_batch_global_specific,
            biopython_name_to_header_name_map_global,
            variants_global
        )
        if not trainExtract.empty:
            print("Head of trainExtract DataFrame:")
            print(trainExtract.head().to_string())
            
    except FileNotFoundError as e:
        print(f"Error: Input FASTA file not found: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred during train feature extraction: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Starting Extraction for Test Data ---")
    try:
        testExtract = extractedDF_specific(
            input_fasta_test,
            "test_specific_features.csv",
            TARGET_FEATURE_HEADER,
            REQUESTED_DINUCLEOTIDES,
            REQUESTED_TRINUCLEOTIDES,
            enzyme_batch_global_specific,
            biopython_name_to_header_name_map_global,
            variants_global
        )
        if not testExtract.empty:
            print("Head of testExtract DataFrame:")
            print(testExtract.head().to_string())

    except FileNotFoundError as e:
        print(f"Error: Input FASTA file not found: {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred during test feature extraction: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Extraction Script Finished ---")