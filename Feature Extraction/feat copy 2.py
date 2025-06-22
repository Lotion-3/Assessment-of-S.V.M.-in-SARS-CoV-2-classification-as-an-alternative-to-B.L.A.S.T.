from Bio.Seq import Seq
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils.ProtParam import ProteinAnalysis
# sklearn imports are not used in this specific script, but good to keep if you plan to use them elsewhere
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
import time
# import timeit # Not used
from itertools import product
import itertools # For generating k-mer combinations
from collections import Counter

# --- Restriction Enzyme Imports ---
from Bio.Restriction import RestrictionBatch, CommOnly # Using a common set, or define your own
# For example, to define a custom batch:
# from Bio.Restriction import EcoRI, HindIII, BamHI
# custom_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI])
from Bio.Restriction import EcoRI, HindIII, BamHI
print("Very Start")
selected_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI])
enzyme_batch = selected_enzymes
# Define standard amino acids and DNA bases globally
print("Later start")
STANDARD_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
DNA_BASES = 'ATGC'

# Generate all possible k-mers for consistent column ordering
def generate_kmers(k, alphabet=DNA_BASES):
    return ["".join(p) for p in itertools.product(alphabet, repeat=k)]
print("slightly later")
ALL_DINUCLEOTIDES = generate_kmers(2)
ALL_TRINUCLEOTIDES = generate_kmers(3)

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

    # --- Define Restriction Enzyme Batch ---
    # Using a predefined batch of commercially available enzymes
    # You can customize this!
    # Example: selected_enzymes = RestrictionBatch([EcoRI, HindIII, BamHI, SmaI, PstI, NotI, XhoI, SpeI, SacI, KpnI])
    # Using CommOnly for a broader set of common enzymes
    enzyme_batch = CommOnly
    enzyme_names = [str(enz).upper() for enz in enzyme_batch] # Get enzyme names for column headers, ensure uniqueness

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
    # --- Add Restriction Enzyme Feature Columns ---
    for enz_name in enzyme_names:
        columns.append(f"{enz_name}_cuts")
    columns.append('Total_RE_Fragments') # Total fragments considering all enzymes in the batch
    columns.append('Avg_RE_Fragment_Length')
    # --- End Restriction Enzyme ---
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

        seq_str_upper = str(seq_obj).upper()

        nuc_counts_abs = {nuc: seq_str_upper.count(nuc) for nuc in DNA_BASES}
        total_atgc_bases = sum(nuc_counts_abs.values())

        if total_atgc_bases == 0:
            nuc_percs = {nuc: 0.0 for nuc in DNA_BASES}
            gc_perc_val = 0.0
        else:
            nuc_percs = {nuc: (nuc_counts_abs[nuc] / total_atgc_bases) * 100
                         for nuc in DNA_BASES}
            gc_perc_val = gc_fraction(seq_obj) * 100

        gc_sum = nuc_counts_abs['G'] + nuc_counts_abs['C']
        at_sum = nuc_counts_abs['A'] + nuc_counts_abs['T']

        gc_skew = ((nuc_counts_abs['G'] - nuc_counts_abs['C']) / gc_sum) if gc_sum > 0 else 0
        at_skew = ((nuc_counts_abs['A'] - nuc_counts_abs['T']) / at_sum) if at_sum > 0 else 0

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

        for name in aa_names_sorted:
            row[f"{name} perc"] = 0.0
        row.update({
            'Protein Molecular Weight': 0.0, 'Protein Isoelectric Point': 0.0,
            'Protein Aromaticity': 0.0, 'Protein Instability Index': 0.0,
            'Protein GRAVY': 0.0, 'Protein Helix Perc': 0.0,
            'Protein Turn Perc': 0.0, 'Protein Sheet Perc': 0.0
        })

        try:
            translated_seq_obj = seq_obj.translate(to_stop=True)
            protein_str_raw = str(translated_seq_obj)
            protein_str_cleaned = protein_str_raw.replace('X', '').replace('*', '')

            if not protein_str_cleaned:
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
        except Exception as e:
            protein_error_count +=1

        # --- Restriction Enzyme Feature Calculation ---
        # Ensure seq_obj is suitable for Restriction.Analysis (needs to be a Bio.Seq object)
        analysis = enzyme_batch.search(seq_obj) # Use the original seq_obj for analysis
        
        total_fragments_from_all_enzymes_in_batch = 1 # Starts with 1 fragment (the whole sequence)
        all_cuts_sites_combined = set() # To store unique cut positions

        for i, enz in enumerate(enzyme_batch):
            enz_name = enzyme_names[i] # Use the stored uppercase name
            cut_sites = analysis.get(enz, []) # Get cut sites for this enzyme
            row[f"{enz_name}_cuts"] = len(cut_sites)
            for site in cut_sites:
                all_cuts_sites_combined.add(site) # Add unique cut positions

        # Calculate total fragments and average length based on the combined unique cut sites
        # Sort unique cut sites
        sorted_unique_cuts = sorted(list(all_cuts_sites_combined))
        
        if not sorted_unique_cuts: # No cuts by any enzyme in the batch
            row['Total_RE_Fragments'] = 1
            row['Avg_RE_Fragment_Length'] = float(seq_len) if seq_len > 0 else 0.0
        else:
            # Number of fragments is number of unique cut sites + 1
            num_fragments = len(sorted_unique_cuts) + 1
            row['Total_RE_Fragments'] = num_fragments
            
            # Fragment lengths
            fragments_lengths = []
            current_pos = 0
            for cut_site in sorted_unique_cuts:
                # cut_site is 1-based index of the base *after* the cut for non-blunt,
                # or the first base of the downstream fragment for blunt.
                # The .catalyse() method or similar would give fragment objects directly.
                # Here, we approximate based on cut site positions from .search().
                # A cut at position X (1-based) means the fragment ends at X-1 (0-based)
                # or the new fragment starts at X (0-based).
                # Let's assume cut_sites from .search() are 1-based positions of the first base of the recognition site.
                # For simplicity, we'll use the cut positions to define fragment boundaries.
                # This needs careful checking with BioPython docs for how .search() reports positions
                # vs. how .catalyse() creates fragments.
                # Assuming cut_sites from search are the START of the recognition site (1-based)
                # A more robust way would be to use .catalyze() and get fragments,
                # but that's more complex if you only want counts for each enzyme from .search()
                
                # For now, let's assume cuts define points.
                # Fragment 1: 0 to cut_site_1
                # Fragment 2: cut_site_1 to cut_site_2 ...
                # Fragment N: cut_site_N-1 to end_of_seq
                
                # A simpler interpretation for total fragments:
                # If there are N unique cut points, there are N+1 fragments.
                # The average length is then seq_len / (N+1).
                row['Avg_RE_Fragment_Length'] = float(seq_len) / num_fragments if num_fragments > 0 else 0.0
        # --- End Restriction Enzyme Calculation ---
        
        data.append(row)
        processed_count += 1

    # Ensure all columns are present even if some processing failed for all sequences
    # (e.g., if all protein analyses failed, those columns should still exist with 0s)
    # This is handled by initializing the row dict with all keys.
    df = pd.DataFrame(data) # Create DataFrame from list of dicts
    # Reorder columns to match the predefined 'columns' list to ensure consistency
    df = df.reindex(columns=columns, fill_value=0.0) # fill_value handles missing columns if any (should not happen here)


    df.to_csv(extractedCsvName, index=False)

    endTime = time.time()
    print(f"Processed {processed_count} sequences.")
    if protein_error_count > 0 :
        print(f"Encountered unexpected errors during protein analysis for {protein_error_count} sequences. Their protein features are zeroed.")
    if empty_protein_after_cleanup_count > 0:
        print(f"Translation (after X/* removal) resulted in empty protein for {empty_protein_after_cleanup_count} sequences. Their protein features are zeroed.")
    print(f"Feature extraction completed in {endTime-startTime:.2f} seconds. Output: {extractedCsvName}")
    return df

print("pre timer")

def timer(operation): # This function is defined but not called in your example.
    start_time = time.time()
    operation() # Assuming operation is a callable
    end_time = time.time()
    timeTaken = (end_time-start_time)
    print(f"Operation took {timeTaken:.2f} seconds")
    return timeTaken

print("start")
# Ensure the FASTA files exist in the same directory or provide the full path
try:
    testExtract = extractedDF("test90(11000)(1).fasta", "test90(11000)(1)(extractedALT2).csv")
    #trainExtract = extractedDF("train10(11000)(1).fasta", "train10(11000)(1)(extractedALT2).csv")
    print("Extraction successful.")
    # print(testExtract.head()) # Optionally print head to check
except FileNotFoundError as e:
    print(f"Error: Input FASTA file not found: {e.filename}")
except Exception as e:
    print(f"An unexpected error occurred during feature extraction: {e}")
print("end")