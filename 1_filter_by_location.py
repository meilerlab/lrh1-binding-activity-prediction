#!/usr/bin/env python3

'''
1_filter_by_location.py : 
Identify compounds with docked pose near pocket mouth.
Reference residues used to calculate the distance are hard-coded in this script.
If multiple poses per compound, keep the entry with the smallest distance. 

Usage: python 1_filter_by_location.py --input_file inlist.csv --output_file outlist.csv --distance_cutoff 3.0 
'''

import sys
import pandas as pd
import gzip
import argparse
from Bio.PDB import PDBParser
from pathlib import Path 

##############################
location_of_all_docks='~/ulls/vulibrary/' # location of all docked poses
##############################
def main(inCSVs, ofname, dcut=10.0):
    parser = PDBParser(QUIET=True)

    # key = compound ID
    # value = (row, min_distance)
    best_per_compound = {}

    ref_resis = [123,218,222] # G421, Y516, K520

    for ifname in Path().glob(inCSVs):
        df = pd.read_csv(ifname)

        for _, row in df.iterrows():

            compound_id = row["ID"]  
            pdbpath = str(Path(location_of_all_docks) / row["posepath"])

            try:
                with gzip.open(pdbpath, "rt") as f:
                    structure = parser.get_structure("s", f)

                model = structure[0]

                protch = next((c for c in ["A","B"] if c in model), None)
                if not protch or "X" not in model:
                    continue

                # heavy atoms only
                prot_atoms = [
                    a for r in model[protch]
                    if r.get_id()[1] in ref_resis
                    for a in r if not a.get_name().startswith("H")
                ]

                check_atoms = [
                    a for r in model["X"]
                    for a in r if not a.get_name().startswith("H")
                ]

                if not prot_atoms or not check_atoms:
                    continue

                # compute minimum atom-atom distance
                min_dist = min(a1 - a2 for a1 in check_atoms for a2 in prot_atoms)
                #print(f"{compound_id},{min_dist},{pdbpath}")

                # only consider if within cutoff
                if min_dist <= dcut:

                    # store distance in row copy
                    row_copy = row.copy()
                    row_copy["min_distance"] = min_dist

                    # update if:
                    # 1) compound not seen before
                    # 2) this pose has smaller distance
                    if (compound_id not in best_per_compound or
                        min_dist < best_per_compound[compound_id][1]):
                        best_per_compound[compound_id] = (row_copy, min_dist)
                        
            except Exception as e:
                print(f"Failed {pdbpath}: {e}")

    if best_per_compound:
        final_rows = [v[0] for v in best_per_compound.values()]
        final_df = pd.DataFrame(final_rows)
        final_df["min_distance"] = final_df["min_distance"].round(2)
        final_df.to_csv(ofname, index=False)
        print(f"Saved {len(final_df)} unique compounds to {ofname}")
    else:
        print("No compounds met distance criteria.")

##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify compounds by proximity to select residues in docked pose.")
    parser.add_argument("--input_file", required=True, help="Input CSV files with compounds to calculation position.")
    parser.add_argument("--output_file", required=True, help="Output CSV containing compounds that meet cutoff criteria.")
    parser.add_argument("--distance_cutoff", type=float, default=10.0, help="Distance cutoff in Å.")
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.distance_cutoff)
