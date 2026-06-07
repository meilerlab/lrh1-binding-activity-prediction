#!/bin/bash

# Remove R group, save as ./bcl_outputs/*sans*

BCL=/your/path/to/bcl/

PDB=5l11
LIG=RJW

# use the *sans* molecules for 2_*sh where removed respective R* groups.
${BCL}/bcl.exe molecule:Mutate -input_filenames ${LIG}_0.sdf -implementation "RemoveBond(mutable_atoms=0)" -output ./bcl_outputs/${LIG}.sans.R1.sdf

${BCL}/bcl.exe molecule:Mutate -input_filenames ${LIG}_0.sdf -implementation "RemoveAtom(mutable_atoms=5 6)" -output ./bcl_outputs/${LIG}.sans.R2.sdf

${BCL}/bcl.exe molecule:Mutate -input_filenames ${LIG}_0.sdf -implementation "RemoveAtom(mutable_atoms=23)" -output ./bcl_outputs/${LIG}.sans.R3.sdf

##############################
# next: generate params to determine which H to use for exo and endo
# GO TO 1.5_bcl_modify_ligand.sh

