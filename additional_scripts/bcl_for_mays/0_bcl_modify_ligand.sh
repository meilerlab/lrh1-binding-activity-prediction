#!/bin/bash

# Modify parent ligand (RJW100) with BCL to generate congeneric series. 
# Using https://meilerlab.org/wp-content/uploads/2022/02/Tutorial_5.pdf as starting point.
# Author of tutorial is Benjamin Brown @ Vanderbilt Univ.

ROSETTA=/your/rosetta/path/to/main/
BCL=/your/path/to/bcl/

##############################
# Prepare ligand # to get RJW ready for R deletions  

PDB=5l11
LIG=RJW
INSDF=../${PDB}_lig.sdf

# clean up sdf file 
${BCL}/bcl.exe molecule:Filter \
	-input_filenames ${INSDF} \
	-output_matched ${LIG}_0.sdf \
	-defined_atom_types \
	-3d \
	-add_h \
	-neutralize

INSDF=${LIG}_0.sdf

# generate rosetta params
${ROSETTA}/source/scripts/python/public/molfile_to_params.py \
	  -n ${LIG} -p ${LIG} ${INSDF} --root_atom=1 --centroid --extra_torsion_output

# convert params to molecule file  - necessary to specify parts to modify 
${ROSETTA}/source/bin/restype_converter.bcl.linuxgccrelease \
	  -extra_res_fa ${LIG}.fa.params -out:pdb

##############################
# next: Rosetta BCLFragmentMutateMover
# GO TO 1_bcl_modify_ligand.sh 

