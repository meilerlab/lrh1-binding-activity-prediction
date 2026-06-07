#!/bin/bash

# Generate params etc files of RJW without R group (*sans* ligands) 

ROSETTA=/your/rosetta/path/to/main/
BCL=/your/path/to/bcl/

# lig name will be "missing R*" ==> XR1

Rmod=R3 # swap this for R1, R2.. 

LIG=X${Rmod}
INSDF=./bcl_outputs/RJW.sans.${Rmod}.sdf

# clean up sdf file 
${BCL}/bcl.exe molecule:Filter \
	-input_filenames ${INSDF} \
	-output_matched ${LIG}.sdf \
	-defined_atom_types \
	-3d \
	-add_h \
	-neutralize

INSDF=${LIG}.sdf

# generate rosetta params
${ROSETTA}/source/scripts/python/public/molfile_to_params.py \
	  -n ${LIG} -p ${LIG} ${INSDF} --root_atom=1 --centroid --extra_torsion_output

# convert params to molecule file  - necessary to specify parts to modify 
${ROSETTA}/source/bin/restype_converter.bcl.linuxgccrelease \
	  -extra_res_fa ${LIG}.fa.params -out:pdb

##############################
# next: use BCL add_medchem to generate modified ligands 
# GO TO 2_bcl_modify_ligand.sh 

