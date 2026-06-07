#!/bin/bash

####################################################################
# Modified from : BCL-Rosetta sample RosettaScripts run file
####################################################################

# Global variables
ROSETTA=/your/rosetta/path/to/main/source/bin/rosetta_scripts.bcl.linuxgccrelease

# Input variables
XML=`readlink -e $1`
PROTEIN=`readlink -e $2`
LIGAND=`readlink -e $3`
PARAMS=`readlink -e $4`
PREFIX=$5
MUTABLE_ATOMS=$6
MEDCHEM_FRAGMENTS=$7
INDEXH=$8

# Derived variables
protein=`basename $PROTEIN .pdb`
ligand=`basename $LIGAND .pdb`

# Run
$ROSETTA \
    -parser:protocol $XML \
    -in:file:s "$PROTEIN $LIGAND" \
    -extra_res_fa ${PARAMS} \
    -parser:script_vars mutatoms="${MUTABLE_ATOMS}" \
    -parser:script_vars medchem_fragments="${MEDCHEM_FRAGMENTS}" \
    -parser:script_vars indexh="${INDEXH}" \
    -out:prefix $PREFIX \
    -out:pdb_gz true \
    -nstruct 5 \
    -in:file:fullatom \
    -restore_talaris_behavior \
    -ignore_zero_occupancy false \
    -linmem_ig 10 \
    -constant_seed true \
    -overwrite

