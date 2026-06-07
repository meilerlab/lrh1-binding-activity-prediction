#!/bin/bash

# Add med chem fragments to XR*
# This script calls separate shell and xml scripts.
# design XML script is modified from tutorial (https://meilerlab.org/wp-content/uploads/2022/02/Tutorial_5.pdf)
# The procedure for each R* is nearly identical.
# When I ran this script, I executed each R* replacement chunk separately. 

PDB=5l11

#########
# R1 (comment in/out the stereochemistry desired) 
LIG=XR1 # missing("X") R1 and ready to be built 
MUTABLE_ATOMS=0 # same per R*

## exo stereochemistry
#TAG=exo
#INDEXH=0 # exo
#OUTPUT=out_r1exo

# endo stereochemistry
TAG=endo
INDEXH=1 # exo
OUTPUT=out_r1endo

for ((j=0;j<=8;j++)); do

    # fragment same regardless of exo/endo
    MEDCHEM_FRAGMENTS=./fragments/R1_${j}.sdf    
    bash mays19jbc.r1.design.sh \
	 mays19jbc.r1.design.xml \
	 ../${PDB}_prot.pdb  \
	 ${LIG}_0001.fa.pdb \
	 ${LIG}.fa.params \
	 ${OUTPUT}/${LIG}_${j}.${TAG}.design_ \
	 ${MUTABLE_ATOMS} \
	 ${MEDCHEM_FRAGMENTS} \
	 ${INDEXH} 
done

#########
# R2

LIG=XR2 # missing("X") R2 and ready to be built 
MUTABLE_ATOMS=4 # same per R*
INDEXH=0 # not necessary, but just in case
OUTPUT=out_r2

# note *{sh,xml} are called "r1", but these are general for this med chem application and can be used for r2 and r3 replacements

for ((j=9;j<=15;j++)); do
    
    MEDCHEM_FRAGMENTS=./fragments/R2_${j}.sdf    
    bash mays19jbc.r1.design.sh \
	 mays19jbc.r1.design.xml \
	 ../${PDB}_prot.pdb  \
	 ${LIG}_0001.fa.pdb \
	 ${LIG}.fa.params \
	 ${OUTPUT}/${LIG}_${j}.design_ \
	 ${MUTABLE_ATOMS} \
	 ${MEDCHEM_FRAGMENTS} \
	 ${INDEXH} 
done

#########
# R3

LIG=XR3 
MUTABLE_ATOMS=22 #41 # same per R*
INDEXH=0 # not necessary, but just in case
OUTPUT=out_r3

# note *{sh,xml} are called "r1", but these are general for this med chem application and can be used for r2 and r3 replacements

for ((j=16;j<=23;j++)); do
    
    MEDCHEM_FRAGMENTS=./fragments/R3_${j}.sdf    
    bash mays19jbc.r1.design.sh \
	 mays19jbc.r1.design.xml \
	 ../${PDB}_prot.pdb  \
	 ${LIG}_0001.fa.pdb \
	 ${LIG}.fa.params \
	 ${OUTPUT}/${LIG}_${j}.design_ \
	 ${MUTABLE_ATOMS} \
	 ${MEDCHEM_FRAGMENTS} \
	 ${INDEXH} 
done

echo "Now go to 2.5_* to separate then 3_score/ to rescore with score12" 

##############################
# next: select decoy with lowest IDX from step 2_, process file for scoring 
# GO TO 2.5_process_design.sh
