#!/bin/bash

# Process designs from 2_bcl_modify_ligand.sh
# Will need to update scoring terms to match with protocol used for alip-l and blip-l features.
# When I ran this script, I executed each R* replacement chunk separately

# These should be the same as variables set in 2_bcl_modify_ligand.sh 
PDB=5l11

#LIG=XR1
#TAG=exo
##TAG=endo
#OUTPUT=out_r1${TAG}

#LIG=XR2
#OUTPUT=out_r2

LIG=XR3
OUTPUT=out_r3

#for ((j=1;j<=8;j++)); do # for R1
#for ((j=9;j<=15;j++)); do  # for R2

for ((j=16;j<=23;j++)); do  # for R3
    # Select decoy with lowest IDX    
    scfile=${OUTPUT}/${LIG}_$j.design_score.sc 
    idx=`awk '{for(k=1;k<=NF;k++) if($k=="interface_delta_X") print k}' ${scfile}`
    desc=`awk '{for(k=1;k<=NF;k++) if($k=="description") print k}' ${scfile}`
    best=$(sort -nk ${idx} ${scfile} | head -n 1 | awk -v col="${desc}" '{print $col}').pdb.gz
    echo $best

    # Separate to *_prot.pdb and *_lig.sdf & save to shortened name
    source /your/path/to/etc/profile.d/conda.sh # if you need to activate conda env 
    conda activate python3 # move this to a python script
    #outname=${PDB}_${LIG}_${TAG}_${j} # for R1 
    outname=${PDB}_${LIG}_${j}  # for R2 and R3 
    python3.8 pymol_separate.py ${best} ${outname} ${OUTPUT}
    conda deactivate 
    
done

##############################
# next: execute scoring 
# GO TO 3_score_pose.sh 
