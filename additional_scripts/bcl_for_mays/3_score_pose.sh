#!/bin/bash

# Score best pose model
# Update scoring terms to match with protocol used for alip-l and blip-l features
# When I ran this script, I executed each R* replacement chunk separately

ROSETTA=/your/rosetta/path/to/main/

PDB=5l11

#LIG=XR1
#TAG=exo 
##TAG=endo
#OUTPUT=../out_r1${TAG}

#LIG=XR2
#OUTPUT=../out_r2

LIG=XR3
OUTPUT=../out_r3

#for ((j=1;j<=8;j++)); do # for R1
#for ((j=9;j<=15;j++)); do # for R2
for ((j=16;j<=23;j++)); do # for R3 

    #ifname=${PDB}_${LIG}_${TAG}_${j} # for R1
    ifname=${PDB}_${LIG}_${j}  # for R2
    insdf=${OUTPUT}/${ifname}_lig.sdf
    inpdb=${OUTPUT}/${ifname}_prot.pdb

    #####
    # need to add hydrogens 
    MOLH=temp.sdf
    rm -rf $MOLH
    obabel ${insdf} -O $MOLH -h

    # get roseta params file. need to get diff pdbs just in case
    NAME='DUM'
    rm -rf ${NAME}.params ${NAME}_0001.pdb 

    ${ROSETTA}/source/scripts/python/public/molfile_to_params.py -n ${NAME} -p ${NAME} ${MOLH} 

    MOL_params=${NAME}.params
    MOL=${NAME}_0001.pdb

    ${ROSETTA}/source/bin/score_jd2.linuxgccrelease \
	      -in:file:s "$inpdb $MOL" -out:pdb -extra_res_fa ${MOL_params}  \
	      -score:weights score12 -mistakes:restore_pre_talaris_2013_behavior \
	      -out:file:scorefile jd2score.sc

    out_jd2=${ifname}_prot_DUM_0001_0001.pdb

    ${ROSETTA}/source/bin/relax.linuxgccrelease \
	      -relax:constrain_relax_to_start_coords -relax:fast \
	      -score:weights score12  -mistakes:restore_pre_talaris_2013_behavior \
	      -s ${out_jd2}  -extra_res_fa ${MOL_params} \
	      -nstruct 10 \
	      -out:file:scorefile miniscore.sc \
	      -overwrite

    # select best relaxed pdb
    min_pdb=$(echo `sort -nk 2 miniscore.sc | head -n 1 | awk '{print $22}'`).pdb
    mv ${min_pdb} ${ifname}.pdb
    min_pdb=${ifname}.pdb
    
    # get IDX score 
    ${ROSETTA}/source/bin/rosetta_scripts.default.linuxgccrelease \
	      -in:file:s "$min_pdb" \
	      -parser:protocol score.xml \
	      -extra_res_fa ${MOL_params} \
	      -score:weights score12 -mistakes:restore_pre_talaris_2013_behavior \
	      -pdb_gz true \
	      -overwrite

    savepdb="${min_pdb%.pdb}_0001.pdb.gz"

    mv ${savepdb} outpdb/.
    rm -rf miniscore.sc jd2score.sc *DUM* temp.sdf ${min_pdb}

done

##############################
# next: get energy terms for alip-l and blip-l model predictions
# GO TO 3.5_get_eterms.sh


  
