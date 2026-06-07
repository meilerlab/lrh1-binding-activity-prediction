#!/bin/bash

# Extract relevant energy terms from score.sc 
# To execute: bash 3.5_get_eterms.sh <PDB>

PDB=$1

fout=out_eterms_${PDB}.csv
echo "make sure to copy temp.csv part to new file"
if [ -z $PDB ]; then 
    echo "Error: specify out_tag"
    exit 1
fi

# prep header of fout
scfile=score.sc

desc=`awk '{for(k=1;k<=NF;k++) if($k=="description") print k}' ${scfile}`
atr=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_fa_atr") print k}' ${scfile}`
elec=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_fa_elec") print k}' ${scfile}`
rep=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_fa_rep") print k}' ${scfile}`
sol=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_fa_sol") print k}' ${scfile}`
hbb=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_hbond_bb_sc") print k}' ${scfile}`
hsc=`awk '{for(k=1;k<=NF;k++) if($k=="if_X_hbond_sc") print k}' ${scfile}`
totx=`awk '{for(k=1;k<=NF;k++) if($k=="total_score_X") print k}' ${scfile}` 
idx=`awk '{for(k=1;k<=NF;k++) if($k=="interface_delta_X") print k}' ${scfile}`
tot=`awk '{for(k=1;k<=NF;k++) if($k=="total_score") print k}' ${scfile}`

awk -v desc="${desc}" -v atr="${atr}" -v elec="${elec}" -v rep="${rep}" -v sol="${sol}" -v hbb="${hbb}" -v hsc="${hsc}" -v totx="${totx}" -v idx="${idx}" -v tot="${tot}" 'NR > 1 {print $desc "," $atr "," $elec "," $rep "," $sol "," $hbb "," $hsc "," $totx "," $idx "," $tot }' $scfile > $fout

##############################
# Now you have the energy terms needed by blip-l and alip-l to get binder and activity predictions!
