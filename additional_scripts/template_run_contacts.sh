#!/bin/bash

# Read RosettaLigand poses PDB, and analyze w cpptraj script
# Need cpptraj, pdb4amber

cpin=cptemp.in

pardir=/your/path/to/docked_poses_pdb/
incsv=/your/path/to/a/csv/file/with/energy_terms/like/what/is/generated/from/bcl_for_mays/3.5_get_eterms.sh

tail -n +2 $incsv | while IFS=, read -r xx ID xx xx xx xx xx xx xx xx xx description xx ; do 

    # csv files contain vuid, energy terms, and pdb pose name     
    # pdb poses are named something like: xtal_0001_algn_s2k_00_0000_ros_0095

    molid=$(grep -o 's2k_[0-9]\{2\}_[0-9]\{4\}' <<< "$description")  # extract s2k_XX_YYYY    
    prefix=${molid%_*}
    pdb="${pardir}/${prefix}/${molid}/IUWtail_7tt8/${description}.pdb.gz"
    
    if [ ! -f "${pdb}" ]; then
	echo "File not found."
	continue
    fi

    echo "Processing $ID"
    short="${ID}"
    
    pdb4amber -i $pdb -o ${short}.pdb
    
    rm -rf *sslink
    rm -rf *nonprot.pdb
    rm -rf *_renum.txt 

    ##############################
    # for cpptraj: write compound-specific script
    ## use this to get contacts of all compounds 
    
    # write contents to file 
    rm -rf $cpin
cat <<EOF > "$cpin"
 	parm       ${short}.pdb
	trajin     ${short}.pdb
	nativecontacts name CONT :241&!@H=  :1-240&!@H= \
	    distance 3.0 \
	    writecontacts ${short}_cont.dat \	       
	    resout ${short}_res.dat \ 
	    byresidue 
	run	
	hbond :1-241 out ${short}_nhb.dat avgout ${short}_avghb.dat nointramol
	run 
EOF

    # execute ccptraj 
    cpptraj $cpin

done 

rm -rf $cpin
