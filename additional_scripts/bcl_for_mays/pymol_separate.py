import sys
import pymol
from pymol import cmd

# Initialize PyMOL without GUI
pymol.finish_launching(['pymol', '-cq'])

# Load PDB
ifname=sys.argv[1] # contains extension
ofname=sys.argv[2] # does not have extension
outfolder=sys.argv[3] # output folder 

cmd.load(ifname,"decoy")

cmd.select("prot","chain A")
cmd.select("lig","chain X")

cmd.save(outfolder+"/"+ofname+"_prot.pdb","prot")
cmd.save(outfolder+"/"+ofname+"_lig.sdf","lig")

cmd.quit()
