# drugfilters.py
# module to define drug-likeness filters 

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import *

##############################

# initialize filter - PAINS
def checkPAINS(molecule):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    entry = catalog.GetFirstMatch(molecule) # = NONE if not PAINS. else, yes PAINS
    return entry

def checkFilter(molecule): 

    results = {
        "Lipinski Rule of 5": 0,
        "Ghose Filter": 0,
        "Veber Filter": 0,
        "Rule of 3 Filter": 0,
        "REOS Filter": 0,
        "Drug-like Filter": 0,
        "Passes All Filters": 0,
    }

    matches=[]

    ## other filters
    #print(Descriptors.ExactMolWt(molecule))
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)
    h_bond_donor = Descriptors.NumHDonors(molecule)
    h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
    rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
    number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(molecule)
    molar_refractivity = Chem.Crippen.MolMR(molecule)
    topological_surface_area_mapping = Chem.QED.properties(molecule).PSA
    formal_charge = Chem.rdmolops.GetFormalCharge(molecule)
    heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(molecule)
    num_of_rings = Chem.rdMolDescriptors.CalcNumRings(molecule)

    # Lipinski
    if molecular_weight <= 500 and logp <= 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and rotatable_bonds <= 5:
        lipinski = True
        matches.append("lipinski, ")
        results["Lipinski Rule of 5"] += 1
        
    # Ghose Filter
    if molecular_weight >= 160 and molecular_weight <= 480 and logp >= -0.4 and logp <= 5.6 and number_of_atoms >= 20 and number_of_atoms <= 70 and molar_refractivity >= 40 and molar_refractivity <= 130:
        ghose_filter = True
        matches.append("ghose,")
        results["Ghose Filter"] += 1

    # Veber Filter
    if rotatable_bonds <= 10 and topological_surface_area_mapping <= 140:
        veber_filter = True
        matches.append("verber,")
        results["Veber Filter"] += 1

    # Rule of 3
    if molecular_weight <= 300 and logp <= 3 and h_bond_donor <= 3 and h_bond_acceptors <= 3 and rotatable_bonds <= 3:
        rule_of_3 = True
        matches.append("rule of 3,")
        results["Rule of 3 Filter"] += 1

    # REOS Filter
    if molecular_weight >= 200 and molecular_weight <= 500 and logp >= int(0 - 5) and logp <= 5 and h_bond_donor >= 0 and h_bond_donor <= 5 and h_bond_acceptors >= 0 and h_bond_acceptors <= 10 and formal_charge >= int(0-2) and formal_charge <= 2 and rotatable_bonds >= 0 and rotatable_bonds <= 8 and heavy_atoms >= 15 and heavy_atoms <= 50:
        reos_filter = True
        matches.append("REOS,")
        results["REOS Filter"] += 1

    #Drug Like Filter
    if molecular_weight < 400 and num_of_rings > 0 and rotatable_bonds < 5 and h_bond_donor <= 5 and h_bond_acceptors <= 10 and logp < 5:
        drug_like_filter = True
        matches.append("drug-like")
        results["Drug-like Filter"] += 1

    return matches

# end checkFilter
