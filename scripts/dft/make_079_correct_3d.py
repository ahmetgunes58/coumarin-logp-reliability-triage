# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors, Descriptors

SMILES = "O=C(NCC1=CN(C(C2=O)=CC3=C(O2)C=CC=C3)N=N1)C4=CC=CC=C4O"

OUT_XYZ = "CMR_GOLD_079_correct_3D.xyz"
OUT_SDF = "CMR_GOLD_079_correct_3D.sdf"
OUT_TXT = "CMR_GOLD_079_correct_3D_check.txt"
OUT_PNG = "CMR_GOLD_079_structure_check.png"

EXPECTED_N_COUNT = 4

mol = Chem.MolFromSmiles(SMILES)
if mol is None:
    raise RuntimeError("SMILES could not be parsed.")

canonical = Chem.MolToSmiles(mol)
formula = rdMolDescriptors.CalcMolFormula(mol)
exact_mass = Descriptors.ExactMolWt(mol)
heavy_atoms = mol.GetNumHeavyAtoms()
n_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "N")

if n_count != EXPECTED_N_COUNT:
    raise RuntimeError(
        "N-count mismatch. Expected {}, found {}.".format(EXPECTED_N_COUNT, n_count)
    )

mol_h = Chem.AddHs(mol)

params = AllChem.ETKDGv3()
params.randomSeed = 42
params.useSmallRingTorsions = True
params.pruneRmsThresh = 0.20

print("Embedding conformers...")
conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=100, params=params))
print("Generated conformers:", len(conf_ids))

if not conf_ids:
    raise RuntimeError("Conformer generation failed.")

print("Optimizing conformers with MMFF94s...")

props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
energies = []

if props is not None:
    for cid in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(
                mol_h,
                mmffVariant="MMFF94s",
                confId=cid,
                maxIters=1000
            )
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=cid)
            energies.append((ff.CalcEnergy(), cid))
        except Exception:
            pass
else:
    print("MMFF not available. Falling back to UFF.")
    for cid in conf_ids:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, confId=cid, maxIters=1000)
            ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
            energies.append((ff.CalcEnergy(), cid))
        except Exception:
            pass

if not energies:
    raise RuntimeError("No conformer could be optimized.")

energies.sort()
best_energy, best_cid = energies[0]

writer = Chem.SDWriter(OUT_SDF)
mol_h.SetProp("_Name", "CMR_GOLD_079_correct_3D")
mol_h.SetProp("Canonical_SMILES", canonical)
mol_h.SetProp("Formula", formula)
mol_h.SetProp("N_count", str(n_count))
mol_h.SetProp("Best_MMFF_or_UFF_energy", str(best_energy))
writer.write(mol_h, confId=best_cid)
writer.close()

conf = mol_h.GetConformer(best_cid)

with open(OUT_XYZ, "w") as f:
    f.write(str(mol_h.GetNumAtoms()) + "\n")
    f.write("CMR_GOLD_079 correct 3D; generated from verified SMILES; best conformer {}\n".format(best_cid))
    for atom in mol_h.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        f.write(
            "{:<2s} {:16.8f} {:16.8f} {:16.8f}\n".format(
                atom.GetSymbol(), pos.x, pos.y, pos.z
            )
        )

Draw.MolToFile(mol, OUT_PNG, size=(1400, 900))

with open(OUT_TXT, "w") as f:
    f.write("CMR_GOLD_079 structure verification\n")
    f.write("=" * 72 + "\n")
    f.write("Input SMILES      : {}\n".format(SMILES))
    f.write("Canonical SMILES  : {}\n".format(canonical))
    f.write("Formula           : {}\n".format(formula))
    f.write("Exact mass        : {:.6f}\n".format(exact_mass))
    f.write("Heavy atoms       : {}\n".format(heavy_atoms))
    f.write("Total atoms incl H: {}\n".format(mol_h.GetNumAtoms()))
    f.write("N atom count      : {}\n".format(n_count))
    f.write("Expected N count  : {}\n".format(EXPECTED_N_COUNT))
    f.write("Best conformer ID : {}\n".format(best_cid))
    f.write("Best energy       : {}\n".format(best_energy))
    f.write("\nIdentity check: PASS\n")

print("=" * 72)
print("DONE")
print("Identity check: PASS")
print("Formula:", formula)
print("N atom count:", n_count)
print("Total atoms incl H:", mol_h.GetNumAtoms())
print("Best conformer ID:", best_cid)
print("Best energy:", best_energy)
print("Written:", OUT_XYZ)
print("Written:", OUT_SDF)
print("Written:", OUT_TXT)
print("Written:", OUT_PNG)
print("=" * 72)