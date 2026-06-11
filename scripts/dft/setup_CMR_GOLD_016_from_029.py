from pathlib import Path
import shutil
import math
import csv

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors


PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")

OLD_ID = "CMR_GOLD_029"
NEW_ID = "CMR_GOLD_016"

SMILES_016 = "O=C1C(C(C(OCC)=O)C2=C(O)C3=CC=CC=C3OC2=O)=C(O)C4=CC=CC=C4O1"

EXPECTED_FORMULA = "C22H16O8"
EXPECTED_N = 0
EXPECTED_TOTAL_ATOMS_WITH_H = 46

OLD_WORKFLOW = PROJECT / "cmr029_workflow.py"
NEW_WORKFLOW = PROJECT / "cmr016_workflow.py"

MOL_DIR = PROJECT / "dft" / "molecules" / NEW_ID
INPUT_DIR = MOL_DIR / "input"
GEOM_DIR = MOL_DIR / "geometry"
CUBES_DIR = MOL_DIR / "cubes"
FIG_DIR = MOL_DIR / "figures"
DATA_DIR = MOL_DIR / "extracted_data"
OUTPUT_DIR = MOL_DIR / "output"

XYZ_RDKIT = INPUT_DIR / f"{NEW_ID}_start_rdkit_mmff.xyz"
XYZ_ORCA = INPUT_DIR / f"{NEW_ID}_start.xyz"
SDF_OUT = INPUT_DIR / f"{NEW_ID}_start_rdkit_mmff.sdf"
SMI_OUT = INPUT_DIR / f"{NEW_ID}.smi"

REPORT_TXT = DATA_DIR / f"{NEW_ID}_start_xyz_validation_report.txt"
REPORT_CSV = DATA_DIR / f"{NEW_ID}_start_xyz_validation_report.csv"


def ensure_dirs():
    for d in [INPUT_DIR, GEOM_DIR, CUBES_DIR, FIG_DIR, DATA_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def create_workflow():
    if not OLD_WORKFLOW.exists():
        raise FileNotFoundError(
            f"{OLD_WORKFLOW} bulunamadı. Önce CMR_GOLD_029 workflow dosyası mevcut olmalı."
        )

    s = OLD_WORKFLOW.read_text(encoding="utf-8")

    s = s.replace("CMR_GOLD_029", "CMR_GOLD_016")

    old_smiles = "O=C1OC2=C(C=CC(OCC(O)CN3C=NC4=C3C=CC=C4)=C2)C(C)=C1"
    s = s.replace(old_smiles, SMILES_016)

    # 32 GB RAM için güvenli ayar
    s = s.replace("NPROCS = 16", "NPROCS = 8")
    s = s.replace("MAXCORE = 4000", "MAXCORE = 3000")
    s = s.replace("NPROCS = 8", "NPROCS = 8")
    s = s.replace("MAXCORE = 3000", "MAXCORE = 3000")

    NEW_WORKFLOW.write_text(s, encoding="utf-8")
    print(f"Workflow oluşturuldu: {NEW_WORKFLOW}")


def min_nonbonded_distance(mol):
    conf = mol.GetConformer()
    bonded = set()

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bonded.add(tuple(sorted((i, j))))

    min_d = 999.0
    min_pair = None

    n = mol.GetNumAtoms()
    for i in range(n):
        pi = conf.GetAtomPosition(i)
        for j in range(i + 1, n):
            if tuple(sorted((i, j))) in bonded:
                continue

            pj = conf.GetAtomPosition(j)
            d = math.sqrt((pi.x - pj.x) ** 2 + (pi.y - pj.y) ** 2 + (pi.z - pj.z) ** 2)

            if d < min_d:
                min_d = d
                min_pair = (i, j)

    return min_d, min_pair


def write_xyz(mol, path):
    conf = mol.GetConformer()

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{NEW_ID} RDKit ETKDGv3 + MMFF94 starting geometry; neutral singlet\n")

        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            sym = atom.GetSymbol()
            f.write(f"{sym:<2s} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}\n")


def create_start_geometry():
    mol0 = Chem.MolFromSmiles(SMILES_016)

    if mol0 is None:
        raise RuntimeError("SMILES okunamadı.")

    Chem.SanitizeMol(mol0)

    canonical_smiles = Chem.MolToSmiles(mol0)
    formula = rdMolDescriptors.CalcMolFormula(mol0)
    exact_mw = Descriptors.ExactMolWt(mol0)
    formal_charge = Chem.GetFormalCharge(mol0)

    n_count = sum(1 for a in mol0.GetAtoms() if a.GetAtomicNum() == 7)
    o_count = sum(1 for a in mol0.GetAtoms() if a.GetAtomicNum() == 8)
    heavy_atoms = mol0.GetNumHeavyAtoms()

    mol = Chem.AddHs(mol0)
    total_atoms = mol.GetNumAtoms()

    params = AllChem.ETKDGv3()
    params.randomSeed = 16016
    params.pruneRmsThresh = 0.20
    params.numThreads = 0
    params.useRandomCoords = False

    # Dimerik yapı olduğu için 029/044'e göre daha fazla konformer deniyoruz.
    n_confs_requested = 250
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs_requested, params=params))

    if not conf_ids:
        raise RuntimeError("3D conformer üretilemedi.")

    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise RuntimeError("MMFF parametreleri eksik. UFF fallback gerekir.")

    mmff_results = AllChem.MMFFOptimizeMoleculeConfs(
        mol,
        numThreads=0,
        maxIters=1500,
        mmffVariant="MMFF94"
    )

    energies = []
    for conf_id, result in zip(conf_ids, mmff_results):
        not_converged, energy = result
        energies.append((conf_id, not_converged, energy))

    best_conf_id, best_not_converged, best_energy = sorted(energies, key=lambda x: x[2])[0]

    best_mol = Chem.Mol(mol)
    best_conf = mol.GetConformer(best_conf_id)
    best_mol.RemoveAllConformers()
    best_mol.AddConformer(best_conf, assignId=True)

    write_xyz(best_mol, XYZ_RDKIT)
    shutil.copy2(XYZ_RDKIT, XYZ_ORCA)

    writer = Chem.SDWriter(str(SDF_OUT))
    writer.write(best_mol)
    writer.close()

    SMI_OUT.write_text(SMILES_016 + "\n", encoding="utf-8")

    min_nb_dist, min_pair = min_nonbonded_distance(best_mol)

    coords_ok = True
    conf = best_mol.GetConformer()
    for i in range(best_mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        if not all(math.isfinite(v) for v in [p.x, p.y, p.z]):
            coords_ok = False

    checks = {
        "Molecule_ID": NEW_ID,
        "SMILES_valid": True,
        "Input_SMILES": SMILES_016,
        "Canonical_SMILES": canonical_smiles,
        "Formula": formula,
        "Exact_MW": f"{exact_mw:.6f}",
        "Formal_charge": formal_charge,
        "Heavy_atoms_no_H": heavy_atoms,
        "Total_atoms_with_H": total_atoms,
        "N_count": n_count,
        "O_count": o_count,
        "Conformers_requested": n_confs_requested,
        "Conformers_generated": len(conf_ids),
        "Best_conformer_ID": best_conf_id,
        "MMFF_not_converged_flag": best_not_converged,
        "MMFF_energy_kcal_mol_like": f"{best_energy:.6f}",
        "Min_nonbonded_distance_A": f"{min_nb_dist:.4f}",
        "Min_nonbonded_pair_atom_indices_0based": str(min_pair),
        "Coordinates_finite": coords_ok,
        "XYZ_RDKIT_written": XYZ_RDKIT.exists(),
        "XYZ_ORCA_written": XYZ_ORCA.exists(),
        "SDF_written": SDF_OUT.exists(),
        "Expected_formula_match": formula == EXPECTED_FORMULA,
        "Expected_N_count_match": n_count == EXPECTED_N,
        "Expected_total_atom_count_match": total_atoms == EXPECTED_TOTAL_ATOMS_WITH_H,
    }

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{NEW_ID} RDKit-MMFF starting geometry validation report\n")
        f.write("=" * 72 + "\n\n")

        for k, v in checks.items():
            f.write(f"{k}: {v}\n")

        f.write("\nExpected key checks:\n")
        f.write(f"- Formula should be {EXPECTED_FORMULA}\n")
        f.write("- Formal charge should be 0\n")
        f.write("- N_count should be 0\n")
        f.write(f"- Total atoms with explicit H should be {EXPECTED_TOTAL_ATOMS_WITH_H}\n")
        f.write("- Coordinates_finite should be True\n")
        f.write("- Min_nonbonded_distance_A should not be suspiciously small\n")

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Field", "Value"])

        for k, v in checks.items():
            writer.writerow([k, v])

    print("\nCMR_GOLD_016 başlangıç geometrisi oluşturuldu ve doğrulandı.")
    print(f"XYZ RDKit-MMFF : {XYZ_RDKIT}")
    print(f"XYZ ORCA      : {XYZ_ORCA}")
    print(f"SDF           : {SDF_OUT}")
    print(f"Rapor TXT     : {REPORT_TXT}")
    print(f"Rapor CSV     : {REPORT_CSV}")

    print("\nÖzet doğrulama:")
    print(f"Formula                 : {formula}")
    print(f"Formal charge            : {formal_charge}")
    print(f"N count                  : {n_count}")
    print(f"O count                  : {o_count}")
    print(f"Heavy atoms              : {heavy_atoms}")
    print(f"Total atoms with H       : {total_atoms}")
    print(f"Generated conformers     : {len(conf_ids)}")
    print(f"Best MMFF energy         : {best_energy:.6f}")
    print(f"Min nonbonded distance   : {min_nb_dist:.4f} Å")
    print(f"Coordinates finite       : {coords_ok}")

    if formula != EXPECTED_FORMULA:
        print(f"\nUYARI: Formula beklenen {EXPECTED_FORMULA} ile uyuşmuyor.")
    if formal_charge != 0:
        print("\nUYARI: Molekül nötral değil.")
    if n_count != EXPECTED_N:
        print("\nUYARI: N count 0 değil.")
    if total_atoms != EXPECTED_TOTAL_ATOMS_WITH_H:
        print(f"\nUYARI: Explicit H ile toplam atom sayısı {EXPECTED_TOTAL_ATOMS_WITH_H} değil.")
    if min_nb_dist < 0.75:
        print("\nUYARI: Çok kısa nonbonded mesafe var; geometri görsel kontrol edilmeli.")
    if not coords_ok:
        print("\nUYARI: Koordinatlarda geçersiz sayı var.")


def main():
    ensure_dirs()
    create_workflow()
    create_start_geometry()

    print("\nSonraki komutlar:")
    print("python cmr016_workflow.py prepare")
    print("python cmr016_workflow.py status")
    print("python cmr016_workflow.py run-opt")


if __name__ == "__main__":
    main()