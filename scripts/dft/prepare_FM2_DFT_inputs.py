# -*- coding: utf-8 -*-
"""
Prepare DFT input package for three additional FM2 anchors:
CMR_GOLD_090, CMR_GOLD_020, CMR_GOLD_092

Protocol matched to existing DFT panel:
Opt/Freq: B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water) Opt Freq
SP:       B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water)

Run from project root:
    cd /d D:\Makaleler\coumarin-logp-working-source
    python prepare_FM2_DFT_inputs.py
"""

from pathlib import Path
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")
DATASET = ROOT / "data" / "processed" / "Dataset_S1_benchmark_dataset.csv"

TARGET_IDS = ["CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"]

DFT_ROOT = ROOT / "dft" / "molecules"
SUMMARY_OUT = ROOT / "data" / "processed" / "Dataset_S29_FM2_DFT_selected_anchors.csv"


OPTFREQ_TEMPLATE = """! B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water) Opt Freq

%pal
  nprocs 8
end

%maxcore 3000

%scf
  MaxIter 500
end

%output
  Print[P_Mulliken] 1
  Print[P_Loewdin] 1
end

* xyzfile 0 1 input/{cid}_start.xyz
"""

SP_TEMPLATE = """! B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water)

%pal
  nprocs 8
end

%maxcore 3000

%scf
  MaxIter 500
end

%output
  Print[P_Mulliken] 1
  Print[P_Loewdin] 1
end

* xyzfile 0 1 geometry/{cid}_opt.xyz
"""


def mol_to_xyz(smiles: str, cid: str, out_xyz: Path) -> None:
    """Generate a reasonable 3D starting geometry using RDKit ETKDG + MMFF/UFF."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES okunamadı: {cid} | {smiles}")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 20260604
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(f"3D embedding başarısız: {cid}")

    # Prefer MMFF if parameters are available, otherwise fallback to UFF.
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
        method = "MMFF94"
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
        method = "UFF"

    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), pos.x, pos.y, pos.z))

    with out_xyz.open("w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{cid} start geometry generated from SMILES using RDKit ETKDGv3 + {method}\n")
        for sym, x, y, z in atoms:
            f.write(f"{sym:<2s} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset bulunamadı: {DATASET}")

    df = pd.read_csv(DATASET)
    required = ["Compound_ID", "SMILES", "FM", "N_count", "logP_exp", "Consensus", "delta_Consensus", "TPSA", "MW"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Eksik kolonlar: {missing}")

    selected = df[df["Compound_ID"].isin(TARGET_IDS)].copy()
    if len(selected) != len(TARGET_IDS):
        found = set(selected["Compound_ID"])
        missing_ids = [cid for cid in TARGET_IDS if cid not in found]
        raise ValueError(f"Eksik hedef compound ID: {missing_ids}")

    # Preserve target order
    selected["__order"] = selected["Compound_ID"].map({cid: i for i, cid in enumerate(TARGET_IDS)})
    selected = selected.sort_values("__order").drop(columns="__order")

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")

    created = []

    for _, row in selected.iterrows():
        cid = row["Compound_ID"]
        smiles = row["SMILES"]

        mol_dir = DFT_ROOT / cid
        input_dir = mol_dir / "input"
        geometry_dir = mol_dir / "geometry"
        output_dir = mol_dir / "output"
        extracted_dir = mol_dir / "extracted_data"
        cube_dir = mol_dir / "cube"
        figures_dir = mol_dir / "figures"

        for d in [input_dir, geometry_dir, output_dir, extracted_dir, cube_dir, figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

        start_xyz = input_dir / f"{cid}_start.xyz"
        optfreq_inp = input_dir / f"{cid}_optfreq.inp"
        sp_inp = input_dir / f"{cid}_sp.inp"

        mol_to_xyz(smiles, cid, start_xyz)
        write_text(optfreq_inp, OPTFREQ_TEMPLATE.format(cid=cid))
        write_text(sp_inp, SP_TEMPLATE.format(cid=cid))

        # Per-molecule run helper.
        run_bat = mol_dir / f"run_{cid}_orca_jobs.bat"
        run_text = f"""@echo off
REM Run from this molecule directory.
REM 1) Run Opt/Freq first.
REM 2) After Opt/Freq finishes, copy or extract the optimised xyz to geometry\\{cid}_opt.xyz.
REM 3) Then run SP.

cd /d "{mol_dir}"

echo Running Opt/Freq for {cid}
orca input\\{cid}_optfreq.inp > output\\{cid}_optfreq.out

echo.
echo Opt/Freq finished. Now make sure geometry\\{cid}_opt.xyz exists before running SP.
echo ORCA may write an optimised xyz file depending on run location/settings.
echo If needed, extract the final optimised geometry and save it as:
echo geometry\\{cid}_opt.xyz
echo.

pause

echo Running SP for {cid}
orca input\\{cid}_sp.inp > output\\{cid}_sp.out

echo Done.
pause
"""
        write_text(run_bat, run_text)

        created.append({
            "Compound_ID": cid,
            "Molecule_dir": str(mol_dir),
            "Start_XYZ": str(start_xyz),
            "OptFreq_INP": str(optfreq_inp),
            "SP_INP": str(sp_inp),
            "Run_BAT": str(run_bat),
        })

    created_df = pd.DataFrame(created)
    created_csv = ROOT / "data" / "processed" / "Dataset_S29_FM2_DFT_input_file_manifest.csv"
    created_df.to_csv(created_csv, index=False, encoding="utf-8-sig")

    print("\nFM2 DFT input preparation tamamlandı.")
    print(f"Selected anchor metadata: {SUMMARY_OUT}")
    print(f"Input file manifest    : {created_csv}")
    print("\nSelected anchors:")
    print(selected[["Compound_ID", "FM", "N_count", "logP_exp", "Consensus", "delta_Consensus", "TPSA", "MW"]].to_string(index=False))
    print("\nCreated files:")
    print(created_df.to_string(index=False))


if __name__ == "__main__":
    main()