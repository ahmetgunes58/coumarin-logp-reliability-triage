from pathlib import Path
import shutil

ROOT = Path(r"D:\Makaleler\coumarin-logp-working-source")

TARGET_IDS = ["CMR_GOLD_090", "CMR_GOLD_020", "CMR_GOLD_092"]

for cid in TARGET_IDS:
    mol_dir = ROOT / "dft" / "molecules" / cid
    input_dir = mol_dir / "input"
    cube_dir = mol_dir / "cube"
    cube_input_dir = cube_dir / "input"

    cube_dir.mkdir(parents=True, exist_ok=True)
    cube_input_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{cid}_sp"

    # ORCA/orca_plot bazen densitiesinfo dosyasını input\<file>.densitiesinfo şeklinde arıyor.
    files_to_copy = [
        f"{stem}.gbw",
        f"{stem}.densities",
        f"{stem}.densitiesinfo",
        f"{stem}.property.txt",
        f"{stem}.cpcm",
        f"{stem}.cpcm_corr",
    ]

    print(f"\n{cid}")

    for fname in files_to_copy:
        src = input_dir / fname
        if not src.exists():
            print(f"  MISSING: {src}")
            continue

        dst_root = cube_dir / fname
        shutil.copy2(src, dst_root)
        print(f"  copied to cube root : {dst_root}")

        dst_input = cube_input_dir / fname
        shutil.copy2(src, dst_input)
        print(f"  copied to cube/input: {dst_input}")

    xyz_src = mol_dir / "geometry" / f"{cid}_opt.xyz"
    if xyz_src.exists():
        xyz_dst = cube_dir / f"{stem}.xyz"
        shutil.copy2(xyz_src, xyz_dst)
        print(f"  copied xyz          : {xyz_dst}")
    else:
        print(f"  MISSING XYZ: {xyz_src}")

print("\nPatch tamamlandı.")