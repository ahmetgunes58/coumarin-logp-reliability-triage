from pathlib import Path
import shutil

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")

SELECTED = {
    "CMR_GOLD_043": {
        "opt_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\01_opt_freq_correct\CMR_GOLD_043_correct_opt_pal8.out"),
        "sp_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\043\02_sp_charge_correct\CMR_GOLD_043_sp_charge_clean.out"),
    },
    "CMR_GOLD_055": {
        "opt_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\055\01_opt_freq\CMR_GOLD_055_correct_opt_pal8_clean.out"),
        "sp_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\055\02_sp_charge\CMR_GOLD_055_sp_charge_clean.out"),
    },
    "CMR_GOLD_058": {
        "opt_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\058\01_opt_freq\CMR_GOLD_058_correct_opt_pal8.out"),
        "sp_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\058\02_sp_charge\CMR_GOLD_058_sp_charge_clean.out"),
    },
    "CMR_GOLD_079": {
        "opt_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\079\01_opt_freq\CMR_GOLD_079_correct_opt_pal8.out"),
        "sp_out": Path(r"C:\orca_tests\DFT_FINAL_PROPERTIES\079\02_sp_charge\CMR_GOLD_079_sp_charge_clean.out"),
    },
}


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"YOK: {src}")
        return False
    shutil.copy2(src, dst)
    print(f"Kopyalandı: {src} -> {dst}")
    return True


def main():
    for mol_id, files in SELECTED.items():
        mol_dir = PROJECT / "dft" / "molecules" / mol_id
        out_dir = mol_dir / "output"
        audit_dir = mol_dir / "audit_source"

        out_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print(mol_id)
        print("=" * 90)

        opt_dst = out_dir / f"{mol_id}_optfreq_existing.out"
        sp_dst = out_dir / f"{mol_id}_sp_existing.out"

        copy_file(files["opt_out"], opt_dst)
        copy_file(files["sp_out"], sp_dst)

        source_note = audit_dir / f"{mol_id}_selected_source_paths.txt"
        source_note.write_text(
            f"{mol_id} selected existing DFT source files\n"
            f"{'=' * 80}\n\n"
            f"OPT/FREQ source:\n{files['opt_out']}\n\n"
            f"SP source:\n{files['sp_out']}\n\n"
            f"Copied opt/freq output:\n{opt_dst}\n\n"
            f"Copied SP output:\n{sp_dst}\n",
            encoding="utf-8"
        )
        print(f"Kaynak notu yazıldı: {source_note}")

    print("\nBitti. Seçilen eski DFT output dosyaları yeni proje klasörüne kopyalandı.")


if __name__ == "__main__":
    main()