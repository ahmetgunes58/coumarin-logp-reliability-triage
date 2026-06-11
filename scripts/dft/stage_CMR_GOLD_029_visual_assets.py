from pathlib import Path
import shutil
import subprocess

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID

INPUT_DIR = MOL_DIR / "input"
GEOM_DIR = MOL_DIR / "geometry"
CUBES_DIR = MOL_DIR / "cubes"
FIG_DIR = MOL_DIR / "figures"
DATA_DIR = MOL_DIR / "extracted_data"

SP_GBW_IN = INPUT_DIR / f"{MOL_ID}_sp.gbw"
OPT_GBW_IN = INPUT_DIR / f"{MOL_ID}_optfreq.gbw"
OPT_XYZ = GEOM_DIR / f"{MOL_ID}_opt.xyz"

SP_GBW_GEOM = GEOM_DIR / f"{MOL_ID}_sp.gbw"
OPT_GBW_GEOM = GEOM_DIR / f"{MOL_ID}_optfreq.gbw"

ORCA_2MKL = Path(r"C:\orca\orca_2mkl.exe")
ORCA_PLOT = Path(r"C:\orca\orca_plot.exe")


def ensure_dirs():
    for d in [GEOM_DIR, CUBES_DIR, FIG_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def copy_if_needed(src, dst):
    if not src.exists():
        raise FileNotFoundError(f"Eksik dosya: {src}")
    shutil.copy2(src, dst)
    print(f"Kopyalandı: {src} -> {dst}")


def run_orca_2mkl():
    if not ORCA_2MKL.exists():
        raise FileNotFoundError(f"orca_2mkl bulunamadı: {ORCA_2MKL}")

    # geometry klasöründe çalıştırıyoruz
    cmd = [str(ORCA_2MKL), f"{MOL_ID}_sp", "-molden"]
    print("Çalıştırılıyor:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(GEOM_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    print(result.stdout)

    out1 = GEOM_DIR / f"{MOL_ID}_sp.molden.input"
    out2 = GEOM_DIR / f"{MOL_ID}_sp.molden"

    if out1.exists():
        print(f"Molden dosyası oluştu: {out1}")
    elif out2.exists():
        print(f"Molden dosyası oluştu: {out2}")
    else:
        print("UYARI: Molden output beklenen isimle görünmedi. geometry klasörünü kontrol et.")


def main():
    ensure_dirs()

    if not SP_GBW_IN.exists():
        raise FileNotFoundError(f"SP GBW dosyası yok: {SP_GBW_IN}")
    if not OPT_GBW_IN.exists():
        raise FileNotFoundError(f"Opt GBW dosyası yok: {OPT_GBW_IN}")
    if not OPT_XYZ.exists():
        raise FileNotFoundError(f"Optimize XYZ dosyası yok: {OPT_XYZ}")

    print("\n=== DOSYA KONTROL ===")
    print(f"SP GBW   : {SP_GBW_IN}")
    print(f"OPT GBW  : {OPT_GBW_IN}")
    print(f"OPT XYZ  : {OPT_XYZ}")
    print(f"orca_plot: {ORCA_PLOT.exists()} -> {ORCA_PLOT}")
    print(f"orca_2mkl: {ORCA_2MKL.exists()} -> {ORCA_2MKL}")

    print("\n=== KOPYALAMA ===")
    copy_if_needed(SP_GBW_IN, SP_GBW_GEOM)
    copy_if_needed(OPT_GBW_IN, OPT_GBW_GEOM)

    print("\n=== MOLDEN ÜRETİMİ ===")
    run_orca_2mkl()

    print("\n=== SON DURUM ===")
    for p in [
        SP_GBW_GEOM,
        OPT_GBW_GEOM,
        OPT_XYZ,
        GEOM_DIR / f"{MOL_ID}_sp.molden.input",
        GEOM_DIR / f"{MOL_ID}_sp.molden",
    ]:
        print(f"{p.name:<35} {'OK' if p.exists() else 'YOK'}")

    print("\nHazırlık tamamlandı.")
    print("Bir sonraki adım: density/ESP cube üretimi ve MEP workflow.")


if __name__ == "__main__":
    main()