from pathlib import Path
import shutil
import subprocess
import time

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID
INPUT_DIR = MOL_DIR / "input"
GEOM_DIR = MOL_DIR / "geometry"
CUBES_DIR = MOL_DIR / "cubes"
DATA_DIR = MOL_DIR / "extracted_data"

ORCA_PLOT = Path(r"C:\orca\orca_plot.exe")

STEM = f"{MOL_ID}_sp"

GBW_INPUT = INPUT_DIR / f"{STEM}.gbw"
DENS_INPUT = INPUT_DIR / f"{STEM}.densities"

GBW_GEOM = GEOM_DIR / f"{STEM}.gbw"
DENS_GEOM = GEOM_DIR / f"{STEM}.densities"

DENSITY_CUBE_FINAL = CUBES_DIR / f"{MOL_ID}_density.cube"
ESP_CUBE_FINAL = CUBES_DIR / f"{MOL_ID}_esp.cube"

DENSITY_LOG = DATA_DIR / f"{MOL_ID}_orcaplot_density.log"
ESP_LOG = DATA_DIR / f"{MOL_ID}_orcaplot_esp.log"


def ensure_dirs():
    for d in [GEOM_DIR, CUBES_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def copy_required_files():
    if not GBW_INPUT.exists():
        raise FileNotFoundError(f"GBW bulunamadı: {GBW_INPUT}")

    if not DENS_INPUT.exists():
        raise FileNotFoundError(f"Density container bulunamadı: {DENS_INPUT}")

    shutil.copy2(GBW_INPUT, GBW_GEOM)
    shutil.copy2(DENS_INPUT, DENS_GEOM)

    print(f"GBW kopyalandı      : {GBW_INPUT} -> {GBW_GEOM}")
    print(f"Densities kopyalandı: {DENS_INPUT} -> {DENS_GEOM}")


def clean_old_cubes():
    for c in GEOM_DIR.glob("*.cube"):
        c.unlink()
    for c in CUBES_DIR.glob(f"{MOL_ID}_*.cube"):
        c.unlink()

    print("Eski cube dosyaları temizlendi.")


def run_orcaplot(sequence, log_path, label):
    if not ORCA_PLOT.exists():
        raise FileNotFoundError(f"orca_plot bulunamadı: {ORCA_PLOT}")

    cmd = [str(ORCA_PLOT), f"{STEM}.gbw", "-i"]

    print(f"\n=== {label} ===")
    print("Çalıştırılıyor:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(GEOM_DIR),
        input=sequence,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800
    )

    log_path.write_text(result.stdout, encoding="utf-8", errors="ignore")

    print(f"Return code: {result.returncode}")
    print(f"Log: {log_path}")

    if result.returncode != 0:
        print("UYARI: orca_plot normal çıkmamış olabilir.")


def list_cubes():
    cubes = list(GEOM_DIR.glob("*.cube"))
    print("\nGeometry klasöründeki cube dosyaları:")
    if not cubes:
        print("Cube dosyası bulunamadı.")
    else:
        for c in cubes:
            print(f"{c.name} | {c.stat().st_size} bytes")
    return cubes


def assign_final_cubes():
    cubes = list_cubes()

    density_candidates = [
        c for c in cubes
        if any(key in c.name.lower() for key in ["eldens", "density", "dens"])
    ]

    esp_candidates = [
        c for c in cubes
        if any(key in c.name.lower() for key in ["esp", "potential", "elpot"])
    ]

    if density_candidates:
        src = sorted(density_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
        shutil.copy2(src, DENSITY_CUBE_FINAL)
        print(f"Density cube final: {src.name} -> {DENSITY_CUBE_FINAL}")
    else:
        print("UYARI: Density cube otomatik tanınamadı.")

    if esp_candidates:
        src = sorted(esp_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
        shutil.copy2(src, ESP_CUBE_FINAL)
        print(f"ESP cube final    : {src.name} -> {ESP_CUBE_FINAL}")
    else:
        print("UYARI: ESP cube otomatik tanınamadı.")

    print("\nFinal cube durumu:")
    print(f"{DENSITY_CUBE_FINAL.name:<28} {'OK' if DENSITY_CUBE_FINAL.exists() else 'YOK'}")
    print(f"{ESP_CUBE_FINAL.name:<28} {'OK' if ESP_CUBE_FINAL.exists() else 'YOK'}")


def main():
    ensure_dirs()
    copy_required_files()
    clean_old_cubes()

    # Electron density cube
    # ORCA plot sequence:
    # 1  -> select plot type
    # 2  -> electron density
    # y  -> accept default density
    # 5  -> output format
    # 7  -> Gaussian cube
    # 11 -> generate plot
    # 12 -> exit
    density_sequence = "\n".join([
        "1",
        "2",
        "y",
        "5",
        "7",
        "11",
        "12",
        ""
    ])

    run_orcaplot(
        density_sequence,
        DENSITY_LOG,
        "Electron density cube"
    )

    time.sleep(2)

    # ESP cube
    # ORCA 6.1 için .densities container kullanıyoruz.
    esp_sequence = "\n".join([
        "1",
        "43",
        f"{STEM}.densities",
        "5",
        "7",
        "11",
        "12",
        ""
    ])

    run_orcaplot(
        esp_sequence,
        ESP_LOG,
        "Electrostatic potential cube"
    )

    time.sleep(2)

    assign_final_cubes()

    print("\nBitti.")
    print("Eğer density veya ESP cube YOK görünürse log dosyalarını okuyacağız:")
    print(DENSITY_LOG)
    print(ESP_LOG)


if __name__ == "__main__":
    main()