from pathlib import Path
import shutil
import subprocess

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

ESP_LOG = DATA_DIR / f"{MOL_ID}_orcaplot_esp_only.log"
ESP_CUBE_FINAL = CUBES_DIR / f"{MOL_ID}_esp.cube"


def ensure_files():
    for d in [GEOM_DIR, CUBES_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if not ORCA_PLOT.exists():
        raise FileNotFoundError(f"orca_plot bulunamadı: {ORCA_PLOT}")

    if not GBW_INPUT.exists():
        raise FileNotFoundError(f"GBW yok: {GBW_INPUT}")

    if not DENS_INPUT.exists():
        raise FileNotFoundError(f"Densities yok: {DENS_INPUT}")

    shutil.copy2(GBW_INPUT, GBW_GEOM)
    shutil.copy2(DENS_INPUT, DENS_GEOM)

    print(f"GBW hazır      : {GBW_GEOM}")
    print(f"Densities hazır: {DENS_GEOM}")


def remove_old_esp_cubes():
    for p in GEOM_DIR.glob("*esp*.cube"):
        p.unlink()
    for p in GEOM_DIR.glob("*ESP*.cube"):
        p.unlink()
    if ESP_CUBE_FINAL.exists():
        ESP_CUBE_FINAL.unlink()


def run_esp():
    # Kritik değişiklik:
    # 43 seçildikten sonra density adı yazmıyoruz; varsayılan density'yi 'y' ile kabul ediyoruz.
    esp_sequence = "\n".join([
        "1",      # Enter type of plot
        "43",     # Electrostatic Potential
        "y",      # accept default density
        "5",      # Select output format
        "7",      # Gaussian cube
        "11",     # Generate plot
        "12",     # Exit
        ""
    ])

    cmd = [str(ORCA_PLOT), f"{STEM}.gbw", "-i"]

    print("Çalıştırılıyor:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=str(GEOM_DIR),
        input=esp_sequence,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=3600
    )

    ESP_LOG.write_text(result.stdout, encoding="utf-8", errors="ignore")

    print(f"Return code: {result.returncode}")
    print(f"Log yazıldı: {ESP_LOG}")

    return result.returncode


def collect_esp_cube():
    cube_files = list(GEOM_DIR.glob("*.cube"))

    print("\nGeometry cube dosyaları:")
    for c in cube_files:
        print(f"{c.name} | {c.stat().st_size} bytes")

    esp_candidates = [
        c for c in cube_files
        if any(key in c.name.lower() for key in ["esp", "potential", "elpot"])
    ]

    if not esp_candidates:
        print("\nESP cube otomatik bulunamadı.")
        print("Log dosyasını kontrol edeceğiz:")
        print(ESP_LOG)
        return False

    src = sorted(esp_candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    shutil.copy2(src, ESP_CUBE_FINAL)

    print(f"\nESP cube final kopyalandı:")
    print(f"{src} -> {ESP_CUBE_FINAL}")
    return True


def main():
    ensure_files()
    remove_old_esp_cubes()
    ret = run_esp()
    ok = collect_esp_cube()

    print("\nFinal durum:")
    print(f"{ESP_CUBE_FINAL.name:<24} {'OK' if ESP_CUBE_FINAL.exists() else 'YOK'}")

    if ret != 0 or not ok:
        print("\nESP üretimi hâlâ tamamlanmadı. Şu komutu çalıştırıp logu gönder:")
        print(f'type "{ESP_LOG}"')


if __name__ == "__main__":
    main()