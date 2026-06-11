from pathlib import Path
import numpy as np
import csv

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID
CUBE_DIR = MOL_DIR / "cubes"
DATA_DIR = MOL_DIR / "extracted_data"

DENSITY_CUBE = CUBE_DIR / f"{MOL_ID}_density.cube"
ESP_CUBE = CUBE_DIR / f"{MOL_ID}_esp.cube"

REPORT_TXT = DATA_DIR / f"{MOL_ID}_cube_validation_report.txt"
REPORT_CSV = DATA_DIR / f"{MOL_ID}_cube_validation_report.csv"


def read_cube(path):
    lines = path.read_text(errors="ignore").splitlines()

    comment1 = lines[0]
    comment2 = lines[1]

    natoms_line = lines[2].split()
    natoms = int(natoms_line[0])
    origin = np.array([float(natoms_line[1]), float(natoms_line[2]), float(natoms_line[3])])

    grid = []
    axes = []

    for i in range(3):
        parts = lines[3 + i].split()
        n = int(parts[0])
        vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        grid.append(n)
        axes.append(vec)

    atom_start = 6
    atom_end = atom_start + abs(natoms)
    atoms = lines[atom_start:atom_end]

    data_tokens = []
    for line in lines[atom_end:]:
        data_tokens.extend(line.split())

    data = np.array([float(x) for x in data_tokens], dtype=float)

    expected = abs(grid[0] * grid[1] * grid[2])

    if data.size != expected:
        raise RuntimeError(
            f"{path.name}: data size mismatch. Expected {expected}, found {data.size}"
        )

    arr = data.reshape((abs(grid[0]), abs(grid[1]), abs(grid[2])))

    return {
        "path": path,
        "comment1": comment1,
        "comment2": comment2,
        "natoms": natoms,
        "origin": origin,
        "grid": tuple(grid),
        "axes": tuple(tuple(v) for v in axes),
        "atoms": atoms,
        "data": arr,
        "flat": data,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DENSITY_CUBE.exists():
        raise FileNotFoundError(f"Density cube yok: {DENSITY_CUBE}")
    if not ESP_CUBE.exists():
        raise FileNotFoundError(f"ESP cube yok: {ESP_CUBE}")

    dens = read_cube(DENSITY_CUBE)
    esp = read_cube(ESP_CUBE)

    same_grid = dens["grid"] == esp["grid"]
    same_origin = np.allclose(dens["origin"], esp["origin"])
    same_axes = dens["axes"] == esp["axes"]
    same_natoms = dens["natoms"] == esp["natoms"]

    rho = dens["flat"]
    vesp = esp["flat"]

    # MEP common scale used in manuscript/SI
    common_min = -0.08
    common_max = 0.08

    # Points near density isosurface 0.004 a.u.
    rho_iso = 0.004
    shell_mask = (rho > rho_iso * 0.90) & (rho < rho_iso * 1.10)

    shell_count = int(shell_mask.sum())
    shell_esp_min = float(np.min(vesp[shell_mask])) if shell_count else None
    shell_esp_max = float(np.max(vesp[shell_mask])) if shell_count else None
    shell_esp_mean = float(np.mean(vesp[shell_mask])) if shell_count else None

    report = {
        "Molecule_ID": MOL_ID,
        "Density_cube": str(DENSITY_CUBE),
        "ESP_cube": str(ESP_CUBE),
        "Same_grid": same_grid,
        "Same_origin": same_origin,
        "Same_axes": same_axes,
        "Same_natoms": same_natoms,
        "Grid": dens["grid"],
        "Natoms": dens["natoms"],
        "Density_min": float(np.min(rho)),
        "Density_max": float(np.max(rho)),
        "Density_mean": float(np.mean(rho)),
        "ESP_min": float(np.min(vesp)),
        "ESP_max": float(np.max(vesp)),
        "ESP_mean": float(np.mean(vesp)),
        "Isosurface_density_au": rho_iso,
        "Near_isosurface_shell_points": shell_count,
        "ESP_min_near_rho_0p004": shell_esp_min,
        "ESP_max_near_rho_0p004": shell_esp_max,
        "ESP_mean_near_rho_0p004": shell_esp_mean,
        "Common_MEP_scale_min_au": common_min,
        "Common_MEP_scale_max_au": common_max,
        "ESP_values_outside_common_scale_total": int(((vesp < common_min) | (vesp > common_max)).sum()),
    }

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{MOL_ID} cube validation report\n")
        f.write("=" * 72 + "\n\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Field", "Value"])
        for k, v in report.items():
            writer.writerow([k, v])

    print("\n=== CUBE VALIDATION ===")
    for k, v in report.items():
        print(f"{k}: {v}")

    print("\nRaporlar yazıldı:")
    print(REPORT_TXT)
    print(REPORT_CSV)

    if same_grid and same_origin and same_axes and same_natoms:
        print("\nSONUÇ: Density ve ESP cube aynı grid üzerinde. MEP rendering için uygun.")
    else:
        print("\nUYARI: Cube grid/origin/axis uyumsuzluğu var. Rendering öncesi kontrol edilmeli.")


if __name__ == "__main__":
    main()