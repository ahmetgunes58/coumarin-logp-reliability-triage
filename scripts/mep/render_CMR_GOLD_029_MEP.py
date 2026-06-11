from pathlib import Path
import numpy as np
from skimage import measure
import pyvista as pv

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID
CUBE_DIR = MOL_DIR / "cubes"
FIG_DIR = MOL_DIR / "figures"

DENSITY_CUBE = CUBE_DIR / f"{MOL_ID}_density.cube"
ESP_CUBE = CUBE_DIR / f"{MOL_ID}_esp.cube"

OUT_PNG = FIG_DIR / f"{MOL_ID}_MEP_common_scale.png"


def read_cube(path):
    lines = path.read_text(errors="ignore").splitlines()

    natoms_line = lines[2].split()
    natoms = int(natoms_line[0])
    origin = np.array([float(natoms_line[1]), float(natoms_line[2]), float(natoms_line[3])], dtype=float)

    grid = []
    axes = []
    for i in range(3):
        parts = lines[3 + i].split()
        n = int(parts[0])
        vec = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        grid.append(abs(n))
        axes.append(vec)

    atom_start = 6
    atom_end = atom_start + abs(natoms)

    data_tokens = []
    for line in lines[atom_end:]:
        data_tokens.extend(line.split())

    data = np.array([float(x) for x in data_tokens], dtype=float)
    arr = data.reshape(tuple(grid))

    return origin, axes, tuple(grid), arr


def interpolate_nearest(values, vertices):
    idx = np.rint(vertices).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, values.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, values.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, values.shape[2] - 1)
    return values[idx[:, 0], idx[:, 1], idx[:, 2]]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    origin, axes, grid, rho = read_cube(DENSITY_CUBE)
    origin2, axes2, grid2, esp = read_cube(ESP_CUBE)

    if grid != grid2:
        raise RuntimeError("Density ve ESP grid farklı.")
    if not np.allclose(origin, origin2):
        raise RuntimeError("Density ve ESP origin farklı.")

    rho_iso = 0.004

    verts, faces, normals, values = measure.marching_cubes(
        rho,
        level=rho_iso,
        spacing=(1.0, 1.0, 1.0)
    )

    real_verts = []
    ax0, ax1, ax2 = axes
    for v in verts:
        r = origin + v[0] * ax0 + v[1] * ax1 + v[2] * ax2
        real_verts.append(r)
    real_verts = np.array(real_verts)

    esp_on_surface = interpolate_nearest(esp, verts)

    pv_faces = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int64)

    mesh = pv.PolyData(real_verts, pv_faces)
    mesh["ESP_au"] = esp_on_surface

    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1400))
    plotter.set_background("white")

    plotter.add_mesh(
        mesh,
        scalars="ESP_au",
        cmap="coolwarm",
        clim=[-0.08, 0.08],
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "MEP (a.u.)",
            "vertical": True,
            "title_font_size": 24,
            "label_font_size": 20,
        },
    )

    plotter.add_text(
        f"{MOL_ID} | electron density isosurface = 0.004 a.u. | MEP scale = -0.08 to +0.08 a.u.",
        position="upper_left",
        font_size=18,
        color="black"
    )

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.25)

    plotter.screenshot(str(OUT_PNG))
    plotter.close()

    print(f"MEP figure written: {OUT_PNG}")


if __name__ == "__main__":
    main()