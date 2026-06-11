from pathlib import Path
import argparse
import subprocess
import shutil
import re
import csv
import sys

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"
MOL = PROJECT / "dft" / "molecules" / MOL_ID

ORCA_EXE = Path(r"C:\orca\orca.exe")

NPROCS = 8
MAXCORE = 3000

SMILES = "O=C1OC2=C(C=CC(OCC(O)CN3C=NC4=C3C=CC=C4)=C2)C(C)=C1"

OPT_INP = MOL / "input" / f"{MOL_ID}_optfreq.inp"
SP_INP = MOL / "input" / f"{MOL_ID}_sp.inp"
START_XYZ = MOL / "input" / f"{MOL_ID}_start.xyz"
OPT_OUT = MOL / "output" / f"{MOL_ID}_optfreq.out"
SP_OUT = MOL / "output" / f"{MOL_ID}_sp.out"
OPT_XYZ = MOL / "geometry" / f"{MOL_ID}_opt.xyz"


def ensure_dirs():
    for sub in [
        "input",
        "output",
        "geometry",
        "cubes",
        "figures",
        "extracted_data",
        "scratch",
    ]:
        (MOL / sub).mkdir(parents=True, exist_ok=True)


def prepare():
    ensure_dirs()

    if not ORCA_EXE.exists():
        raise FileNotFoundError(f"ORCA bulunamadı: {ORCA_EXE}")

    # Eğer starter zip içinden gelen rdkit dosyası varsa otomatik kopyalamaya çalışır.
    possible_xyz = [
        MOL / "input" / f"{MOL_ID}_start_rdkit_mmff.xyz",
        MOL / f"{MOL_ID}_start_rdkit_mmff.xyz",
        PROJECT / f"{MOL_ID}_start_rdkit_mmff.xyz",
    ]

    if not START_XYZ.exists():
        for p in possible_xyz:
            if p.exists():
                shutil.copy2(p, START_XYZ)
                print(f"Başlangıç XYZ kopyalandı: {p} -> {START_XYZ}")
                break

    (MOL / "input" / f"{MOL_ID}.smi").write_text(SMILES + "\n", encoding="utf-8")

    opt_text = f"""! B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water) Opt Freq

%pal
  nprocs {NPROCS}
end

%maxcore {MAXCORE}

%scf
  MaxIter 500
end

%output
  Print[P_Mulliken] 1
  Print[P_Loewdin] 1
end

* xyzfile 0 1 input/{MOL_ID}_start.xyz
"""

    sp_text = f"""! B3LYP D3BJ def2-SVP TightSCF RIJCOSX def2/J CPCM(Water)

%pal
  nprocs {NPROCS}
end

%maxcore {MAXCORE}

%scf
  MaxIter 500
end

%output
  Print[P_Mulliken] 1
  Print[P_Loewdin] 1
end

* xyzfile 0 1 geometry/{MOL_ID}_opt.xyz
"""

    OPT_INP.write_text(opt_text, encoding="utf-8")
    SP_INP.write_text(sp_text, encoding="utf-8")

    print("Hazırlık tamamlandı.")
    print(f"Proje klasörü: {MOL}")
    print(f"Opt/Freq input: {OPT_INP}")
    print(f"SP input: {SP_INP}")

    if START_XYZ.exists():
        print(f"Başlangıç XYZ mevcut: {START_XYZ}")
    else:
        print("\nUYARI: Başlangıç XYZ bulunamadı.")
        print(f"Şu dosyayı bu yola koymalısın:\n{START_XYZ}")


def run_orca(inp: Path, out: Path):
    ensure_dirs()

    if not ORCA_EXE.exists():
        raise FileNotFoundError(f"ORCA bulunamadı: {ORCA_EXE}")

    if not inp.exists():
        raise FileNotFoundError(f"Input dosyası bulunamadı: {inp}")

    print(f"Çalıştırılıyor: {inp.name}")
    print(f"Output: {out}")

    with out.open("w", encoding="utf-8", errors="ignore") as f:
        proc = subprocess.Popen(
            [str(ORCA_EXE), str(inp)],
            cwd=str(MOL),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        ret = proc.wait()

    print(f"ORCA çıkış kodu: {ret}")
    if ret != 0:
        print("UYARI: ORCA normal çıkmamış olabilir. Output dosyasını kontrol et.")
    else:
        print("ORCA komutu tamamlandı.")


def run_opt():
    if not START_XYZ.exists():
        raise FileNotFoundError(
            f"Başlangıç XYZ yok: {START_XYZ}\n"
            "CMR_GOLD_029_start.xyz dosyasını input klasörüne koy."
        )
    run_orca(OPT_INP, OPT_OUT)


def check_opt():
    if not OPT_OUT.exists():
        raise FileNotFoundError(f"Opt/Freq output bulunamadı: {OPT_OUT}")

    text = OPT_OUT.read_text(errors="ignore")

    normal = "ORCA TERMINATED NORMALLY" in text
    converged = (
        "THE OPTIMIZATION HAS CONVERGED" in text
        or "HURRAY" in text
        or "OPTIMIZATION RUN DONE" in text
    )

    freqs = []
    in_freq = False

    for line in text.splitlines():
        if "VIBRATIONAL FREQUENCIES" in line:
            in_freq = True
            continue

        if in_freq:
            if "NORMAL MODES" in line or "IR SPECTRUM" in line:
                break

            m = re.match(r"\s*\d+\s*:\s*(-?\d+\.\d+)\s*cm\*\*-1", line)
            if m:
                freqs.append(float(m.group(1)))

    imag = [f for f in freqs if f < -20.0]

    print("\n=== OPT/FREQ KONTROL ===")
    print(f"Normal termination: {normal}")
    print(f"Optimization converged: {converged}")
    print(f"Frekans sayısı: {len(freqs)}")

    if imag:
        print(f"UYARI: Imaginary frequency olabilir: {imag[:10]}")
        print("SP hesabına geçmeden önce geometri kontrol edilmeli.")
    else:
        print("Belirgin imaginary frequency görülmedi.")

    if normal and converged and not imag:
        print("\nSONUÇ: Opt/Freq uygun görünüyor. Final XYZ çıkarılabilir.")
    else:
        print("\nSONUÇ: SP öncesi output dikkatle kontrol edilmeli.")


def extract_final_xyz():
    if not OPT_OUT.exists():
        raise FileNotFoundError(f"Opt/Freq output bulunamadı: {OPT_OUT}")

    lines = OPT_OUT.read_text(errors="ignore").splitlines()
    blocks = []

    for i, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            block = []

            for j in range(i + 1, len(lines)):
                s = lines[j].strip()

                if not s:
                    if block:
                        break
                    continue

                if set(s) <= {"-"}:
                    continue

                parts = s.split()

                if len(parts) >= 4:
                    atom = parts[0]
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        block.append((atom, x, y, z))
                    except ValueError:
                        if block:
                            break
                elif block:
                    break

            if block:
                blocks.append(block)

    if not blocks:
        raise RuntimeError("ORCA output içinde Cartesian coordinate block bulunamadı.")

    final = blocks[-1]
    OPT_XYZ.parent.mkdir(parents=True, exist_ok=True)

    with OPT_XYZ.open("w", encoding="utf-8") as f:
        f.write(f"{len(final)}\n")
        f.write(f"{MOL_ID} optimized geometry from ORCA opt/freq\n")
        for atom, x, y, z in final:
            f.write(f"{atom:<2s} {x:16.8f} {y:16.8f} {z:16.8f}\n")

    print(f"Final XYZ yazıldı: {OPT_XYZ}")
    print(f"Atom sayısı: {len(final)}")


def run_sp():
    if not OPT_XYZ.exists():
        raise FileNotFoundError(
            f"Optimize XYZ bulunamadı: {OPT_XYZ}\n"
            "Önce: python cmr029_workflow.py extract-xyz"
        )
    run_orca(SP_INP, SP_OUT)


def extract_sp():
    if not SP_OUT.exists():
        raise FileNotFoundError(f"SP output bulunamadı: {SP_OUT}")

    text = SP_OUT.read_text(errors="ignore")
    lines = text.splitlines()

    orbital_rows = []
    in_orb = False

    for line in lines:
        if "ORBITAL ENERGIES" in line:
            in_orb = True
            continue

        if in_orb:
            if "MOLECULAR ORBITALS" in line or "MULLIKEN" in line or "LOEWDIN" in line:
                if orbital_rows:
                    break

            parts = line.split()
            if len(parts) >= 4:
                try:
                    idx = int(parts[0])
                    occ = float(parts[1])
                    e_hartree = float(parts[2])
                    e_ev = float(parts[3])
                    orbital_rows.append((idx, occ, e_hartree, e_ev))
                except ValueError:
                    pass

    homo = None
    lumo = None

    for row in orbital_rows:
        idx, occ, eh, ev = row
        if occ > 0.0:
            homo = row
        elif occ == 0.0 and homo is not None:
            lumo = row
            break

    dipole_debye = None
    for i, line in enumerate(lines):
        if "Magnitude (Debye)" in line:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if nums:
                dipole_debye = float(nums[-1])

    normal = "ORCA TERMINATED NORMALLY" in text

    out_csv = MOL / "extracted_data" / f"{MOL_ID}_sp_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Compound_ID", "Normal_termination", "HOMO_eV", "LUMO_eV", "Gap_eV", "Dipole_Debye"])

        if homo and lumo:
            homo_ev = homo[3]
            lumo_ev = lumo[3]
            gap = lumo_ev - homo_ev
            writer.writerow([MOL_ID, normal, homo_ev, lumo_ev, gap, dipole_debye])
        else:
            writer.writerow([MOL_ID, normal, None, None, None, dipole_debye])

    print("\n=== SP EXTRACTION ===")
    print(f"Normal termination: {normal}")

    if homo and lumo:
        print(f"HOMO: {homo[3]:.6f} eV")
        print(f"LUMO: {lumo[3]:.6f} eV")
        print(f"Gap : {lumo[3] - homo[3]:.6f} eV")
    else:
        print("HOMO/LUMO otomatik çıkarılamadı; ORBITAL ENERGIES bloğu manuel kontrol edilmeli.")

    print(f"Dipole: {dipole_debye} Debye")
    print(f"CSV yazıldı: {out_csv}")


def status():
    print("\n=== CMR_GOLD_029 STATUS ===")
    for p in [START_XYZ, OPT_INP, OPT_OUT, OPT_XYZ, SP_INP, SP_OUT]:
        print(f"{p.name:<32} {'OK' if p.exists() else 'YOK'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "run-opt",
            "check-opt",
            "extract-xyz",
            "run-sp",
            "extract-sp",
            "status",
        ],
    )
    args = parser.parse_args()

    if args.stage == "prepare":
        prepare()
    elif args.stage == "run-opt":
        run_opt()
    elif args.stage == "check-opt":
        check_opt()
    elif args.stage == "extract-xyz":
        extract_final_xyz()
    elif args.stage == "run-sp":
        run_sp()
    elif args.stage == "extract-sp":
        extract_sp()
    elif args.stage == "status":
        status()


if __name__ == "__main__":
    main()