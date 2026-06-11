from pathlib import Path
import csv
import re

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_044"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID
SP_OUT = MOL_DIR / "output" / f"{MOL_ID}_sp.out"
CHARGE_CSV = MOL_DIR / "extracted_data" / f"{MOL_ID}_charges_mulliken_loewdin.csv"
HEAVY_CSV = MOL_DIR / "extracted_data" / f"{MOL_ID}_heavy_atom_charges.csv"
SUMMARY_TXT = MOL_DIR / "extracted_data" / f"{MOL_ID}_charge_summary.txt"


def parse_charge_block(lines, start_marker):
    charges = []
    inside = False

    for line in lines:
        if start_marker in line:
            inside = True
            continue

        if inside:
            s = line.strip()

            if not s:
                if charges:
                    break
                continue

            if "Sum of atomic charges" in s:
                break

            # Typical ORCA format:
            # 0 C : -0.123456
            # or 0 C -0.123456
            parts = s.replace(":", " ").split()

            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    atom = parts[1]
                    charge = float(parts[2])
                    charges.append((idx, atom, charge))
                except ValueError:
                    pass

    return charges


def main():
    if not SP_OUT.exists():
        raise FileNotFoundError(f"SP output bulunamadı: {SP_OUT}")

    lines = SP_OUT.read_text(errors="ignore").splitlines()

    mulliken = parse_charge_block(lines, "MULLIKEN ATOMIC CHARGES")
    loewdin = parse_charge_block(lines, "LOEWDIN ATOMIC CHARGES")

    if not mulliken:
        raise RuntimeError("Mulliken charge bloğu bulunamadı.")
    if not loewdin:
        raise RuntimeError("Loewdin charge bloğu bulunamadı.")

    loewdin_map = {idx: charge for idx, atom, charge in loewdin}

    rows = []
    for idx, atom, mull_charge in mulliken:
        rows.append({
            "Compound_ID": MOL_ID,
            "Atom_index_0based": idx,
            "Atom_index_1based": idx + 1,
            "Element": atom,
            "Mulliken_charge": mull_charge,
            "Loewdin_charge": loewdin_map.get(idx, None),
            "Is_heavy_atom": atom != "H",
            "Is_N": atom == "N",
            "Is_O": atom == "O",
        })

    CHARGE_CSV.parent.mkdir(parents=True, exist_ok=True)

    with CHARGE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    heavy_rows = [r for r in rows if r["Is_heavy_atom"]]

    with HEAVY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(heavy_rows)

    n_rows = [r for r in rows if r["Is_N"]]
    o_rows = [r for r in rows if r["Is_O"]]

    most_neg_mull = sorted(heavy_rows, key=lambda r: r["Mulliken_charge"])[:6]
    most_pos_mull = sorted(heavy_rows, key=lambda r: r["Mulliken_charge"], reverse=True)[:6]

    with SUMMARY_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{MOL_ID} Mulliken / Loewdin charge summary\n")
        f.write("=" * 72 + "\n\n")

        f.write("Nitrogen atoms:\n")
        for r in n_rows:
            f.write(
                f"Atom {r['Atom_index_1based']:>2} {r['Element']:>2} | "
                f"Mulliken {r['Mulliken_charge']:>10.6f} | "
                f"Loewdin {r['Loewdin_charge']:>10.6f}\n"
            )

        f.write("\nOxygen atoms:\n")
        for r in o_rows:
            f.write(
                f"Atom {r['Atom_index_1based']:>2} {r['Element']:>2} | "
                f"Mulliken {r['Mulliken_charge']:>10.6f} | "
                f"Loewdin {r['Loewdin_charge']:>10.6f}\n"
            )

        f.write("\nMost negative heavy atoms by Mulliken charge:\n")
        for r in most_neg_mull:
            f.write(
                f"Atom {r['Atom_index_1based']:>2} {r['Element']:>2} | "
                f"Mulliken {r['Mulliken_charge']:>10.6f} | "
                f"Loewdin {r['Loewdin_charge']:>10.6f}\n"
            )

        f.write("\nMost positive heavy atoms by Mulliken charge:\n")
        for r in most_pos_mull:
            f.write(
                f"Atom {r['Atom_index_1based']:>2} {r['Element']:>2} | "
                f"Mulliken {r['Mulliken_charge']:>10.6f} | "
                f"Loewdin {r['Loewdin_charge']:>10.6f}\n"
            )

    print("Charge extraction tamamlandı.")
    print(f"Tam charge tablosu : {CHARGE_CSV}")
    print(f"Heavy atom tablosu : {HEAVY_CSV}")
    print(f"Özet rapor         : {SUMMARY_TXT}")

    print("\nNitrogen atoms:")
    for r in n_rows:
        print(
            f"Atom {r['Atom_index_1based']:>2} {r['Element']} | "
            f"Mulliken {r['Mulliken_charge']:.6f} | "
            f"Loewdin {r['Loewdin_charge']:.6f}"
        )

    print("\nOxygen atoms:")
    for r in o_rows:
        print(
            f"Atom {r['Atom_index_1based']:>2} {r['Element']} | "
            f"Mulliken {r['Mulliken_charge']:.6f} | "
            f"Loewdin {r['Loewdin_charge']:.6f}"
        )


if __name__ == "__main__":
    main()