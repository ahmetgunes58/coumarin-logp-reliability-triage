from pathlib import Path
import re
import csv

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
MOL_ID = "CMR_GOLD_029"

MOL_DIR = PROJECT / "dft" / "molecules" / MOL_ID

OPT_INP = MOL_DIR / "input" / f"{MOL_ID}_optfreq.inp"
SP_INP = MOL_DIR / "input" / f"{MOL_ID}_sp.inp"
OPT_OUT = MOL_DIR / "output" / f"{MOL_ID}_optfreq.out"
SP_OUT = MOL_DIR / "output" / f"{MOL_ID}_sp.out"

DATA_DIR = MOL_DIR / "extracted_data"
REPORT_TXT = DATA_DIR / f"{MOL_ID}_method_audit_report.txt"
REPORT_CSV = DATA_DIR / f"{MOL_ID}_method_audit_report.csv"

EXPECTED_KEYWORDS = [
    "B3LYP",
    "D3BJ",
    "def2-SVP",
    "TightSCF",
    "RIJCOSX",
    "def2/J",
    "CPCM(Water)",
]

EXPECTED_OPT_KEYWORDS = EXPECTED_KEYWORDS + ["Opt", "Freq"]


def read(path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def contains_all(text, keywords):
    return {k: (k in text) for k in keywords}


def get_charge_mult(inp_text):
    for line in inp_text.splitlines():
        s = line.strip()
        if s.startswith("* xyzfile"):
            parts = s.split()
            if len(parts) >= 4:
                return parts[2], parts[3]
    return None, None


def has_normal_termination(out_text):
    return "ORCA TERMINATED NORMALLY" in out_text


def has_converged(out_text):
    return (
        "THE OPTIMIZATION HAS CONVERGED" in out_text
        or "HURRAY" in out_text
        or "OPTIMIZATION RUN DONE" in out_text
    )


def count_freqs_and_imag(out_text):
    freqs = []
    in_freq = False

    for line in out_text.splitlines():
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
    return len(freqs), imag


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    opt_inp = read(OPT_INP)
    sp_inp = read(SP_INP)
    opt_out = read(OPT_OUT)
    sp_out = read(SP_OUT)

    opt_keywords = contains_all(opt_inp, EXPECTED_OPT_KEYWORDS)
    sp_keywords = contains_all(sp_inp, EXPECTED_KEYWORDS)

    opt_charge, opt_mult = get_charge_mult(opt_inp)
    sp_charge, sp_mult = get_charge_mult(sp_inp)

    nfreq, imag = count_freqs_and_imag(opt_out)

    rows = [
        ("Molecule_ID", MOL_ID),
        ("Expected_protocol", "ORCA 6.1.1 | B3LYP-D3BJ/def2-SVP | CPCM(Water) | TightSCF | RIJCOSX/def2/J | neutral singlet"),
        ("Opt_input_exists", OPT_INP.exists()),
        ("SP_input_exists", SP_INP.exists()),
        ("Opt_output_exists", OPT_OUT.exists()),
        ("SP_output_exists", SP_OUT.exists()),
        ("Opt_route_line", next((l.strip() for l in opt_inp.splitlines() if l.strip().startswith("!")), "")),
        ("SP_route_line", next((l.strip() for l in sp_inp.splitlines() if l.strip().startswith("!")), "")),
        ("Opt_charge", opt_charge),
        ("Opt_multiplicity", opt_mult),
        ("SP_charge", sp_charge),
        ("SP_multiplicity", sp_mult),
        ("Opt_normal_termination", has_normal_termination(opt_out)),
        ("Opt_converged", has_converged(opt_out)),
        ("Frequency_count", nfreq),
        ("Imaginary_frequencies_less_than_minus_20_cm-1", imag),
        ("SP_normal_termination", has_normal_termination(sp_out)),
    ]

    for k, v in opt_keywords.items():
        rows.append((f"Opt_keyword_{k}", v))

    for k, v in sp_keywords.items():
        rows.append((f"SP_keyword_{k}", v))

    # Overall method consistency decision
    method_ok = (
        all(opt_keywords.values())
        and all(sp_keywords.values())
        and opt_charge == "0"
        and opt_mult == "1"
        and sp_charge == "0"
        and sp_mult == "1"
        and has_normal_termination(opt_out)
        and has_converged(opt_out)
        and len(imag) == 0
        and has_normal_termination(sp_out)
    )

    rows.append(("Overall_method_audit", "PASS" if method_ok else "CHECK"))

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{MOL_ID} method audit report\n")
        f.write("=" * 80 + "\n\n")
        for k, v in rows:
            f.write(f"{k}: {v}\n")

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Field", "Value"])
        writer.writerows(rows)

    print(f"\n{MOL_ID} method audit")
    print("=" * 80)
    for k, v in rows:
        print(f"{k}: {v}")

    print("\nRaporlar:")
    print(REPORT_TXT)
    print(REPORT_CSV)


if __name__ == "__main__":
    main()