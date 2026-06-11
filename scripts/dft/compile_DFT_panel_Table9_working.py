from pathlib import Path
import pandas as pd

PROJECT = Path(r"D:\Makaleler\coumarin-logp-working-source")
OUTDIR = PROJECT / "tables"
OUTDIR.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "Compound_ID": "CMR_GOLD_055",
        "FM_role": "N-free reference / parent core",
        "N_count": 0,
        "logP_exp": 1.39,
        "Consensus_logP": 1.82,
        "Delta_logP_Exp_minus_Pred": -0.43,
        "HOMO_eV": -6.5758,
        "LUMO_eV": -1.9387,
        "Gap_eV": 4.6371,
        "Dipole_D": 6.5831,
        "Status": "existing value; verify from old output before final",
    },
    {
        "Compound_ID": "CMR_GOLD_043",
        "FM_role": "FM0 oxadiazole accurate #1",
        "N_count": 2,
        "logP_exp": 1.95,
        "Consensus_logP": 1.78,
        "Delta_logP_Exp_minus_Pred": 0.17,
        "HOMO_eV": -6.7041,
        "LUMO_eV": -2.5884,
        "Gap_eV": 4.1157,
        "Dipole_D": 5.7832,
        "Status": "existing value; verify from old output before final",
    },
    {
        "Compound_ID": "CMR_GOLD_044",
        "FM_role": "FM0 oxadiazole accurate #2",
        "N_count": 2,
        "logP_exp": 2.12,
        "Consensus_logP": 2.15,
        "Delta_logP_Exp_minus_Pred": -0.03,
        "HOMO_eV": -6.6834,
        "LUMO_eV": -2.5383,
        "Gap_eV": 4.1451,
        "Dipole_D": 6.350236912,
        "Status": "new calculation completed",
    },
    {
        "Compound_ID": "CMR_GOLD_029",
        "FM_role": "FM1 N=2 conjugated failure",
        "N_count": 2,
        "logP_exp": 0.55,
        "Consensus_logP": 2.59,
        "Delta_logP_Exp_minus_Pred": -2.04,
        "HOMO_eV": -6.1419,
        "LUMO_eV": -1.6896,
        "Gap_eV": 4.4523,
        "Dipole_D": 10.369461119,
        "Status": "new calculation completed",
    },
    {
        "Compound_ID": "CMR_GOLD_058",
        "FM_role": "FM1 extreme D-pi-A failure",
        "N_count": 2,
        "logP_exp": 0.97,
        "Consensus_logP": 6.16,
        "Delta_logP_Exp_minus_Pred": -5.19,
        "HOMO_eV": -4.9939,
        "LUMO_eV": -2.3859,
        "Gap_eV": 2.6080,
        "Dipole_D": 6.1104,
        "Status": "existing value; verify from old output before final",
    },
    {
        "Compound_ID": "CMR_GOLD_079",
        "FM_role": "FM3 high-N cancellation",
        "N_count": 4,
        "logP_exp": 2.02,
        "Consensus_logP": 1.97,
        "Delta_logP_Exp_minus_Pred": 0.05,
        "HOMO_eV": -6.1912,
        "LUMO_eV": -2.3344,
        "Gap_eV": 3.8568,
        "Dipole_D": 7.4431,
        "Status": "existing value; verify from old output before final",
    },
    {
        "Compound_ID": "CMR_GOLD_016",
        "FM_role": "FM4 dimeric coumarin opposite-bias",
        "N_count": 0,
        "logP_exp": 4.93,
        "Consensus_logP": 2.66,
        "Delta_logP_Exp_minus_Pred": 2.27,
        "HOMO_eV": -6.3417,
        "LUMO_eV": -1.8424,
        "Gap_eV": 4.4993,
        "Dipole_D": 3.402408252,
        "Status": "new calculation completed",
    },
]

df = pd.DataFrame(rows)

df["Abs_Delta_logP"] = df["Delta_logP_Exp_minus_Pred"].abs()

csv_path = OUTDIR / "Table9_DFT_panel_working.csv"
xlsx_path = OUTDIR / "Table9_DFT_panel_working.xlsx"

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_excel(xlsx_path, index=False)

print("Table 9 working files written:")
print(csv_path)
print(xlsx_path)
print()
print(df.to_string(index=False))