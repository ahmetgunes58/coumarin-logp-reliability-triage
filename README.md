# Coumarin logP Reliability Triage

This repository contains the data, scripts, figures, DFT/MEP files, and reproducibility materials accompanying the study:

**Scaffold-Local Fragment-Additivity Breakdown in Coumarin logP Prediction: DFT-Supported Reliability Triage for Medicinal Chemistry Decision-Making**

This is a journal-neutral reproducibility repository. It includes curated coumarin logP benchmark data, external predictor outputs, SangsterLogP audit files, DFT/MEP files, final figures, scripts, and integrity checksums.

## Study overview

The study evaluates reliability limits in fragment-additivity-oriented logP prediction for experimentally characterized coumarin derivatives.

The repository includes:

- a curated 95-compound coumarin benchmark dataset,
- experimental logP provenance and curation records,
- SwissADME-associated logP outputs and error terms,
- external comparator outputs from RDKit MolLogP, DataWarrior cLogP, ALOGPS 2.1, and OPERA logP,
- non-overlapping SangsterLogP external audit files,
- failure-mode labels and reliability-triage outputs,
- a 10-compound DFT/MEP diagnostic panel,
- final main and supporting figures,
- scripts for analysis, auditing, figure generation, DFT/MEP processing, and external-comparator processing.

The repository does not define a new predictive QSPR correction model. Instead, it provides a reliability-triage framework for identifying regions where standard fragment-based logP predictions can become unreliable for polar nitrogen-containing coumarins.

## Repository structure

data/
  raw/
  processed/
  reports/

scripts/
  analysis/
  dft/
  external_comparators/
  figures/
  mep/
  sangsterlogp/

figures/
  main/
  supporting/
  graphical_abstract/

dft/
  molecules/

environment/
metadata/

## Core dataset files

Key files include:

data/raw/Dataset_S0_raw_literature_collection.xlsx
data/processed/benchmark/Dataset_S1_benchmark_dataset.csv
data/processed/benchmark/Dataset_S2_experimental_sources.csv
data/processed/benchmark/Dataset_S3_exclusion_log.csv

## DFT/MEP panel

The DFT/MEP panel covers ten compounds:

CMR_GOLD_055
CMR_GOLD_043
CMR_GOLD_044
CMR_GOLD_029
CMR_GOLD_058
CMR_GOLD_079
CMR_GOLD_016
CMR_GOLD_090
CMR_GOLD_020
CMR_GOLD_092

## Checksums

A SHA256 checksum list is provided at:

metadata/SHA256SUMS.txt

## Citation

Please cite this repository using CITATION.cff. After archival deposition, the Zenodo DOI should also be cited.

## Author

Ahmet GÜNEŞ
National Defence University, Turkish Naval Academy, Department of Basic Sciences, Istanbul, Türkiye
ORCID: https://orcid.org/0000-0003-0966-4025
Email: ahmet.gunes3@msu.edu.tr

## License

This repository is released under the MIT License. See LICENSE for details.
