# SimECG-N Synthetic ECG Pretraining Corpus Collection

## Summary

This asset contains simulator-generated single-lead-II ECG waveforms intended for research on synthetic ECG pretraining and self-supervised ECG representation learning. The corpus is released to support reproduction and extension of the associated NeurIPS 2026 submission, which evaluates when patient-free ECG simulators can substitute for real ECG pretraining under matched transfer protocols.

## Dataset contents

Each waveform is a 10-second, 500-Hz, single-channel lead-II synthetic ECG signal. The pretraining corpus is unlabeled and is intended for masked-autoencoding pretraining. The released files include waveform arrays, a manifest file, generation configuration files, random seeds, and subset definitions used in the benchmark.

## Intended use

The corpus is intended for:
- benchmarking synthetic ECG pretraining sources;
- evaluating ECG representation-learning methods under matched single-lead-II protocols;
- reproducing the SimECG-N experiments in the associated paper;
- comparing future ECG simulators, learned generators, or self-supervised objectives.

## Non-recommended use

The corpus is not intended for:
- clinical diagnosis;
- clinical validation;
- patient monitoring;
- medical-device development;
- deployment approval;
- replacing clinically representative real ECG validation data;
- evaluating demographic fairness or subgroup robustness.

## Provenance

The corpus was generated using the SimECG-N knowledge-driven ECG simulator with fixed code, documented parameter distributions, and random seeds. No real patient ECG records, patient identifiers, demographic attributes, clinical labels, hospital identifiers, or downstream test records were used to generate the released SimECG-N corpus.

## Known limitations

The corpus contains only single-lead-II, 10-second, 500-Hz synthetic ECG waveforms and does not represent full 12-lead clinical ECG information. It does not explicitly model demographic variation, device variation, site variation, comorbidities, clinical workflows, or rare-disease prevalence. Some generated waveforms may be physiologically atypical because broad simulator perturbations were used to increase synthetic diversity.

## Known or suspected biases

The corpus reflects the assumptions and parameter ranges of the SimECG-N simulator. It may over-represent simulator-accessible morphology, rhythm, beat-to-beat variation, timing, and baseline patterns, while under-representing demographic, device, site, multi-lead, rare-pathology, and noisy clinical acquisition factors.

## Sensitive information

The released SimECG-N corpus does not contain real patient ECG records or patient identifiers. ECG is nevertheless a health-related signal modality, so the corpus should be used only as a research and evaluation resource and should not be interpreted as clinically representative patient data.

## License

CC 4.0

## Croissant metadata

The machine-readable dataset metadata is provided in `metadata/simecg_n_croissant.json`. The Croissant file includes core dataset metadata and Responsible AI fields for intended use, non-recommended use, known limitations, known or suspected biases, sensitive-information status, social-impact considerations, and synthetic-data provenance.
