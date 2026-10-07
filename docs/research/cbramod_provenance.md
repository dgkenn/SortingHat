# CBraMod pretrained checkpoint: provenance record

Recorded: 2026-10-07 (UTC). Subject: "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding" (ICLR 2025; arXiv 2412.07236).

## 1. Code repository

- Repo: https://github.com/wjq-learning/CBraMod
- Default-branch HEAD commit: `b9e961003214326972c567eff390e75b0287e32a` (via `git ls-remote`, 2026-10-07)
- License file: `LICENSE`, MIT, "Copyright (c) 2025 Jiquan Wang"
- README (main branch) links the checkpoint: "We have released a pretrained checkpoint on Hugginface" -> https://huggingface.co/weighting666/CBraMod
- Quick Start loads weights from `pretrained_weights/pretrained_weights.pth`.
- The README does not name the pretraining dataset, state a license for the weights, or mention TUAB/TUEV/TUSZ.

## 2. Checkpoint

- Host: Hugging Face, model repo `weighting666/CBraMod` (linked from the official GitHub README). The HF account name differs from the GitHub owner (`wjq-learning`); the link is the authors' own README link, but the HF owner's identity was not independently verified.
- Direct URL: https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth
- Filename: `pretrained_weights.pth`
- Size: 19,775,842 bytes (19.8 MB)
- SHA-256: `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`
  - Matches the Git-LFS OID reported by the HF tree API (`lfs.oid`), so the downloaded bytes are the file HF serves.
- HF repo revision at download: `sha 500543c7e30bda1b22bfd51a49301b238dee21fd` (lastModified 2024-12-11)
- Download date: 2026-10-07, 19:54 UTC, via `curl -L`. Local copy: `/tmp/claude-0/-home-user/a164876e-c089-590a-bb4a-922f46166e8d/scratchpad/cbramod/pretrained_weights.pth` (scratch copy, not in the repo).
- HF model card license: `apache-2.0` (see license discrepancy in section 5).
- Not verified: whether this file is the exact checkpoint used in the paper's reported results.

## 3. Pretraining data

Source: paper Section 3.1 ("Pre-training Dataset"), ICLR 2025 version.

Quoted: "CBraMod is pre-trained on a very large public dataset, Temple University Hospital EEG corpus (TUEG) (Obeid & Picone, 2016). The TUEG dataset consists of a diverse archive of 69,652 clinical EEG recordings from 14,987 subjects across 26,846 sessions, with a total duration of 27,062 hours."

Preprocessing, as stated in the paper:
- Recordings of 5 minutes or less removed; first and last minute of each recording discarded.
- 19 common 10-20 channels selected (Fp1 ... O2).
- Band-pass 0.3-75 Hz; 60 Hz notch.
- Resampled to 200 Hz; segmented into 30-second non-overlapping samples.
- Bad-sample removal: samples with any amplitude above 100 µV dropped.
- Quoted: "there are 1,109,545 EEG samples retained for pre-training and longer than 9000 hours in total".

Gaps:
- The paper does not state the TUEG version number.
- The README does not describe which pretraining run (with or without bad-sample dropping) produced the released checkpoint. The paper's ablations compare these variants (the paper's ablation study, "clean pre-training" setting).

## 4. Overlap with downstream benchmarks

- TUEV (event type, 6-class; 112,491 5-s samples): the paper says "Given that the TUEV dataset is a subset of the pretraining dataset TUEG". The authors also report "CBraMod (excluding TUEV)", re-pretrained without TUEV.
- TUAB (abnormal detection, 2-class; 409,455 10-s samples): the paper says "Considering that the TUAB dataset is a subset of the pretraining dataset TUEG". The authors also report "CBraMod (excluding TUAB)", re-pretrained without TUAB.
- TUSZ: not mentioned anywhere in the paper.
- CHB-MIT (seizure detection, 2-class; 326,993 10-s samples): used as a downstream benchmark. The paper contains no statement on whether CHB-MIT overlaps with TUEG.
- Conclusion: the released checkpoint was pretrained on all of TUEG as described. Any downstream evaluation on TUEV or TUAB with the released weights is subject to the paper's stated overlap. Pretraining on TUEG and evaluating on TUEG-derived sets is in-distribution, not held-out.

## 5. License discrepancy

- GitHub code: MIT.
- HF model card (weights): Apache-2.0.
- These are separate licenses for code and weights. Which terms govern the weights is an open question (see section 6).

## 6. Open questions for counsel

1. Do the Temple University Hospital (TUH) EEG Corpus terms cover derivative model weights trained on TUEG, and does that cover use in a regulated commercial product? (TUH access terms were not reviewed in this session.)
2. Does a data-use agreement for TUEG restrict redistribution of weights, and do the authors' Apache-2.0 (HF) or MIT (GitHub) grants override or conflict with the TUH terms?
3. The HF owner `weighting666` is not the GitHub owner `wjq-learning`. Is the HF account authorized by the authors, and is its checkpoint the authors' official release?
4. Does the released checkpoint include TUEV/TUAB in pretraining (the paper's standard setting)? Only the "excluding" variants were ablated.
5. For regulated use (for example, medical-device claims), does pretraining on a clinical corpus with unverified version and filtering create a validation burden we must document?
6. Is an alternate braindecode-hosted copy (`braindecode/cbramod-pretrained`, referenced in braindecode docs) the same weights? Not verified here.
