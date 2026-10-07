# Study 2 montage geometries: vendor-sourced electrode lists

Compiled 2026-10-07 for Study 2 (research_plan_v1.txt, lines 248-257). Machine-readable version: `/home/user/SortingHat/configs/montages.yaml`.

Method note: FDA 510(k) text was read via innolitics.com mirrors of the FDA summaries (not accessdata.fda.gov PDFs); verify against the FDA originals before any regulatory use. The Masimo eIFU PDF exceeded the fetch size limit. Alias convention: T3=T7, T4=T8, T5=P7, T6=P8.

## Summary table

| Geometry | Electrodes (10-20 labels) | Reference / ground | Rate | Derivations | Confidence | Main source |
|---|---|---|---|---|---|---|
| full_19ch_1020 | Fp1 Fp2 F7 F3 Fz F4 F8 T3 C3 Cz C4 T4 T5 P3 Pz P4 T6 O1 O2 | dataset-dependent; bipolar analysis | n/a | Double banana (16) + Fz-Cz, Cz-Pz | high | standard convention |
| ceribell_headband_10el | Fp1 F7 T3 T5 O1 / Fp2 F8 T4 T6 O2 (10) | Reference and ground NOT found; display is bipolar | 250 Hz, 0.5-100 Hz | Fp1-F7, F7-T3, T3-T5, T5-O1, Fp2-F8, F8-T4, T4-T6, T6-O2 (8 ch) | medium | K210805, K191301, DECIDE 2020 |
| ceribell_headcap | 9 to 19 sites; labels not published | not found | not found | not found | low | K223086 |
| zeto_wr19 | 19 dry, "10-20" (labels not printed; assumed standard 19) + 1 DRL | up to 19 referential; reference unnamed; DRL ground site unnamed | 500 Hz | software montages | medium | K172735 |
| brainscope_ahead | Fp1 Fp2 Fpz AFz F7 F8 A1 A2 (8) | not specified | 1000 Hz, down to 100 Hz | not published | medium | K183241, K161068 |
| masimo_sedline_psi_legacy | Fp1 Fp2 F7 F8 | reference AFz; ground undocumented | not found | 4 ch referenced to AFz | medium | K033999, Drover 2004 |
| masimo_sedline_current | L1 L2 R1 R2 active; CT ref; CB ground (site mapping unverified) | CT / CB | not found | unknown | low | Masimo eIFU (via snippet) |
| medtronic_bis | unverified (Quatro = 4 electrodes incl. ground) | unknown | unknown | unknown | low | Medtronic product page |
| StatNet / BraiNet / Neuronostics / Persyst-compatible / EMU rapid-response / Rhythm | not found | not found | not found | not found | low (not researched to a source) | none |

## Notes per geometry

**Ceribell headband.** The K210805 comparison table lists 10 locations: Fp1, F7, T3, T5, O1, Fp2, F8, T4, T6, O2. Pocket EEG (K191301) is an 8-channel device taking 10 electrodes (5 left, 5 right). The DECIDE study (Vespa et al., Crit Care Med 2020;48:1249) describes a bipolar montage of five electrodes (four pairs) per hemisphere, with channels "approximately" Fp1-F7, F7-T3, T3-T5, T5-O1 and the right-sided equivalents, acquired at 250 Hz with a 0.5-100 Hz response. The Frontiers 2022 follow-up (10.3389/fneur.2022.915385) describes the same device as a ten-electrode, eight-channel bipolar, circumferential hairline montage without parasagittal or midline channels. No source gave the amplifier reference or ground site. Because the display is bipolar this does not change the simulation, but it matters for impedance and artifact modelling. Mapping: all 10 are standard 10-20 sites, so use the exact channel subset. "Approximately" means the fixed-spacing band may place sites slightly off nominal; run a displacement-sensitivity analysis.

**Ceribell headcap (K223086).** Only "9 to 19 Ag/AgCl electrodes, positioned per 10-20 or 10-10, depending on clinical need" was found. The yaml lists 19 as an assumed upper bound; it is equivalent to the reference geometry until the IFU is obtained. Do not present it as a distinct geometry yet.

**Zeto WR19 (K172735).** 19 dry signal electrodes plus one dedicated dry ground/DRL, placed per 10-20; 500 Hz; up to 19 referential channels with software montages. Per-site labels and the reference were not in the text retrieved, so the standard 19 is inferred. Zeto ONE (21 soft-tip electrodes) is a later device and not characterised here. The 510(k) gives hardware differences (dry electrodes), not a spatial difference.

**BrainScope Ahead.** The 510(k) technological-characteristics tables for Ahead 300 (K161068) and Ahead 400 (K183241) list Fp1, Fp2, Fpz, AFz, F7, F8, A1, A2 (International 10-20), a single-use Ag/AgCl solid-gel array, and 1 kHz sampling downsampled to 100 Hz. BrainScope describes an "8-electrode disposable headset", consistent with this list. Reference and ground are not stated, and A1/A2 are ear-lobe sites; the linked-ear reference in the yaml is a simulation assumption, not a vendor fact. Not in the 19-ch set: Fpz, AFz, A1, A2. Approximations: Fpz ~ (Fp1+Fp2)/2; AFz ~ 0.25 Fp1 + 0.25 Fp2 + 0.5 Fz; both better done by spherical-spline interpolation on the standard 10-20 sphere. A1/A2: use recorded values if the source has them, else average reference, flagged as a sensitivity analysis.

**Masimo SedLine / Physiometrix PSA.** The 2004 validation (Drover) tested Fp1, AFz, Fpz, Fp2, Cz, Pz, A1, A2, F7, F8 and chose Fp1, Fp2, F7, F8 referenced to AFz (4 channels). K033999 lists F7, F8, Fp1, Fp2, AFz as inputs. A later clinical-trial protocol (NCT03947060) lists Fp1, Fp2, F7, F8, Fpz and AFz, conflicting on Fpz. The current sensor is described (via search snippet of the Masimo eIFU) as 6 electrodes with 4 active channels L1, L2, R1, R2, reference CT, ground CB; I did not verify the mapping to 10-20 labels. Treat the current sensor as the legacy layout until confirmed.

**Medtronic BIS.** No source found that maps BIS Quatro or bilateral electrodes to 10-20 labels. The yaml entry is intentionally empty.

**Not found.** StatNet, BraiNet templates, Neuronostics/CereScope, Persyst-compatible headsets, EMU rapid-response products and Rhythm were not researched to a source in this pass.

## Open items before simulating

1. Ceribell IFU/service documentation for reference/ground sites and the exact headcap site list.
2. Masimo SedLine eIFU (lab-8473d) placement figure and BIS IFU diagram.
3. FDA accessdata PDFs for K210805, K161068, K183241, K172735 to confirm the mirror text.
4. Confirm whether the 19-ch source recordings contain A1/A2.

## Sources

- https://fda.innolitics.com/device/K210805 (Ceribell Instant EEG Headband)
- https://fda.innolitics.com/device/K191301 (Ceribell Pocket EEG)
- https://fda.innolitics.com/device/K223086 (Ceribell Instant EEG Headcap)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7735649/ (Vespa et al., DECIDE, Crit Care Med 2020)
- https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2022.915385/pdf
- https://fda.innolitics.com/device/K172735 (Zeto WR19)
- https://fda.innolitics.com/device/K161068 and https://fda.innolitics.com/device/K183241 (BrainScope Ahead)
- https://fda.innolitics.com/device/K033999 and the Drover 2004 PSI placement study at https://www.masimo.fr/siteassets/us/documents/pdf/clinical-evidence/sedline/drover-validation-of-the-eeg-electrode-placement-for-the-patient-state-index-oct-2004.pdf
- https://clinicaltrials.gov/study/NCT03947060 (SedLine sensor locations)
- https://techdocs.masimo.com/globalassets/techdocs/pdf/lab-8473d-eifu.pdf (not fully read)
- https://www.medtronic.com/en-us/healthcare-professionals/products/patient-monitoring/brain-monitoring/brain-sensors/bis-quatro-sensor.html
