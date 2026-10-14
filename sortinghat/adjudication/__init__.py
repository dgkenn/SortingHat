"""Gold adjudication pilot for the pre-specified ED-like intended-use stratum IU-large (D-157).

Modules
  aggregates   small-cell suppression helpers for the aggregate-only stdout / JSON of the adjudication scripts
  sampling     deterministic seeded stratified sample (site x silver-label pattern), sampling weights, duplicates
  blinding     EEG-blinding rules for packets: note-type exclusion, sentence redaction, structured-row blocks, final audit
  sources      readers for the OMOP tables a packet needs (local cache / store / in-memory)
  packets      EEG-blind rater packet assembly and writing (local_only, human readers only)
  crf          case report form (configs/adjudication_crf.yaml) -> REDCap data dictionary and rating-export parsing
  agreement    inter-/intra-rater agreement, silver-vs-gold, consensus, Gate 0 inputs (aggregate-only)

Rules that apply to everything here (CLAUDE.md): synthetic data only inside agent sessions; every printed or aggregate-file number
is suppressed below 11; record-level files (the sample, id map, packets, ratings) live only under a local_only/ directory with mode 0600
and are never printed or read by an agent; no LLM reads a note (rule 4): the packets are for HUMAN raters.
"""
