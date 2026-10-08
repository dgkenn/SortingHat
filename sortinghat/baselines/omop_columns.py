"""OMOP columns the baseline builder (``scripts/build_baselines.py``) reads, per table. Kept in the package so the shared OMOP cache
(``sortinghat.omop_cache``) can take the union over every step without importing a script."""
from __future__ import annotations

from .. import schema

MEAS_COLS = ["person_id", "measurement_datetime", "measurement_date", "measurement_source_value", "value_as_number",
             "unit_source_value", *schema.COLUMN_ALIASES["measurement.result_datetime"]]
DRUG_COLS = ["person_id", "drug_exposure_start_datetime", "drug_exposure_end_datetime", "drug_source_value", "quantity",
             "drug_concept_id"]
COND_COLS = ["person_id", "condition_start_datetime", "condition_source_value"]
PROC_COLS = ["person_id", "procedure_datetime", "procedure_date", "procedure_source_value"]
OBS_COLS = ["person_id", "observation_datetime", "observation_date", "observation_source_value", "value_as_string"]
