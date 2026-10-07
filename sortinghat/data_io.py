"""Data access for every dataset the SortingHat plan touches, ported from the earlier research code.

Covers (see ``docs/data_access.md`` for per-dataset method, gotchas and what was NOT found):

* BDSP S3 access points, by dataset name: HEEDB, I-CARE, MORGOTH data, other BDSP restricted/project prefixes
  (``bdsp_*`` helpers, ``DATASETS`` registry, ``read_csv_table`` / ``iter_omop_batches`` for HEEDB).
* PhysioNet over HTTPS: anonymous for open projects, Django-session login for credentialed ones
  (``physionet_*``). HTTP Basic auth does NOT work on PhysioNet file paths (source finding).
* VitalDB public API (``vitaldb_*``): gzip and BOM handling.
* OpenNeuro public S3 mirror over anonymous HTTPS (``openneuro_*``).
* TUH/NEDC rsync-over-SSH: command construction only (``tuh_rsync_argv``); nothing is executed here.
* Model-weight pins (``CBRAMOD``) and ``sha256_file``.

Sources in dgkenn/codex-playground- (research-program-continuation branch): ``analysis/heedb_omop_extract.py``,
``analysis/heedb_bs_*.py``, ``pipeline/stream_fetch.py``, ``pipeline/tuh_fetch.py``, ``common/awsenv.py``,
``bsde/src/bsde/ingestion/{vitaldb,openneuro_s3,http_edf}.py``, ``bsde/scripts/extract_mimic_sedation.py``,
``bsde/scripts/physionet_fetch.sh``.

HUMAN-RUN ONLY for anything credentialed. ``make_client`` and ``physionet_session`` raise
``RestrictedDataError`` inside an agent session (``CLAUDECODE=1`` / ``SORTINGHAT_AGENT_SESSION=1``), per
CLAUDE.md rule 2. Callers must also honour rule 3 (aggregate-only output): this module returns record-level
frames to *your* process and never prints them.

No credentials live here. They come from the standard boto3 chain (optionally pinned to a profile via the
``profile`` argument or ``HEEDB_AWS_PROFILE`` / ``AWS_PROFILE``), and for PhysioNet from ``PHYSIONET_USER`` /
``PHYSIONET_PASSWORD`` or a ``~/.netrc`` entry for ``physionet.org``. See ``docs/credentials_template.md``.

``boto3`` and ``pyarrow`` are imported lazily so importing this module (and unit-testing it with an injected
fake client) needs neither.
"""

from __future__ import annotations

import gzip
import hashlib
import http.cookiejar
import io
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pandas as pd

from . import schema
from .agent_safety import RestrictedDataError, assert_not_restricted_in_agent, in_agent_session

# BDSP exposes ACCESS POINTS, not buckets: the full ARN goes in ``Bucket=`` (the bare name gives NoSuchBucket,
# which reads like a permissions problem). Account 184438910517 is BDSP's, not yours. Each approved user can
# instead use their own access-point alias from the BDSP Cloud Credentials dashboard (env override below).
_ARN = "arn:aws:s3:us-east-1:184438910517:accesspoint/"
ACCESS_POINTS: dict[str, str] = {
    "credentialed": _ARN + "bdsp-credentialed-access-point",       # EEG/, OMOP/, EHR/, PSG/, ECG/, Imaging/, NAX/
    "restricted": _ARN + "bdsp-restricted-access-point",           # i-care/, burst-supression/, sparcnet_data/, ...
    "projects": _ARN + "bdsp-credentialed-projects-ap",            # morgoth1/, morgoth2/, icu-sleep/, sleepFM/, ...
}
DEFAULT_ACCESS_POINT = ACCESS_POINTS["credentialed"]
REGION = "us-east-1"


@dataclass(frozen=True)
class DatasetSpec:
    """Where a named dataset lives. ``status`` records how well the source code confirms it."""
    access_point: str          # key into ACCESS_POINTS
    prefix: str
    status: str                # "seen-in-code" or "documented-only"
    note: str = ""


# Dataset name -> location. Prefixes come from the source scripts and notes (docs/data_access.md).
DATASETS: dict[str, DatasetSpec] = {
    "heedb_metadata": DatasetSpec("credentialed", "EEG/HEEDB_Metadata/", "seen-in-code"),
    "heedb_eeg_metadata": DatasetSpec("credentialed", "EEG/eeg-metadata/", "seen-in-code"),
    "heedb_bids": DatasetSpec("credentialed", "EEG/bids/", "seen-in-code", "EDF; 12 BIDS datasets incl. Neurotech"),
    "heedb_omop": DatasetSpec("credentialed", "OMOP/Merged/", "seen-in-code", "parquet parts per OMOP table"),
    "icare": DatasetSpec("restricted", "ICARE_train/training/", "seen-in-code",
                         "per-patient dirs: <pid>.txt, <pid>_<seg>_<hr>_EEG.mat + .hea"),
    "burst_suppression_labels": DatasetSpec("restricted", "burst-supression/", "documented-only",
                                            "BDSP's spelling; 86 .mat with sample-level expert labels"),
    "sparcnet": DatasetSpec("restricted", "sparcnet_data/", "documented-only", "IIIC training data"),
    "sah": DatasetSpec("restricted", "sah/", "documented-only"),
    "e_cam_s": DatasetSpec("restricted", "e-cam-s/", "documented-only", "delirium cEEG"),
    "morgoth_data": DatasetSpec("projects", "morgoth1/data/", "seen-in-code",
                                "internal_dataset/<TASK>/..., pretrain/*.mat"),
    "morgoth_models": DatasetSpec("projects", "morgoth2/models/", "documented-only", "38 checkpoints, 1.39 GB (2026-05)"),
}


def ap_arn(name: str = "credentialed") -> str:
    """Access-point ARN by short name; env ``BDSP_ACCESS_POINT_<NAME>`` (or ``HEEDB_ACCESS_POINT`` for
    'credentialed') overrides it, e.g. with your own alias."""
    env = os.environ.get(f"BDSP_ACCESS_POINT_{name.upper()}")
    if not env and name == "credentialed":
        env = os.environ.get("HEEDB_ACCESS_POINT")
    return env or ACCESS_POINTS[name]


EEG_METADATA_PREFIX = "EEG/eeg-metadata/"
HEEDB_METADATA_PREFIX = "EEG/HEEDB_Metadata/"
BIDS_PREFIX = "EEG/bids/"
OMOP_MERGED_PREFIX = "OMOP/Merged/"

# Columns requested per OMOP table: the schema's list (CONFIRMED + NAMED + ASSUMED). A requested column that does
# not exist is silently skipped by ``iter_omop_batches`` (as in the source extractor), so use
# ``table_columns`` / the audit's ``--dry-run-schema`` to see what really exists.
OMOP_COLUMNS: dict[str, list[str]] = {
    t[len("omop_"):]: schema.columns(t) for t in schema.SCHEMA if t.startswith("omop_")}

# ASSUMED placeholder location. The dry run (2026-10-07) found NOTHING here; the real prefixes are
# Imaging/<SITE>/{BIDS,Clinical,Non-BIDS}/ for I0001 and I0004 (schema.IMAGING_REAL_LAYOUT), not listed deeper.
IMAGING_PREFIX = "Imaging/imaging_metadata/"


@dataclass(frozen=True)
class TableSpec:
    """Where a named table lives under the access point."""
    kind: str                      # "csv_site", "csv_global", "omop_parquet" or "parquet_dir"
    pattern: str                   # key / prefix; ``{site}`` and ``{table}`` are substituted
    fallbacks: tuple[str, ...] = ()


# Registry of table name -> location. Keys come from the source scripts (see docs/heedb_access.md).
TABLES: dict[str, TableSpec] = {
    # one row per EEG session; the date in the file name varies by release, so it is globbed
    "eeg_metadata": TableSpec("csv_site", EEG_METADATA_PREFIX + "{site}_eeg_metadata_"),
    # one row per EEG report; double underscore in the name (single-underscore variant seen as fallback)
    "reports_findings": TableSpec("csv_site", HEEDB_METADATA_PREFIX + "{site}_EEG__reports_findings.csv",
                                  (HEEDB_METADATA_PREFIX + "{site}_EEG_reports_findings.csv",)),
    "heedb_patients": TableSpec("csv_global", HEEDB_METADATA_PREFIX + "HEEDB_patients.csv"),
    "icd10_neurology": TableSpec("csv_global", HEEDB_METADATA_PREFIX + "HEEDB_ICD10_for_Neurology.csv"),
    "medication_atc": TableSpec("csv_global", HEEDB_METADATA_PREFIX + "HEEDB_Medication_ATC.csv"),
}
for _t in OMOP_COLUMNS:
    TABLES["omop_" + _t] = TableSpec("omop_parquet", OMOP_MERGED_PREFIX + "{table}/")
# ASSUMED location and layout (see IMAGING_PREFIX); parquet parts under one prefix
TABLES["imaging"] = TableSpec("parquet_dir", IMAGING_PREFIX)

_AWS_KEY_ID = re.compile(r"^(AKIA|ASIA)[A-Z0-9]{16}$")


def drop_placeholder_env(environ: dict[str, str] | None = None) -> bool:
    """Remove ambient ``AWS_*`` variables that are provably NOT real AWS credentials.

    Mirrors the source project's ``common/awsenv.py``: a sandbox may export a short proxy token as
    ``AWS_ACCESS_KEY_ID``; boto3 prefers it over ``~/.aws/credentials`` and every call then fails 403.
    A key that looks like a real one (20 chars, AKIA/ASIA) is never touched, and nothing is removed
    unless a shared credentials file exists to fall back to. Returns True if anything was removed.
    """
    env = os.environ if environ is None else environ
    key = env.get("AWS_ACCESS_KEY_ID", "")
    if not key or _AWS_KEY_ID.match(key):
        return False
    shared = env.get("AWS_SHARED_CREDENTIALS_FILE") or os.path.expanduser("~/.aws/credentials")
    if not os.path.exists(shared):
        return False
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(var, None)
    return True


def access_point(name: str = "credentialed") -> str:
    return ap_arn(name)


def make_client(profile: str | None = None, *, read_timeout: int = 600):
    """Build a boto3 S3 client for the BDSP access point (human-run processes only).

    Profile precedence: ``profile`` argument, ``HEEDB_AWS_PROFILE``, ``AWS_PROFILE``, else the default
    boto3 chain. ``payload_signing_enabled=False`` and a long read timeout match the source project's
    working configuration (``visit_occurrence`` is a single ~9 GB part).
    """
    if in_agent_session():
        raise RestrictedDataError(
            "heedb_io.make_client refused: restricted HEEDB access must run from a plain terminal "
            "or scheduler, not inside an agent session (CLAUDE.md rule 2).")
    import boto3
    from botocore.config import Config

    drop_placeholder_env()
    profile = profile or os.environ.get("HEEDB_AWS_PROFILE") or os.environ.get("AWS_PROFILE") or None
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    cfg = Config(s3={"payload_signing_enabled": False}, read_timeout=read_timeout, connect_timeout=30,
                 retries={"max_attempts": 8, "mode": "standard"})
    return session.client("s3", region_name=REGION, config=cfg)


def list_keys(s3, prefix: str, *, bucket: str | None = None, suffix: str | None = None) -> list[str]:
    """All keys under ``prefix`` (paginated), sorted."""
    bucket = bucket or access_point()
    keys: list[str] = []
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if suffix:
        keys = [k for k in keys if k.endswith(suffix)]
    return sorted(keys)


# A name is "id-like" when it could be a patient/session/date identifier. Such names are never printed.
_ID_LIKE_RE = re.compile(r"(^|[^A-Za-z])(sub|ses)-|\d{5,}|^\d+/?$|[0-9a-f]{12,}|\d{4}[-_]\d{2}[-_]\d{2}", re.I)
ID_LIKE = "<id-like name>"
MAX_LEVEL_ITEMS = 40          # a level with more prefixes than this is not listed at all (population-like)


_PART_RE = re.compile(r"^part-\d{5}(\.\w+)?$")      # generic parquet part numbering is not an identifier


def looks_id_like(name: str) -> bool:
    return not _PART_RE.match(name) and bool(_ID_LIKE_RE.search(name))


def list_level(s3, prefix: str = "", *, bucket: str | None = None, max_items: int = MAX_LEVEL_ITEMS) -> dict:
    """One metadata level under ``prefix`` (``Delimiter='/'``): names of child prefixes and files, NAMES ONLY.

    Id-like names are replaced by ``<id-like prefix>`` / ``<id-like file>``. A level with more than
    ``max_items`` prefixes (or files) is reported as ``overflow`` with NO names (it looks like a population
    of per-patient folders), and nothing beneath it is ever listed. No counts are returned.
    """
    bucket = bucket or access_point()
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
    cps = [c["Prefix"] for c in resp.get("CommonPrefixes", [])]
    files = [o["Key"] for o in resp.get("Contents", []) if o["Key"] != prefix]
    out: dict[str, Any] = {"prefix": prefix, "prefixes": [], "files": [], "prefixes_overflow": False,
                           "files_overflow": False}
    if len(cps) > max_items or (resp.get("IsTruncated") and len(cps) >= max_items):
        out["prefixes_overflow"] = True
    else:
        out["prefixes"] = [c if not looks_id_like(c[len(prefix):]) else prefix + "<id-like prefix>/" for c in cps]
    if len(files) > max_items or resp.get("IsTruncated") and len(files) >= max_items:
        out["files_overflow"] = True
    else:
        out["files"] = [f if not looks_id_like(f[len(prefix):]) else prefix + "<id-like file>" for f in files]
    return out


def resolve_key(s3, table: str, site: str | None = None, *, bucket: str | None = None) -> str:
    """Resolve a CSV table name (and site, for per-site tables) to one object key."""
    spec = TABLES[table]
    if spec.kind in ("omop_parquet", "parquet_dir"):
        raise ValueError(f"{table} is a directory of parquet parts; use iter_omop_batches / omop_parts")
    if spec.kind == "csv_site" and not site:
        raise ValueError(f"table {table!r} is per-site; pass site= (e.g. 'S0001')")
    fmt = lambda p: p.format(site=site)  # noqa: E731
    if table == "eeg_metadata":
        keys = list_keys(s3, fmt(spec.pattern), bucket=bucket, suffix=".csv")
        if not keys:
            raise FileNotFoundError(f"no eeg-metadata CSV for site {site}")
        return keys[-1]                    # newest dated file sorts last
    for cand in (spec.pattern, *spec.fallbacks):
        key = fmt(cand)
        if list_keys(s3, key, bucket=bucket):
            return key
    raise FileNotFoundError(f"{table} not found for site {site!r}")


def read_csv_table(table: str, site: str | None = None, *, s3=None, profile: str | None = None,
                   usecols: Iterable[str] | None = None, dtype: Any = str) -> pd.DataFrame:
    """Read one CSV table into a DataFrame. Every column is read as text unless ``dtype`` says otherwise.

    Text by default on purpose: the findings columns hold free labels, ``None``/``nan`` strings and
    empties, and IDs must not lose leading zeros. Use ``finding_present`` for the label convention.
    """
    s3 = s3 or make_client(profile)
    key = resolve_key(s3, table, site)
    body = s3.get_object(Bucket=access_point(), Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), dtype=dtype, usecols=None if usecols is None else list(usecols),
                       encoding="utf-8-sig", low_memory=False)


def read_site_table(table: str, site: str, *, s3=None, profile: str | None = None, dtype: Any = str) -> pd.DataFrame:
    """Read one site's ``eeg_metadata`` / ``reports_findings`` CSV and map its columns onto CANONICAL names.

    Reads the header first and loads only the columns that are not on the site's never-load list
    (``DeidentifiedName(Reports)`` is name-like text and is never read into memory), then applies
    ``normalise_site_table``. Raises ``FileNotFoundError`` when the site has no such file (I0008/I0009 have no
    reports_findings)."""
    s3 = s3 or make_client(profile)
    key = resolve_key(s3, table, site)
    skip = set(schema.never_load(table, site))
    use = [c for c in csv_header(s3, key) if c not in skip]
    body = s3.get_object(Bucket=access_point(), Key=key)["Body"].read()
    df = pd.read_csv(io.BytesIO(body), dtype=dtype, usecols=use, encoding="utf-8-sig", low_memory=False)
    return normalise_site_table(table, site, df)


def normalise_site_table(table: str, site: str, df: pd.DataFrame) -> pd.DataFrame:
    """Map a site's REAL columns onto the canonical names of ``schema.SCHEMA`` (``schema.SITE_VARIANTS``).

    * renames each site's header names (I0008/I0009: ``StartDateTime``->``StartTime``, ``EndDateTime``->``EndTime``,
      ``RecordingDuration``->``DurationInSeconds``; reports_findings ``CreationTime(EEG)``->``ReportCreationTime``...);
    * drops never-load columns (``DeidentifiedName(Reports)``) and unmapped duplicates (``N1``);
    * fills ``SiteID`` from the site code where the header has none (I0008/I0009 have ``InstituteID`` instead);
    * leaves canonical columns the site does not have ABSENT (never invents values). A site with no known variant
      is returned unchanged.
    Names only are touched; no value is interpreted except the SiteID fill (from the file name, not the data).
    """
    m = schema._variant_map(table, site)
    if m is None:
        return df
    out = df.drop(columns=[c for c in df.columns if m.get(c, c) is None], errors="ignore")
    out = out.rename(columns={a: c for a, c in m.items() if c is not None and a in out.columns and a != c})
    if table == "eeg_metadata" and ("SiteID" not in out.columns or out["SiteID"].isna().all()):
        out["SiteID"] = site
    return out


def denormalise_site_table(table: str, site: str, df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of ``normalise_site_table`` for the synthetic writer: emit exactly the site's real header (in the
    variant's order). Skipped header names (``DeidentifiedName(Reports)``, ``N1``) are written empty."""
    m = schema._variant_map(table, site)
    if m is None:
        return df
    return pd.DataFrame({a: (df[c] if c is not None and c in df.columns else None) for a, c in m.items()},
                        index=df.index)


def finding_present(series: pd.Series) -> pd.Series:
    """A reports_findings label cell is 'asserted' when non-empty and not the strings None/nan."""
    s = series.fillna("").astype(str).str.strip()
    return ~s.isin(["", "None", "nan"])


def bids_edf_key(site: str, bids_folder: str, session_id: str, eeg_folder: str | None) -> str:
    """Key of a session's EDF. Task is 'cEEG' when ``EEGFolder`` starts with 'ceeg' (any case), else 'EEG'.

    ``EEGFolder`` exists only in the S0001/S0002 headers (the I-sites have none), so for other sites ``eeg_folder``
    is None and the task token defaults to 'EEG' (UNVERIFIED for cEEG sessions there)."""
    task = "cEEG" if (eeg_folder or "").lower().startswith("ceeg") else "EEG"
    return (f"{BIDS_PREFIX}{site}/{bids_folder}/ses-{session_id}/eeg/"
            f"{bids_folder}_ses-{session_id}_task-{task}_eeg.edf")


def bids_folder_for(site: str, bdsp_patient_id: str | int) -> str:
    """``BidsFolder`` convention: ``sub-<SITE><BDSPPatientID>`` (the metadata BDSPPatientID can be blank)."""
    return f"sub-{site}{bdsp_patient_id}"


def parquet_parts(s3, prefix: str, *, bucket: str | None = None) -> list[str]:
    """Parquet part keys under any prefix (used for the ASSUMED ``Imaging/`` location)."""
    return list_keys(s3, prefix, bucket=bucket, suffix=".parquet")


def omop_parts(s3, table: str, *, bucket: str | None = None) -> list[str]:
    """Parquet part keys of a merged OMOP table (e.g. 181 parts for condition_occurrence)."""
    if table not in OMOP_COLUMNS:
        raise KeyError(f"unknown OMOP table {table!r}; known: {sorted(OMOP_COLUMNS)}")
    return list_keys(s3, OMOP_MERGED_PREFIX + f"{table}/", bucket=bucket, suffix=".parquet")


class S3RangeFile(io.RawIOBase):
    """Seekable read-only file over an S3 object using HTTP range requests (from the source extractor).

    Lets pyarrow read just the footer, selected columns and row groups of a multi-GB parquet part,
    so memory scales with a row group instead of the file.
    """

    def __init__(self, s3, bucket: str, key: str, size: int):
        self._s3, self._bucket, self._key, self._size, self._pos = s3, bucket, key, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, min(base + offset, self._size))
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        if n <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + n, self._size) - 1
        body = self._s3.get_object(Bucket=self._bucket, Key=self._key,
                                   Range=f"bytes={self._pos}-{end}")["Body"].read()
        self._pos += len(body)
        return body

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def iter_omop_batches(table: str, *, person_ids: Iterable[int] | None = None,
                      columns: list[str] | None = None, s3=None, profile: str | None = None,
                      batch_rows: int = 65536, prefix: str | None = None,
                      on_error: Callable[[str, Exception], None] | None = None) -> Iterator[Any]:
    """Stream a merged OMOP table part by part as pyarrow RecordBatches, optionally cohort-filtered.

    ``person_ids`` are integer OMOP ``person_id`` values; they equal ``int(BDSPPatientID)`` (source
    ``heedb_bs_ascertainment.py``). The filter is applied inside Arrow before any Python object is
    built. A part that fails is reported through ``on_error`` and skipped; the caller decides whether
    a partial result is acceptable (source catalogue rule 5: empty is not absence).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    s3 = s3 or make_client(profile)
    bucket = access_point()
    want = columns or (schema.columns(table) if table in schema.SCHEMA else OMOP_COLUMNS[table])
    pid_arr = pa.array(sorted(int(p) for p in person_ids), type=pa.int64()) if person_ids is not None else None
    parts = parquet_parts(s3, prefix, bucket=bucket) if prefix else omop_parts(s3, table, bucket=bucket)
    for key in parts:
        try:
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            pf = pq.ParquetFile(io.BufferedReader(S3RangeFile(s3, bucket, key, size), buffer_size=8 << 20))
            have = pf.schema_arrow.names
            use = [c for c in want if c in have]
            for batch in pf.iter_batches(batch_size=batch_rows, columns=use):
                if pid_arr is not None and "person_id" in use:
                    batch = batch.filter(pc.is_in(batch.column("person_id"), value_set=pid_arr).fill_null(False))
                if batch.num_rows:
                    yield batch
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never swallowed silently
            if on_error is None:
                raise
            on_error(key, exc)


# ---------------------------------------------------------------------------------------------------------
# Local directory mode (synthetic data, or a local mirror of the access-point layout) and schema probing
# ---------------------------------------------------------------------------------------------------------
class LocalStore:
    """Read-only stand-in for the S3 client over a local directory laid out like the access point
    (``EEG/eeg-metadata/...``, ``OMOP/Merged/<table>/*.parquet``). Implements only the calls this module
    makes (``list_objects_v2``, ``get_object`` with ``Range``, ``head_object``), so every reader here
    (``read_csv_table``, ``iter_omop_batches``, ``table_columns``) works unchanged on a directory.

    Refuses restricted paths inside an agent session (CLAUDE.md rule 1).
    """

    def __init__(self, root: str | Path):
        assert_not_restricted_in_agent(root)
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"data root is not a directory: {root}")

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise ValueError("key escapes the data root")
        return p

    def list_objects_v2(self, Bucket=None, Prefix: str = "", MaxKeys: int = 1000, ContinuationToken=None,
                        Delimiter: str | None = None):
        if Delimiter:                                     # one level only, like S3 (CommonPrefixes + Contents)
            base = self._path(Prefix.rstrip("/")) if Prefix.endswith("/") else self._path(Prefix.rsplit("/", 1)[0]) \
                if "/" in Prefix else self.root
            cps, files = [], []
            if base.is_dir():
                for f in sorted(base.iterdir()):
                    k = f.relative_to(self.root).as_posix()
                    if not k.startswith(Prefix):
                        continue
                    if f.is_dir():
                        cps.append({"Prefix": k + "/"})
                    else:
                        files.append({"Key": k})
            return {"CommonPrefixes": cps, "Contents": files, "IsTruncated": False}
        base = self._path(Prefix.rsplit("/", 1)[0]) if "/" in Prefix else self.root
        keys: list[str] = []
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    k = f.relative_to(self.root).as_posix()
                    if k.startswith(Prefix):
                        keys.append(k)
        return {"Contents": [{"Key": k} for k in sorted(keys)], "IsTruncated": False}

    def get_object(self, Bucket=None, Key: str = "", Range: str | None = None):
        p = self._path(Key)
        if not p.is_file():
            raise FileNotFoundError(Key)
        data = p.read_bytes() if Range is None else self._ranged(p, Range)
        return {"Body": io.BytesIO(data)}

    @staticmethod
    def _ranged(p: Path, rng: str) -> bytes:
        a, b = rng.removeprefix("bytes=").split("-")
        with open(p, "rb") as fh:
            fh.seek(int(a))
            return fh.read(int(b) - int(a) + 1)

    def head_object(self, Bucket=None, Key: str = ""):
        p = self._path(Key)
        if not p.is_file():
            raise FileNotFoundError(Key)
        return {"ContentLength": p.stat().st_size}


def open_store(data: str | Path | None = None, *, profile: str | None = None):
    """``LocalStore`` for a directory, else the real S3 client (human-run only; ``make_client`` refuses
    inside an agent session)."""
    return LocalStore(data) if data else make_client(profile)


def discover_sites(s3, *, bucket: str | None = None) -> list[str]:
    """Site codes that have an ``eeg-metadata`` CSV (e.g. S0001, I0002), from key names only."""
    keys = list_keys(s3, EEG_METADATA_PREFIX, bucket=bucket, suffix=".csv")
    sites = {m.group(1) for k in keys if (m := re.match(r"^([A-Za-z]\d{4})_eeg_metadata_", k.rsplit("/", 1)[-1]))}
    return sorted(sites)


def csv_header(s3, key: str, *, bucket: str | None = None, max_bytes: int = 4 << 20) -> list[str]:
    """Column names of a CSV object from its first line only (ranged read; values are never parsed)."""
    import csv as _csv
    bucket = bucket or access_point()
    n = 1 << 16
    while True:
        chunk = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{n - 1}")["Body"].read()
        if b"\n" in chunk or len(chunk) < n or n >= max_bytes:
            break
        n *= 4
    first = chunk.decode("utf-8-sig", "replace").split("\n", 1)[0].rstrip("\r")
    return next(_csv.reader([first]), [])


def parquet_column_names(s3, key: str, *, bucket: str | None = None) -> list[str]:
    """Column names of one parquet part from its footer only (ranged reads; no row data)."""
    import pyarrow.parquet as pq
    bucket = bucket or access_point()
    size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    pf = pq.ParquetFile(io.BufferedReader(S3RangeFile(s3, bucket, key, size), buffer_size=1 << 20))
    return list(pf.schema_arrow.names)


def table_columns(s3, table: str, site: str | None = None, *, bucket: str | None = None) -> list[str] | None:
    """Actual column names of a registered table (names only), or ``None`` when the table is not found.

    CSV tables: header line of the resolved key. Parquet tables: footer of the first part.
    """
    spec = TABLES[table]
    bucket = bucket or access_point()
    try:
        if spec.kind in ("csv_site", "csv_global"):
            key = resolve_key(s3, table, site, bucket=bucket) if spec.kind == "csv_site" else None
            if key is None:
                cands = (spec.pattern, *spec.fallbacks)
                key = next((c for c in cands if list_keys(s3, c, bucket=bucket)), None)
                if key is None:
                    return None
            return csv_header(s3, key, bucket=bucket)
        parts = (omop_parts(s3, table[len("omop_"):], bucket=bucket) if spec.kind == "omop_parquet"
                 else parquet_parts(s3, spec.pattern, bucket=bucket))
        return parquet_column_names(s3, parts[0], bucket=bucket) if parts else None
    except FileNotFoundError:
        return None


def parse_heedb_time(value: str | None):
    """Parse the timestamp formats seen in the source (with/without fractional seconds, date only)."""
    if not value or not str(value).strip():
        return pd.NaT
    return pd.to_datetime(str(value).strip()[:26], errors="coerce")


def save_local(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a frame to a gitignored local path (e.g. under ``data/restricted/``); never print it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False) if p.suffix == ".parquet" else df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------------------------------------
# Other BDSP datasets by name (I-CARE, MORGOTH data, restricted prefixes)
# ---------------------------------------------------------------------------------------------------------
def bdsp_list(dataset: str, sub: str = "", *, s3=None, profile: str | None = None,
              suffix: str | None = None) -> list[str]:
    """List keys of a registered BDSP dataset (``DATASETS``), under an optional sub-prefix.

    Human-run only (``make_client``). I-CARE example: ``bdsp_list("icare", "<pid>/", suffix="_EEG.mat")``.
    """
    spec = DATASETS[dataset]
    s3 = s3 or make_client(profile)
    return list_keys(s3, spec.prefix + sub, bucket=ap_arn(spec.access_point), suffix=suffix)


def bdsp_get(dataset: str, key_suffix: str, *, s3=None, profile: str | None = None,
             byte_range: tuple[int, int] | None = None) -> bytes:
    """Read one object of a registered BDSP dataset (optionally a byte range, inclusive)."""
    spec = DATASETS[dataset]
    s3 = s3 or make_client(profile)
    kw: dict[str, Any] = {"Bucket": ap_arn(spec.access_point), "Key": spec.prefix + key_suffix}
    if byte_range:
        kw["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return s3.get_object(**kw)["Body"].read()


def parse_icare_txt(text: str) -> dict[str, str]:
    """I-CARE per-patient ``<pid>.txt``: ``Key: value`` lines (Hospital, Age, Sex, ROSC, OHCA,
    Shockable Rhythm, TTM, Outcome, CPC). The outcome/CPC fields are 3-6 month prognostic labels, not
    contemporaneous state."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------------------------------------
# PhysioNet over HTTPS (source: bsde/scripts/extract_mimic_sedation.py, physionet_fetch.sh,
# bsde/docs/DEPOSIT_ACCESS_STATUS.md)
# ---------------------------------------------------------------------------------------------------------
PHYSIONET = "https://physionet.org"


def physionet_url(slug: str, version: str, path: str = "") -> str:
    """File URL of a project version, e.g. ``physionet_url('eegmmidb', '1.0.0', 'S001/S001R01.edf')``.

    Check the CURRENT version first: a stale version returns 403, indistinguishable from a missing DUA.
    """
    return f"{PHYSIONET}/files/{slug}/{version}/{path.lstrip('/')}"


def _physionet_credentials() -> tuple[str, str]:
    user, pw = os.environ.get("PHYSIONET_USER"), os.environ.get("PHYSIONET_PASSWORD")
    if not (user and pw):
        import netrc
        try:
            auth = netrc.netrc().authenticators("physionet.org")
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if auth:
            user, pw = auth[0], auth[2]
    if not (user and pw):
        raise RuntimeError("PhysioNet credentials not found: set PHYSIONET_USER / PHYSIONET_PASSWORD or add a "
                           "physionet.org entry to ~/.netrc (see docs/credentials_template.md).")
    return user, pw


def physionet_session():
    """Logged-in urllib opener for credentialed PhysioNet projects (human-run only).

    PhysioNet needs a Django SESSION COOKIE. HTTP Basic auth, including a ``~/.netrc`` handed straight to
    wget/curl, returns 403 on ``/files/`` paths (measured in the source). So: GET /login/, read the CSRF
    token, POST username/password, keep the cookie jar. Access is granted PER PROJECT: 403 with a valid
    login means that project's DUA is not signed (or the version path is stale).
    """
    if in_agent_session():
        raise RestrictedDataError("physionet_session refused inside an agent session (CLAUDE.md rule 2).")
    user, pw = _physionet_credentials()
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    op.addheaders = [("User-Agent", "Mozilla/5.0"), ("Referer", f"{PHYSIONET}/login/")]
    html = op.open(f"{PHYSIONET}/login/", timeout=60).read().decode("utf8", "replace")
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', html)
    if not m:
        raise RuntimeError("PhysioNet login page had no CSRF token")
    data = urllib.parse.urlencode({"csrfmiddlewaretoken": m.group(1), "username": user, "password": pw}).encode()
    op.open(urllib.request.Request(f"{PHYSIONET}/login/", data=data), timeout=60)
    if not any(c.name == "sessionid" for c in jar):
        raise RuntimeError("PhysioNet login produced no session cookie (bad credentials?)")
    return op


def http_get(url: str, *, opener=None, byte_range: tuple[int, int] | None = None, timeout: float = 120.0) -> bytes:
    """GET (optionally a byte range) with an optional logged-in opener; works for any HTTPS source."""
    headers = {"User-Agent": "sortinghat/0.0.1"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    return (opener.open(req, timeout=timeout) if opener else urllib.request.urlopen(req, timeout=timeout)).read()


# ---------------------------------------------------------------------------------------------------------
# VitalDB public API (source: bsde/src/bsde/ingestion/vitaldb.py). No credentials.
# ---------------------------------------------------------------------------------------------------------
VITALDB_API = "https://api.vitaldb.net"


def vitaldb_decode(blob: bytes) -> str:
    """The API gzips regardless of Accept-Encoding, and ``/cases`` starts with a UTF-8 BOM that would rename
    the first column to '\\ufeffcaseid'. Decompress by magic bytes and decode ``utf-8-sig``."""
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    return blob.decode("utf-8-sig", "replace")


def vitaldb_fetch(path: str, timeout: float = 300.0) -> str:
    """Fetch an API path (``cases``, ``trks``, ``labs``, or a track id) as text."""
    return vitaldb_decode(http_get(f"{VITALDB_API}/{path.lstrip('/')}", timeout=timeout))


def vitaldb_table(path: str) -> pd.DataFrame:
    """``cases`` / ``trks`` / ``labs`` as a DataFrame. Cluster on ``subjectid``, not ``caseid`` (237 of 6,388
    cases share a patient). License on api.vitaldb.net is CC BY-NC-SA 4.0, R&D only."""
    return pd.read_csv(io.StringIO(vitaldb_fetch(path)), low_memory=False)


# ---------------------------------------------------------------------------------------------------------
# OpenNeuro public S3 mirror over anonymous HTTPS (source: bsde/src/bsde/ingestion/openneuro_s3.py)
# ---------------------------------------------------------------------------------------------------------
OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def openneuro_url(accession: str, key: str = "") -> str:
    return f"{OPENNEURO_BASE}/{accession}/{key.lstrip('/')}"


def parse_list_objects_v2_xml(xml_text: str) -> tuple[list[str], str | None, bool]:
    """(keys, next continuation token, is_truncated) from an S3 ListObjectsV2 XML page."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_text)
    keys = [e.text or "" for e in root.iter(f"{_S3_NS}Key")]
    tok = root.find(f"{_S3_NS}NextContinuationToken")
    trunc = root.find(f"{_S3_NS}IsTruncated")
    return keys, (tok.text if tok is not None else None), bool(trunc is not None and trunc.text == "true")


def openneuro_list(accession: str, prefix: str = "") -> list[str]:
    """All keys of a dataset (or sub-prefix) from the public mirror; no credentials."""
    keys: list[str] = []
    token = None
    while True:
        q = {"list-type": "2", "prefix": f"{accession}/{prefix}"}
        if token:
            q["continuation-token"] = token
        page = http_get(f"{OPENNEURO_BASE}/?{urllib.parse.urlencode(q)}", timeout=30)
        got, token, trunc = parse_list_objects_v2_xml(page.decode("utf8"))
        keys += got
        if not trunc:
            return keys


# ---------------------------------------------------------------------------------------------------------
# TUH / NEDC (source: pipeline/tuh_fetch.py). Command construction only; nothing is executed.
# ---------------------------------------------------------------------------------------------------------
TUH_HOST = "www.isip.piconepress.com"
TUH_USER = "nedc-tuh-eeg"
TUH_ROOT = "data/tuh_eeg"


def tuh_rsync_argv(remote_path: str, dest: str, ssh_key: str = "~/.ssh/id_ed25519") -> list[str]:
    """argv for the NEDC rsync-over-SSH transport (your own NEDC-registered key; port 22 must be open)."""
    key = os.path.expanduser(ssh_key)
    ssh = f"ssh -i {key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
    return ["rsync", "-auvxL", "-e", ssh, f"{TUH_USER}@{TUH_HOST}:{remote_path}", dest]


# ---------------------------------------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------------------------------------
CBRAMOD = {
    "repo_id": "weighting666/CBraMod",           # Hugging Face; original-author checkpoint
    "file": "pretrained_weights.pth",            # 19.8 MB
    "sha256": "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",   # pin from the source config.yaml
    "units": "microvolts",                       # mne returns volts: multiply by 1e6
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
