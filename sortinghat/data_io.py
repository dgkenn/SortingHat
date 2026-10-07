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

import bisect
import gzip
import hashlib
import http.cookiejar
import io
import os
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------------------------------------
# Resilient S3 reads: every GET (and the Body read, which is where a dropped stream surfaces) is retried
# ---------------------------------------------------------------------------------------------------------
# A large read through the sandbox proxy can drop mid-stream (botocore ResponseStreamingError wrapping an
# IncompleteRead). boto3's own retry layer does NOT cover errors raised while reading ``Body``, so each read here
# is wrapped: exception classes are matched by NAME (botocore / urllib3 / requests are not imported), full object
# reads are split into ranged chunks so a retry re-fetches little, and a short body counts as a failed attempt.
_RETRY_NAMES = frozenset({
    "ResponseStreamingError", "IncompleteRead", "IncompleteReadError", "ReadTimeoutError", "ReadTimeout",
    "ConnectTimeoutError", "ConnectTimeout", "ConnectionError", "ConnectionClosedError", "EndpointConnectionError",
    "ProtocolError", "ProxyError", "ProxyConnectionError", "SSLError", "ChunkedEncodingError", "HTTPClientError",
    "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError", "RemoteDisconnected", "TimeoutError",
    "ShortReadError"})
_RETRY_CODES = frozenset({"SlowDown", "Throttling", "ThrottlingException", "RequestTimeout", "RequestTimeoutException",
                          "RequestLimitExceeded", "InternalError", "ServiceUnavailable", "TooManyRequestsException",
                          "BandwidthLimitExceeded", "429", "500", "502", "503", "504"})
_NO_RETRY_OS = (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)
RETRY_COUNTS: Counter = Counter()          # exception CLASS name -> retries so far (a count, safe to print)


class ShortReadError(OSError):
    """A body shorter than the requested range (a silently truncated stream)."""


def is_retryable(exc: BaseException) -> bool:
    """True for dropped streams / connection problems / throttling / 5xx; False for not-found, denied, bad input."""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):                         # botocore ClientError
        code = str(resp.get("Error", {}).get("Code", ""))
        status = resp.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in _RETRY_CODES or (isinstance(status, int) and (status >= 500 or status == 429))
    if isinstance(exc, _NO_RETRY_OS):
        return False
    if any(c.__name__ in _RETRY_NAMES for c in type(exc).__mro__):
        return True
    return isinstance(exc, (OSError, EOFError))        # ConnectionError, TimeoutError, pyarrow ArrowIOError, ...


@dataclass
class RetryPolicy:
    """Attempts, exponential backoff with jitter (half to full of the cap), injectable clock for tests."""
    max_attempts: int = 8
    backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    sleep: Callable[[float], None] = time.sleep
    rand: Callable[[], float] = random.random

    def delay(self, attempt: int) -> float:
        cap = min(self.max_backoff_s, self.backoff_s * (2 ** (attempt - 1)))
        return cap * (0.5 + 0.5 * self.rand())


DEFAULT_RETRY = RetryPolicy()
GET_CHUNK_BYTES = 16 << 20           # one ranged GET is at most this big, so a dropped stream re-fetches little


def with_retries(fn: Callable[[], Any], policy: RetryPolicy | None = None) -> Any:
    """Call ``fn()`` until it succeeds, retrying retryable exceptions with jittered exponential backoff."""
    p = policy or DEFAULT_RETRY
    for attempt in range(max(1, p.max_attempts)):
        if attempt:
            p.sleep(p.delay(attempt))
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if not is_retryable(exc) or attempt + 1 >= max(1, p.max_attempts):
                raise
            RETRY_COUNTS[type(exc).__name__] += 1
    raise AssertionError("unreachable")


_CR_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")


def _content_range_total(value) -> int | None:
    m = _CR_RE.match(str(value or "").strip())
    return int(m.group(3)) if m and m.group(3) != "*" else None


def get_bytes(s3, bucket: str, key: str, start: int | None = None, end: int | None = None, *,
              expected: int | None = None, policy: RetryPolicy | None = None) -> tuple[bytes, int | None]:
    """One GET (ranged when ``start`` is given; ``end`` inclusive) with the body read INSIDE the retry, so a dropped
    stream is retried. A body of the wrong length (``expected``, else the response ``ContentLength``) is retried too.
    Returns ``(body, object size from Content-Range or None)``."""
    def once():
        kw: dict[str, Any] = {"Bucket": bucket, "Key": key}
        if start is not None:
            kw["Range"] = f"bytes={start}-{end}"
        resp = s3.get_object(**kw)
        body = resp["Body"].read()
        want = expected
        if want is None and isinstance(resp.get("ContentLength"), int):
            want = resp["ContentLength"]
        if want is not None and len(body) != want:
            raise ShortReadError("short body")
        return body, _content_range_total(resp.get("ContentRange"))
    return with_retries(once, policy)


def _is_invalid_range(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    return isinstance(resp, dict) and (str(resp.get("Error", {}).get("Code", "")) == "InvalidRange"
                                       or resp.get("ResponseMetadata", {}).get("HTTPStatusCode") == 416)


def read_object(s3, bucket: str, key: str, *, chunk: int = GET_CHUNK_BYTES, policy: RetryPolicy | None = None) -> bytes:
    """Whole object as bytes via consecutive ranged GETs of ``chunk`` bytes, each retried independently."""
    out: list[bytes] = []
    start, total = 0, None
    while True:
        expected = None if total is None else min(chunk, total - start)
        try:
            body, t = get_bytes(s3, bucket, key, start, start + chunk - 1, expected=expected, policy=policy)
        except Exception as exc:  # noqa: BLE001
            if _is_invalid_range(exc):                  # start is at/after the end (or the object is empty)
                break
            raise
        total = t if t is not None else total
        out.append(body)
        start += len(body)
        if not body or (total is not None and start >= total) or (total is None and len(body) < chunk):
            break
    return b"".join(out)


def head_size(s3, bucket: str, key: str, *, policy: RetryPolicy | None = None) -> int:
    return int(with_retries(lambda: s3.head_object(Bucket=bucket, Key=key), policy)["ContentLength"])


def list_keys(s3, prefix: str, *, bucket: str | None = None, suffix: str | None = None,
              policy: RetryPolicy | None = None) -> list[str]:
    """All keys under ``prefix`` (paginated), sorted."""
    bucket = bucket or access_point()
    keys: list[str] = []
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = with_retries(lambda kw=kw: s3.list_objects_v2(**kw), policy)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    if suffix:
        keys = [k for k in keys if k.endswith(suffix)]
    return sorted(keys)


# A name is "id-like" when it could be a patient/session/date identifier. Such names are never printed.
_ID_LIKE_RE = re.compile(r"(^|[^A-Za-z])(sub|ses)-|\d{5,}|^\d+/?$|[0-9a-f]{12,}|\d{4}[-_]\d{2}[-_]\d{2}|\d{1,2}(st|nd|rd|th)?[A-Za-z]{3,9}\d{4}", re.I)
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
                   usecols: Iterable[str] | None = None, dtype: Any = str,
                   retry: RetryPolicy | None = None) -> pd.DataFrame:
    """Read one CSV table into a DataFrame. Every column is read as text unless ``dtype`` says otherwise.

    Text by default on purpose: the findings columns hold free labels, ``None``/``nan`` strings and
    empties, and IDs must not lose leading zeros. Use ``finding_present`` for the label convention.
    """
    s3 = s3 or make_client(profile)
    key = resolve_key(s3, table, site)
    body = read_object(s3, access_point(), key, policy=retry)         # chunked ranged GETs, each retried
    return pd.read_csv(io.BytesIO(body), dtype=dtype, usecols=None if usecols is None else list(usecols),
                       encoding="utf-8-sig", low_memory=False)


def read_site_table(table: str, site: str, *, s3=None, profile: str | None = None, dtype: Any = str,
                    retry: RetryPolicy | None = None) -> pd.DataFrame:
    """Read one site's ``eeg_metadata`` / ``reports_findings`` CSV and map its columns onto CANONICAL names.

    Reads the header first and loads only the columns that are not on the site's never-load list
    (``DeidentifiedName(Reports)`` is name-like text and is never read into memory), then applies
    ``normalise_site_table``. Raises ``FileNotFoundError`` when the site has no such file (I0008/I0009 have no
    reports_findings)."""
    s3 = s3 or make_client(profile)
    key = resolve_key(s3, table, site)
    skip = set(schema.never_load(table, site))
    use = [c for c in csv_header(s3, key, policy=retry) if c not in skip]
    body = read_object(s3, access_point(), key, policy=retry)         # chunked ranged GETs, each retried
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


# ---------------------------------------------------------------------------------------------------------
# Robust EDF key resolution (documented pattern first; then listing ONLY the recording's own folder)
# ---------------------------------------------------------------------------------------------------------
# Pattern NAMES are the only thing diagnostics may report (never keys).
EDF_PATTERNS = ("documented", "alt_task", "no_task", "documented+sid_variant", "alt_task+sid_variant",
                "no_task+sid_variant", "folder_listing", "patient_listing", "not_found")


def _sid_variants(session_id) -> list[str]:
    """The SessionID as given, then spellings a CSV round trip can produce (``12.0`` -> ``12``; padding stripped)."""
    sid = str(session_id).strip()
    out = [sid]
    if re.fullmatch(r"\d+\.0+", sid):
        out.append(sid.split(".")[0])
    base = out[-1]
    if base.isdigit() and base != base.lstrip("0") and base.lstrip("0"):
        out.append(base.lstrip("0"))
    return list(dict.fromkeys(out))


def _bids_names(bids_folder) -> list[str]:
    b = str(bids_folder).strip()
    return [b] if b.startswith("sub-") else [b, "sub-" + b]


def bids_edf_candidates(site: str, bids_folder: str, session_id, eeg_folder: str | None = None) -> list[tuple[str, str]]:
    """Ordered ``(pattern name, key)`` candidates: the documented key (task token from ``EEGFolder``), the other task
    token, and a task-less BIDS name; then the same with SessionID spelling variants (``12.0`` -> ``12``)."""
    first = "cEEG" if (eeg_folder or "").lower().startswith("ceeg") else "EEG"
    other = "EEG" if first == "cEEG" else "cEEG"
    out: list[tuple[str, str]] = []
    for bf in _bids_names(bids_folder):
        for n, sid in enumerate(_sid_variants(session_id)):
            base = f"{BIDS_PREFIX}{site}/{bf}/ses-{sid}/eeg/{bf}_ses-{sid}"
            suffix = "" if n == 0 else "+sid_variant"
            out += [("documented" + suffix, f"{base}_task-{first}_eeg.edf"),
                    ("alt_task" + suffix, f"{base}_task-{other}_eeg.edf"),
                    ("no_task" + suffix, f"{base}_eeg.edf")]
    return list(dict.fromkeys(out))


_NOT_FOUND_CODES = {"NoSuchKey", "NotFound", "404", "NoSuchBucket", "AccessDenied", "403", "Forbidden"}


def key_exists(s3, key: str, *, bucket: str | None = None, policy: RetryPolicy | None = None) -> bool:
    """Whether one exact key exists (HEAD, else an exact-prefix listing). Missing / denied is False; transient errors
    are retried and then raised."""
    bucket = bucket or access_point()
    try:
        if hasattr(s3, "head_object"):
            with_retries(lambda: s3.head_object(Bucket=bucket, Key=key), policy)
            return True
        resp = with_retries(lambda: s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1), policy)
        return any(o["Key"] == key for o in resp.get("Contents", []))
    except FileNotFoundError:
        return False
    except Exception as exc:  # noqa: BLE001
        resp = getattr(exc, "response", None)
        if isinstance(resp, dict) and (str(resp.get("Error", {}).get("Code", "")) in _NOT_FOUND_CODES
                                       or resp.get("ResponseMetadata", {}).get("HTTPStatusCode") in (403, 404)):
            return False
        raise


def file_ext(name: str) -> str:
    """Technical extension of a file name: everything after the first '.' of the last path segment, lower case
    (``x_eeg.edf`` -> ``edf``, ``x.edf.gz`` -> ``edf.gz``); ``other`` when absent or odd."""
    seg = name.rsplit("/", 1)[-1]
    ext = seg.split(".", 1)[1].lower() if "." in seg else ""
    return ext if re.fullmatch(r"[a-z0-9]{1,8}(\.[a-z0-9]{1,8})?", ext) else "other"


@dataclass
class FolderListing:
    """One folder level (``Delimiter='/'``): whether it holds anything, file-extension counts, the .edf names."""
    exists: bool = False
    ext_counts: Counter = field(default_factory=Counter)
    edf_keys: list[str] = field(default_factory=list)
    subfolders: list[str] = field(default_factory=list)


def list_folder(s3, prefix: str, *, bucket: str | None = None, policy: RetryPolicy | None = None) -> FolderListing:
    """List ONE folder (``prefix`` ends with '/'), one level, paginated. Names stay in memory; callers report counts."""
    bucket = bucket or access_point()
    out = FolderListing()
    token = None
    while True:
        kw: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = with_retries(lambda kw=kw: s3.list_objects_v2(**kw), policy)
        for o in resp.get("Contents", []):
            if o["Key"] == prefix:
                continue
            out.exists = True
            ext = file_ext(o["Key"])
            out.ext_counts[ext] += 1
            if ext == "edf":
                out.edf_keys.append(o["Key"])
        subs = [c["Prefix"] for c in resp.get("CommonPrefixes", [])]
        out.subfolders += subs
        out.exists = out.exists or bool(subs)
        if not resp.get("IsTruncated"):
            return out
        token = resp.get("NextContinuationToken")


def pick_edf(keys: list[str], bids_folder: str, session_id, task: str) -> str | None:
    """The session's EDF among ``keys``: prefer a name starting ``<BidsFolder>_ses-<id>``, then the expected task."""
    if not keys:
        return None
    sids = _sid_variants(session_id)

    def score(k: str) -> tuple:
        name = k.rsplit("/", 1)[-1]
        starts = any(name.startswith(f"{b}_ses-{sid}") for b in _bids_names(bids_folder) for sid in sids)
        return (not starts, f"task-{task}_" not in name, name)
    return min(keys, key=score)


@dataclass
class EdfResolution:
    key: str | None                      # NEVER print
    pattern: str = "not_found"           # one of EDF_PATTERNS
    folder_exists: bool | None = None    # session eeg/ folder (None = not listed)
    n_edf: int | None = None             # .edf files in that folder (None = not listed)
    ext_counts: Counter = field(default_factory=Counter)

    @property
    def found(self) -> bool:
        return self.key is not None


def resolve_edf_key(s3, site: str, bids_folder: str, session_id, eeg_folder: str | None = None, *,
                    bucket: str | None = None, list_fallback: bool = True, patient_fallback: bool = False,
                    always_list: bool = False, policy: RetryPolicy | None = None) -> EdfResolution:
    """Resolve one session's EDF key robustly.

    1. the documented key, the other task token and a task-less name (``bids_edf_candidates``), by exact-key HEAD;
    2. ``list_fallback``: list ONLY the session's own folder ``EEG/bids/<site>/<BidsFolder>/ses-<id>/eeg/``
       (``Delimiter='/'``) and take the ``.edf`` whose name matches the session (``pick_edf``);
    3. ``patient_fallback`` (off by default): list the subject's own folder for a ``ses-`` folder whose id equals the
       session id under a spelling variant, then list that session's ``eeg/``.
    ``always_list`` also lists the session folder when step 1 already found the key, to fill ``folder_exists`` /
    ``n_edf`` / ``ext_counts`` for diagnostics. Keys are returned, never printed or logged here.
    """
    bucket = bucket or access_point()
    task = "cEEG" if (eeg_folder or "").lower().startswith("ceeg") else "EEG"
    res = EdfResolution(None)
    for pat, key in bids_edf_candidates(site, bids_folder, session_id, eeg_folder):
        if key_exists(s3, key, bucket=bucket, policy=policy):
            res.key, res.pattern = key, pat
            break
    if (res.key is None and list_fallback) or always_list:
        listing = FolderListing()
        for bf in _bids_names(bids_folder):
            for sid in _sid_variants(session_id):
                listing = list_folder(s3, f"{BIDS_PREFIX}{site}/{bf}/ses-{sid}/eeg/", bucket=bucket, policy=policy)
                if listing.exists:
                    break
            if listing.exists:
                break
        res.folder_exists, res.n_edf, res.ext_counts = listing.exists, len(listing.edf_keys), listing.ext_counts
        if res.key is None:
            k = pick_edf(listing.edf_keys, bids_folder, session_id, task)
            if k is not None:
                res.key, res.pattern = k, "folder_listing"
    if res.key is None and patient_fallback:
        for bf in _bids_names(bids_folder):
            top = list_folder(s3, f"{BIDS_PREFIX}{site}/{bf}/", bucket=bucket, policy=policy)
            if not top.exists:
                continue
            want = set(_sid_variants(session_id)) | {v.lstrip("0") for v in _sid_variants(session_id)}
            for cp in top.subfolders:
                seg = cp.rstrip("/").rsplit("/", 1)[-1]
                if seg.startswith("ses-") and (seg[4:] in want or seg[4:].lstrip("0") in want):
                    lst = list_folder(s3, cp + "eeg/", bucket=bucket, policy=policy)
                    k = pick_edf(lst.edf_keys, bids_folder, session_id, task)
                    if k is not None:
                        res.key, res.pattern = k, "patient_listing"
                        res.folder_exists, res.n_edf, res.ext_counts = True, len(lst.edf_keys), lst.ext_counts
                        break
            if res.key is not None:
                break
    return res


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
    so memory scales with a row group instead of the file. Every read is split into ranged GETs of at most
    ``max_get_bytes``; each GET (body read included) is retried with jittered backoff on dropped streams
    (``ResponseStreamingError``, ``IncompleteRead``), connection errors, read timeouts, throttling and 5xx, and a
    short body is retried too (see ``get_bytes``).
    """

    def __init__(self, s3, bucket: str, key: str, size: int, *, policy: RetryPolicy | None = None,
                 max_get_bytes: int = GET_CHUNK_BYTES):
        self._s3, self._bucket, self._key, self._size, self._pos = s3, bucket, key, size, 0
        self._policy, self._max_get = policy, max(1, int(max_get_bytes))

    @classmethod
    def open(cls, s3, bucket: str, key: str, *, policy: RetryPolicy | None = None,
             max_get_bytes: int = GET_CHUNK_BYTES) -> "S3RangeFile":
        return cls(s3, bucket, key, head_size(s3, bucket, key, policy=policy), policy=policy,
                   max_get_bytes=max_get_bytes)

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
        n = min(n, self._size - self._pos)
        if n <= 0:
            return b""
        parts = []
        while n > 0:
            take = min(n, self._max_get)
            body, _ = get_bytes(self._s3, self._bucket, self._key, self._pos, self._pos + take - 1, expected=take,
                                policy=self._policy)
            parts.append(body)
            self._pos += take
            n -= take
        return parts[0] if len(parts) == 1 else b"".join(parts)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def _outer(policy: RetryPolicy | None) -> RetryPolicy:
    """Policy for the row-group / open layer that wraps ``S3RangeFile``'s own per-GET retries: only a couple of
    extra attempts, for failures that pyarrow re-raises after the per-GET retries are exhausted."""
    p = policy or DEFAULT_RETRY
    return RetryPolicy(min(3, p.max_attempts), p.backoff_s, p.max_backoff_s, p.sleep, p.rand)


def open_parquet(s3, bucket: str, key: str, *, policy: RetryPolicy | None = None, buffer_size: int = 1 << 20,
                 max_get_bytes: int = GET_CHUNK_BYTES):
    """``pyarrow.parquet.ParquetFile`` over a retrying ranged-GET file."""
    import pyarrow.parquet as pq
    raw = S3RangeFile.open(s3, bucket, key, policy=policy, max_get_bytes=max_get_bytes)
    return pq.ParquetFile(io.BufferedReader(raw, buffer_size=buffer_size))


def _rowgroup_may_match(pf, rg: int, col: str, sorted_ids: list[int]) -> bool:
    """False only when the row group's min/max statistics for ``col`` prove that none of ``sorted_ids`` is inside."""
    try:
        md = pf.metadata.row_group(rg)
        for j in range(md.num_columns):
            if md.column(j).path_in_schema == col:
                st = md.column(j).statistics
                if st is None or not st.has_min_max:
                    return True
                i = bisect.bisect_left(sorted_ids, int(st.min))
                return i < len(sorted_ids) and sorted_ids[i] <= int(st.max)
    except Exception:  # noqa: BLE001 - statistics are an optimisation only (e.g. string-typed ids)
        pass
    return True


def iter_omop_batches(table: str, *, person_ids: Iterable[int] | None = None,
                      columns: list[str] | None = None, s3=None, profile: str | None = None,
                      batch_rows: int = 65536, prefix: str | None = None,
                      on_error: Callable[[str, Exception], None] | None = None,
                      retry: RetryPolicy | None = None, max_get_bytes: int = GET_CHUNK_BYTES,
                      buffer_size: int = 1 << 20) -> Iterator[Any]:
    """Stream a merged OMOP table part by part as pyarrow RecordBatches, optionally cohort-filtered.

    Reading is by ROW GROUP over retrying ranged GETs (``S3RangeFile``): column-pruned, and with ``person_ids`` the
    row group is skipped from its ``person_id`` min/max statistics, else only ``person_id`` is read first and the
    other columns are fetched only if some id matches (then filtered to those rows in Arrow, before pandas).
    ``person_ids`` are integer OMOP ``person_id`` values; they equal ``int(BDSPPatientID)`` (source
    ``heedb_bs_ascertainment.py``). A row-group read that fails is retried (``retry``); a part that still fails is
    reported through ``on_error`` and skipped; the caller decides whether a partial result is acceptable (source
    catalogue rule 5: empty is not absence).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    s3 = s3 or make_client(profile)
    bucket = access_point()
    want = columns or (schema.columns(table) if table in schema.SCHEMA else OMOP_COLUMNS[table])
    ids = sorted({int(p) for p in person_ids}) if person_ids is not None else None
    pid_arr = pa.array(ids, type=pa.int64()) if ids is not None else None
    parts = parquet_parts(s3, prefix, bucket=bucket) if prefix else omop_parts(s3, table, bucket=bucket)
    outer = _outer(retry)
    for key in parts:
        try:
            pf = with_retries(lambda key=key: open_parquet(s3, bucket, key, policy=retry, max_get_bytes=max_get_bytes,
                                                     buffer_size=buffer_size),
                              outer)
            have = pf.schema_arrow.names
            use = [c for c in want if c in have]
            if not use:
                continue
            filt = pid_arr is not None and "person_id" in use
            rest = [c for c in use if c != "person_id"]
            for rg in range(pf.num_row_groups):
                if filt:
                    if not ids or not _rowgroup_may_match(pf, rg, "person_id", ids):
                        continue
                    pid = with_retries(lambda rg=rg: pf.read_row_group(rg, columns=["person_id"]), outer)
                    mask = pc.fill_null(pc.is_in(pc.cast(pid.column("person_id"), pa.int64(), safe=False),
                                                 value_set=pid_arr), False)
                    if not pc.any(mask).as_py():
                        continue
                    if rest:
                        other = with_retries(lambda rg=rg: pf.read_row_group(rg, columns=rest), outer)
                        tbl = pa.table({c: (pid.column(c) if c == "person_id" else other.column(c)) for c in use})
                    else:
                        tbl = pid
                    tbl = tbl.filter(mask)
                else:
                    tbl = with_retries(lambda rg=rg: pf.read_row_group(rg, columns=use), outer)
                for batch in tbl.to_batches(max_chunksize=batch_rows):
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


def csv_header(s3, key: str, *, bucket: str | None = None, max_bytes: int = 4 << 20,
               policy: RetryPolicy | None = None) -> list[str]:
    """Column names of a CSV object from its first line only (ranged read; values are never parsed)."""
    import csv as _csv
    bucket = bucket or access_point()
    n = 1 << 16
    while True:
        chunk, _ = get_bytes(s3, bucket, key, 0, n - 1, policy=policy)
        if b"\n" in chunk or len(chunk) < n or n >= max_bytes:
            break
        n *= 4
    first = chunk.decode("utf-8-sig", "replace").split("\n", 1)[0].rstrip("\r")
    return next(_csv.reader([first]), [])


def parquet_column_names(s3, key: str, *, bucket: str | None = None, policy: RetryPolicy | None = None) -> list[str]:
    """Column names of one parquet part from its footer only (ranged reads; no row data)."""
    import pyarrow.parquet as pq
    bucket = bucket or access_point()
    size = head_size(s3, bucket, key, policy=policy)
    pf = pq.ParquetFile(io.BufferedReader(S3RangeFile(s3, bucket, key, size, policy=policy), buffer_size=1 << 20))
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
