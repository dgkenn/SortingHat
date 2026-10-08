"""Names-only probe of the MORGOTH release on the BDSP projects access point (project-lead authorised).

    cd <repo> && env -u CLAUDECODE -u SORTINGHAT_AGENT_SESSION scripts/heedb_run.sh python3 -m sortinghat.morgoth.probe

Prints NAMES (and object sizes) of code / weights / config / split files only: ``Delimiter='/'`` listings at depth <= 3
under each MORGOTH root (depth 0..3 means TASK folders are listed one level; up to 5 under prefixes whose path names models / code / splits / configs). It never descends
below a ``<TASK>/`` folder unless the name says split/config/fold/list/meta/manifest, nor into ``pretrain/``, never into an id-like prefix, never
reads an object body, and prints no counts of data files (only "other files present"). Output is not patient data.
"""
from __future__ import annotations

import re
import sys

from .. import data_io

ROOTS = ("morgoth1/", "morgoth2/", "baby_morgoth/model/")      # NOT the derived-analysis prefixes (morgoth-slowing/, baby_morgoth/results|dataset)
LISTED_EXT = (".pth", ".pt", ".ckpt", ".bin", ".safetensors", ".onnx", ".py", ".ipynb", ".yaml", ".yml", ".json", ".txt",
              ".md", ".rst", ".cfg", ".toml", ".sh", ".xlsx", ".csv", ".tsv", ".pkl", ".zip", ".tar", ".gz", ".tgz",
              ".license", "license", "readme")
MODEL_HINT = re.compile(r"(model|code|weight|checkpoint|ckpt|split|config|src|release|202\d{3})", re.I)
NEVER_DESCEND = re.compile(r"(pretrain/|/bids/|/raw/|internal_dataset/[^/]+/[^/]+/(?!.*(split|config|fold|list|meta|manifest)))", re.I)
MAX_PREFIXES = 60
MAX_FILES = 40
_ID_NAME = re.compile(r"(^|[^A-Za-z])(sub|ses)-|[0-9a-f]{12,}|^\d+(\.\w+)?$")      # patient-like; dates / version stamps are fine


def _listable(name: str) -> bool:
    n = name.lower()
    return n.endswith(LISTED_EXT) or n.rsplit("/", 1)[-1] in ("license", "readme", "sha256sums.txt")


def list_one(s3, bucket: str, prefix: str) -> dict:
    token, cps, files, other = None, [], [], False
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/", "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        cps += [c["Prefix"] for c in r.get("CommonPrefixes", [])]
        for o in r.get("Contents", []):
            if o["Key"] == prefix:
                continue
            if _listable(o["Key"]) and not _ID_NAME.search(o["Key"][len(prefix):]):
                files.append((o["Key"], int(o["Size"])))
            else:
                other = True
        if not r.get("IsTruncated"):
            break
        token = r.get("NextContinuationToken")
    return {"prefix": prefix, "prefixes": cps, "files": files, "other_files_present": other}


def walk(s3, bucket: str, prefix: str, depth: int, max_depth: int, out: list) -> None:
    lvl = list_one(s3, bucket, prefix)
    if len(lvl["prefixes"]) > MAX_PREFIXES:
        lvl["prefixes"] = []
        lvl["overflow"] = True
    if len(lvl["files"]) > MAX_FILES:
        lvl["files"] = []
        lvl["overflow"] = True
    out.append(lvl)
    for c in lvl["prefixes"]:
        if _ID_NAME.search(c[len(prefix):]) or NEVER_DESCEND.search(c):
            continue
        limit = 5 if MODEL_HINT.search(c) else max_depth
        if depth + 1 <= limit:
            walk(s3, bucket, c, depth + 1, max_depth, out)


def main(argv=None) -> int:
    s3 = data_io.make_client(None)
    bucket = data_io.ap_arn("projects")
    top = list_one(s3, bucket, "")
    print("top-level prefixes matching morgoth/nesi:", sorted(p for p in top["prefixes"] if "morgoth" in p.lower() or "nesi" in p.lower()))
    out: list = []
    for root in ROOTS:
        walk(s3, bucket, root, 0, 3, out)
    for lvl in out:
        print(f"[{lvl['prefix']}]" + ("  (overflow: not listed)" if lvl.get("overflow") else ""))
        for p in lvl["prefixes"]:
            nm = p[len(lvl["prefix"]):]
            tag = "  (not descended)" if NEVER_DESCEND.search(p) or _ID_NAME.search(nm) else ""
            print("   dir ", nm if not _ID_NAME.search(nm) else "<id-like>", tag)
        for k, sz in lvl["files"]:
            print("   file", k[len(lvl["prefix"]):], sz)
        if lvl["other_files_present"]:
            print("   (other data files present, not listed)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:                       # class name only: never a message
        print(f"morgoth probe failed: {type(e).__name__}", file=sys.stderr)
        sys.exit(1)
