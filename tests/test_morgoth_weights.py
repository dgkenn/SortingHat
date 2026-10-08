"""Weights/code pinning with a fake S3 (no network, no real checkpoints)."""
import hashlib
import json

import pytest

from sortinghat import data_io
from sortinghat.morgoth import heads, weights


class FakeS3:
    def __init__(self, objects):
        self.objects, self.calls = objects, []

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        a, b = (int(x) for x in Range[6:].split("-"))
        if a >= len(data):
            err = Exception("InvalidRange")
            err.response = {"Error": {"Code": "InvalidRange"}, "ResponseMetadata": {"HTTPStatusCode": 416}}
            raise err
        b = min(b, len(data) - 1)
        self.calls.append((Key, a, b))
        body = data[a:b + 1]

        class B:
            def read(self_):
                return body
        return {"Body": B(), "ContentRange": f"bytes {a}-{b}/{len(data)}", "ContentLength": len(body)}


@pytest.fixture
def blobs():
    return {weights.WEIGHTS_PREFIX + "NORMAL.pth": bytes(range(256)) * 40000,       # ~10 MB: forces several ranges
            weights.WEIGHTS_PREFIX + "BS.pth": b"tiny"}


def test_fetch_records_size_and_sha_and_skips_when_present(tmp_path, blobs, monkeypatch):
    monkeypatch.setattr(data_io, "ap_arn", lambda name="credentialed": "ap")
    s3 = FakeS3(blobs)
    got = weights.fetch_weights(["normal", "bs"], s3=s3, cache=tmp_path)
    blob = blobs[weights.WEIGHTS_PREFIX + "NORMAL.pth"]
    assert got["NORMAL.pth"] == {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                                 "key": weights.WEIGHTS_PREFIX + "NORMAL.pth"}
    assert (tmp_path / "weights" / "BS.pth").read_bytes() == b"tiny"
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert set(m["weights"]) == {"NORMAL.pth", "BS.pth"}
    n = len(s3.calls)
    weights.fetch_weights(["normal", "bs"], s3=s3, cache=tmp_path)
    assert len(s3.calls) == n                                       # present with the recorded size: not refetched
    assert weights.verify(tmp_path) == {"NORMAL.pth": "ok", "BS.pth": "ok"}
    (tmp_path / "weights" / "BS.pth").write_bytes(b"evil")
    assert weights.verify(tmp_path)["BS.pth"] == "mismatch"
    (tmp_path / "weights" / "BS.pth").unlink()
    assert weights.verify(tmp_path)["BS.pth"] == "missing"


def test_available_needs_code_and_every_checkpoint(tmp_path):
    assert not weights.available(("normal",), tmp_path)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "NORMAL.pth").write_bytes(b"x")
    assert not weights.available(("normal",), tmp_path)
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "backbone.py").write_text("")
    assert weights.available(("normal",), tmp_path)
    assert not weights.available(("normal", "iiic"), tmp_path)


def test_cache_dir_env_override_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("SORTINGHAT_MORGOTH_CACHE", str(tmp_path))
    assert weights.cache_dir() == tmp_path
    monkeypatch.delenv("SORTINGHAT_MORGOTH_CACHE")
    assert str(weights.cache_dir()).endswith(".cache/sortinghat/morgoth")


def test_checkpoint_files_cover_default_heads_and_eeg_level():
    f = weights.checkpoint_files()
    assert "NORMAL.pth" in f and "IIIC.pth" in f and "SLEEP.pth" not in f
    assert len(weights.checkpoint_files(eeg_level=True)) == len(f) + 12
    assert all(heads.HEADS[h].checkpoint in weights.checkpoint_files([h]) for h in heads.HEADS)


def test_pinned_commit_looks_like_a_sha():
    assert len(weights.CODE_COMMIT) == 40 and set(weights.CODE_COMMIT) <= set("0123456789abcdef")
