"""heedb_io unit tests: fake S3 client, synthetic bytes only. No network, no credentials."""

import io

import pandas as pd
import pytest

from sortinghat import data_io as heedb_io
from sortinghat.agent_safety import RestrictedDataError


class FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, Bucket, Key, Range=None):
        data = self.objects[Key]
        if Range:
            a, b = Range.removeprefix("bytes=").split("-")
            data = data[int(a):int(b) + 1]
        return {"Body": io.BytesIO(data)}


def test_make_client_refuses_in_agent_session(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(RestrictedDataError):
        heedb_io.make_client()


def test_resolve_and_read_synthetic_csvs():
    ap = heedb_io.DEFAULT_ACCESS_POINT
    objs = {
        "EEG/eeg-metadata/S0001_eeg_metadata_2026_01_01.csv": b"BidsFolder,SessionID\nsub-S0001999,1\n",
        "EEG/eeg-metadata/S0001_eeg_metadata_2026_04_30.csv": b"\xef\xbb\xbfBidsFolder,SessionID,EEGFolder\nsub-S0001999,1,cEEG\n",
        "EEG/HEEDB_Metadata/S0001_EEG_reports_findings.csv": b"BDSPPatientID,bs\n999,\n",
    }
    s3 = FakeS3(objs)
    assert heedb_io.resolve_key(s3, "eeg_metadata", "S0001").endswith("2026_04_30.csv")
    meta = heedb_io.read_csv_table("eeg_metadata", "S0001", s3=s3)
    assert list(meta.columns) == ["BidsFolder", "SessionID", "EEGFolder"]     # BOM stripped
    # the double-underscore name is absent, so the single-underscore fallback is used
    assert heedb_io.resolve_key(s3, "reports_findings", "S0001").endswith("_EEG_reports_findings.csv")
    with pytest.raises(ValueError):
        heedb_io.resolve_key(s3, "reports_findings")                          # per-site table needs a site
    assert ap  # default access point is defined


def test_key_builders_and_finding_convention():
    assert heedb_io.bids_folder_for("S0001", 123) == "sub-S0001123"
    k = heedb_io.bids_edf_key("S0001", "sub-S0001123", "7", "cEEG_xyz")
    assert k == "EEG/bids/S0001/sub-S0001123/ses-7/eeg/sub-S0001123_ses-7_task-cEEG_eeg.edf"
    assert heedb_io.bids_edf_key("S0001", "sub-S0001123", "7", "").endswith("task-EEG_eeg.edf")
    s = pd.Series(["", "None", "nan", None, "x", "1"])
    assert heedb_io.finding_present(s).tolist() == [False, False, False, False, True, True]


def test_drop_placeholder_env_only_drops_non_aws_keys(tmp_path):
    creds = tmp_path / "credentials"
    creds.write_text("")
    env = {"AWS_ACCESS_KEY_ID": "stub", "AWS_SECRET_ACCESS_KEY": "x", "AWS_SHARED_CREDENTIALS_FILE": str(creds)}
    assert heedb_io.drop_placeholder_env(env) is True and "AWS_ACCESS_KEY_ID" not in env
    real_shape = {"AWS_ACCESS_KEY_ID": "AKIA" + "A" * 16, "AWS_SHARED_CREDENTIALS_FILE": str(creds)}
    assert heedb_io.drop_placeholder_env(real_shape) is False and "AWS_ACCESS_KEY_ID" in real_shape
    no_file = {"AWS_ACCESS_KEY_ID": "stub", "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "missing")}
    assert heedb_io.drop_placeholder_env(no_file) is False


def test_access_point_registry_and_env_override(monkeypatch):
    for name in ("credentialed", "restricted", "projects"):
        assert heedb_io.ap_arn(name).startswith("arn:aws:s3:us-east-1:")
    monkeypatch.setenv("BDSP_ACCESS_POINT_PROJECTS", "my-alias-s3alias")
    assert heedb_io.ap_arn("projects") == "my-alias-s3alias"
    assert all(spec.access_point in heedb_io.ACCESS_POINTS for spec in heedb_io.DATASETS.values())


def test_bdsp_list_and_icare_parse():
    s3 = FakeS3({"ICARE_train/training/0284/0284.txt": b"Hospital: A\nAge: 60\n",
                 "ICARE_train/training/0284/0284_001_004_EEG.mat": b"x"})
    assert heedb_io.bdsp_list("icare", "0284/", s3=s3, suffix="_EEG.mat") == [
        "ICARE_train/training/0284/0284_001_004_EEG.mat"]
    txt = heedb_io.bdsp_get("icare", "0284/0284.txt", s3=s3).decode()
    assert heedb_io.parse_icare_txt(txt) == {"Hospital": "A", "Age": "60"}


def test_physionet_session_refuses_in_agent_session(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(RestrictedDataError):
        heedb_io.physionet_session()
    assert heedb_io.physionet_url("eegmmidb", "1.0.0", "/S001/x.edf") == \
        "https://physionet.org/files/eegmmidb/1.0.0/S001/x.edf"


def test_vitaldb_decode_handles_gzip_and_bom():
    import gzip
    raw = "\ufeffcaseid,subjectid\n1,9\n".encode("utf-8")
    for blob in (raw, gzip.compress(raw)):
        assert heedb_io.vitaldb_decode(blob).startswith("caseid,")


def test_openneuro_xml_and_tuh_argv():
    xml = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>true</IsTruncated>'
           '<NextContinuationToken>T</NextContinuationToken><Contents><Key>ds1/a.edf</Key></Contents></ListBucketResult>')
    assert heedb_io.parse_list_objects_v2_xml(xml) == (["ds1/a.edf"], "T", True)
    argv = heedb_io.tuh_rsync_argv("data/tuh_eeg/TEST", ".", "/k")
    assert argv[0] == "rsync" and argv[-2] == "nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/TEST"


def test_heedb_io_shim_reexports():
    from sortinghat import heedb_io as shim
    assert shim.make_client is heedb_io.make_client


def test_sha256_file(tmp_path):
    p = tmp_path / "w"
    p.write_bytes(b"abc")
    assert heedb_io.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
