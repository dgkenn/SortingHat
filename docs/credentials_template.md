# Credentials template (placeholders only)

Everything below describes **where** the earlier research code looked for credentials and **what the entries
are called**. Every value is a placeholder. Real values live only on your own machine, outside this repo, with
mode 600 on files. Never paste real keys, passwords or private keys into this repo, a chat, or an agent
session (CLAUDE.md rules 1, 2 and 6). Which dataset needs which entry: `docs/data_access.md`.

## 1. AWS profiles for the BDSP S3 access points

BDSP access is **one AWS key pair for the AWS account you registered with BDSP**; the same pair opens all
three access points (credentialed, restricted, projects) that your DUA covers. The source project wrote the
same pair under two profile names.

`~/.aws/credentials` (mode 600):

```ini
[physionet]                       # profile NAME the source scripts passed to boto3.Session(profile_name=...)
aws_access_key_id     = <YOUR_BDSP_REGISTERED_ACCESS_KEY_ID>
aws_secret_access_key = <YOUR_BDSP_REGISTERED_SECRET_ACCESS_KEY>

[default]                         # the source also wrote identical keys here (scripts that use the default chain)
aws_access_key_id     = <YOUR_BDSP_REGISTERED_ACCESS_KEY_ID>
aws_secret_access_key = <YOUR_BDSP_REGISTERED_SECRET_ACCESS_KEY>
```

`~/.aws/config`:

```ini
[default]
region = us-east-1
s3 =
  payload_signing_enabled = false

[profile physionet]
region = us-east-1
s3 =
  payload_signing_enabled = false
```

Profile selection in `sortinghat/data_io.py`: `make_client(profile=...)`, else env `HEEDB_AWS_PROFILE`, else
`AWS_PROFILE`, else the default boto3 chain. The profile name is arbitrary; `physionet` is just what the
source used. Run through `scripts/heedb_run.sh` so a sandbox's placeholder `AWS_ACCESS_KEY_ID` cannot shadow
the profile (`docs/heedb_access.md` section 3).

Important caveat from the source (`bsde/docs/DEPOSIT_ACCESS_STATUS.md`): the key pair on disk resolved to the
AWS account **root**, while PhysioNet's own S3 grant names an IAM **user** (`physionet-user`). That mismatch
only matters for PhysioNet's S3 mirror, not for BDSP.

## 2. Access-point overrides (optional environment variables, names only)

| Variable | Meaning |
|---|---|
| `HEEDB_AWS_PROFILE` | profile name for BDSP calls (see above) |
| `BDSP_ACCESS_POINT_CREDENTIALED` (alias `HEEDB_ACCESS_POINT`) | your own access-point alias instead of the shared ARN, e.g. `<your-alias>-s3alias` |
| `BDSP_ACCESS_POINT_RESTRICTED`, `BDSP_ACCESS_POINT_PROJECTS` | same, for the I-CARE/restricted and MORGOTH/projects access points |

## 3. PhysioNet (HTTPS) logins

For credentialed PhysioNet projects (separate from BDSP; BDSP keys do nothing here). Either environment
variables:

```bash
export PHYSIONET_USER=<your physionet.org username>
export PHYSIONET_PASSWORD=<your physionet.org password>
```

or `~/.netrc` (mode 600), which `sortinghat.data_io.physionet_session()` reads via Python's `netrc` module:

```
machine physionet.org login <YOUR_PHYSIONET_USERNAME> password <YOUR_PHYSIONET_PASSWORD>
```

The code uses these to **log in and obtain a session cookie**; passing them as HTTP Basic auth to `wget`/`curl`
returns 403 on `/files/` paths. Credentialed access is granted per project: your account must have signed that
project's DUA, and CITI training must be current.

## 4. TUH / NEDC (rsync over SSH)

An ed25519 key registered with NEDC, at `~/.ssh/id_ed25519` (mode 600). Connection facts: user
`nedc-tuh-eeg`, host `www.isip.piconepress.com`, port 22 (must be open on your network), remote root
`data/tuh_eeg`. The source's cloud setup carried the key as base64 in an environment variable
(`NEDC_SSH_KEY_B64`, optional `NEDC_SSH_USER`); on your own machine just keep the key file.

## 5. No credentials needed

OpenNeuro (anonymous S3 over HTTPS), PhysioNet open projects (Sleep-EDF, EEGMMIDB, I-CARE v2.1 per the
source's registry), VitalDB public API, Zenodo/figshare open records, Hugging Face `weighting666/CBraMod`
(the source downloaded it without a token). Hugging Face and GitHub tokens did not appear in the source's
data paths; if you hit a gated model, store the token in the standard `huggingface-cli login` location, not here.

## 6. Hygiene checklist

- `chmod 600 ~/.aws/credentials ~/.netrc ~/.ssh/id_ed25519`
- `.gitignore` here already excludes `data/heedb/`, `data/restricted/`, `**/local_only/`, `*.edf`, `*.parquet`.
- Rotate any secret that ever appeared in a chat transcript (the source repo's own note: "a transcript is not
  a secret store").
- Test access with a count only, never rows:
  `scripts/heedb_run.sh python -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Arn'])"`
