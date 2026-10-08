import pytest

from sortinghat import checkpoint, omop_cache
from sortinghat.synthetic import generate, write_tables


@pytest.fixture(scope="session", autouse=True)
def _checkpoint_root(tmp_path_factory):
    """Long steps checkpoint under out/local_only/checkpoints and share the OMOP cache under out/local_only/omop_cache by default;
    tests (and the subprocesses they start) must not write synthetic entries next to real ones."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(checkpoint.ENV_ROOT, str(tmp_path_factory.mktemp("checkpoints")))
        mp.setenv(omop_cache.ENV_ROOT, str(tmp_path_factory.mktemp("omop_cache")))
        yield


@pytest.fixture(scope="session")
def synth():
    tables, truth = generate(3000, seed=20260101)
    return tables, truth


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, synth):
    """Synthetic data written in the real HEEDB layout (EEG/..., OMOP/Merged/<table>/*.parquet)."""
    d = tmp_path_factory.mktemp("synthetic")
    write_tables(synth[0], d, synth[1])
    return d
