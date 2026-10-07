import pytest

from sortinghat.synthetic import generate, write_tables


@pytest.fixture(scope="session")
def synth():
    tables, truth = generate(3000, seed=20260101)
    return tables, truth


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory, synth):
    d = tmp_path_factory.mktemp("synthetic")
    write_tables(synth[0], d, synth[1], fmt="csv")
    return d
