import pytest

from sortinghat.labels.ontology import (ALL_LABELS, CORE_PRIMARY_LABELS, GoldState, LABELS, Role,
                                         SILVER_CIRCULARITY_LABELS, e7_is_primary, get_label,
                                         primary_endpoint_labels)


def test_label_set_and_roles():
    assert ALL_LABELS == ("E1", "E2", "E3", "E4a", "E4b", "E5", "E6", "E7")
    assert LABELS["E3"].role is Role.POSITIVE_CONTROL
    assert LABELS["E4b"].role is Role.COVARIATE_SECONDARY
    assert LABELS["E7"].role is Role.PRIMARY_CONDITIONAL
    assert LABELS["E1"].sublabels == ("focal", "diffuse")


def test_primary_endpoint_excludes_e3_e4b_and_conditions_e7():
    assert primary_endpoint_labels() == ("E1", "E2", "E4a", "E5", "E6") == CORE_PRIMARY_LABELS
    for args in [(None, None), (500, 3), (99, 5), (500, 1)]:
        s = primary_endpoint_labels(*args)
        assert "E3" not in s and "E4b" not in s
    assert "E7" in primary_endpoint_labels(100, 2)
    assert "E7" not in primary_endpoint_labels(99, 2)
    assert "E7" not in primary_endpoint_labels(100, 1)
    assert e7_is_primary(100, 2) and not e7_is_primary(100, 1)


def test_silver_circularity_labels():
    assert SILVER_CIRCULARITY_LABELS == ("E1", "E2", "E4a", "E5", "E6", "E7")


def test_gold_states_ordinal():
    assert [int(GoldState[n]) for n in ("ABSENT", "POSSIBLE", "PROBABLE", "DEFINITE")] == [0, 1, 2, 3]
    assert GoldState.parse("Probable").is_positive and GoldState.parse("definite").is_positive
    assert not GoldState.POSSIBLE.is_positive and not GoldState.ABSENT.is_positive
    assert not GoldState.UNASSESSABLE.is_ordinal
    with pytest.raises(KeyError):
        GoldState.parse("maybe")


def test_get_label_unknown():
    with pytest.raises(KeyError):
        get_label("E9")
