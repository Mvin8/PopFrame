import importlib


def test_version_sync():
    import popframe

    # popframe.__version__ should be in sync with pyproject version (0.1)
    assert popframe.__version__ == "0.1"


def test_imports():
    # Smoke import of main subpackages and classes
    assert importlib.import_module("popframe.method.aglomeration")
    assert importlib.import_module("popframe.method.anchor_settlement")
    assert importlib.import_module("popframe.method.landuse_assessment")
    assert importlib.import_module("popframe.preprocessing.adjacency_calculator")


def test_constants_present():
    from popframe.utils import const

    assert hasattr(const, "METERS_PER_DEGREE")
    assert hasattr(const, "TIME_TO_METERS_FACTOR")
    assert hasattr(const, "LANDUSE_TAGS")
