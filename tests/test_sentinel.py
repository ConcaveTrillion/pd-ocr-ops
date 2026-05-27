import pdomain_ops


def test_pdomain_ops_imports():
    # Version is read from importlib.metadata; just assert it's a non-empty string
    assert isinstance(pdomain_ops.__version__, str)
    assert pdomain_ops.__version__
