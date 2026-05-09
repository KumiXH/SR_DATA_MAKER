def test_package_imports_version():
    import sr_data_maker

    assert isinstance(sr_data_maker.__version__, str)
    assert sr_data_maker.__version__
