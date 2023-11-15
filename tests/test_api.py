def test_module():
    import caikit_nlp_client

    assert set(caikit_nlp_client.__all__) == {
        "GrpcClient",
        "HttpClient",
    }
