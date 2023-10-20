def prepare_certs(server_cert=None, client_key=None, client_ca=None):
    if server_cert is not None:
        with open(server_cert, "rb") as f:
            server_cert = f.read()
    if client_key is not None:
        with open(client_key, "rb") as key:
            client_key = key.read()
    if client_ca is not None:
        with open(client_ca, "rb") as ca:
            client_ca = ca.read()
    return server_cert, client_key, client_ca
