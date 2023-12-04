from caikit_nlp_client import GrpcClient, HttpClient

host = "localhost"
http_port = 8080
grpc_port = 8085
text = "What is the boiling point of Nitrogen?"

# Using the http client
http_client = HttpClient(f"http://{host}:{http_port}")

generated_text = http_client.generate_text("flan-t5-small-caikit", text)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 15, "finish_reason": "EOS_TOKEN", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}

http_client.generate_text(
    "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 20, "finish_reason": "MAX_TOKENS", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}


# using the grpc client using insecure (no encryption)
grpc_client = GrpcClient(host=host, port=grpc_port, insecure=True)
text = grpc_client.generate_text(
    "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
)
print(text)
# 'this is the generated text'


# using a context manager (grpc only):
# (makes sure that the underlying resources are cleaned)
with GrpcClient(host, port=grpc_port) as client:
    text = client.generate_text(
        "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
    )


# using the streaming implementation, grpc
chunks = []
for chunk in grpc_client.generate_text_stream("flan-t5-small-caikit", text):
    print(f"Got {chunk=}")
    chunks.append(chunk)
print(f"generated text: {''.join(chunks)}")

# using the streaming implementation, http
chunks = []
for chunk in http_client.generate_text_stream("flan-t5-small-caikit", text):
    print(f"Got {chunk=}")
    chunks.append(chunk)
print(f"generated text: {''.join(chunks)}")


# Using a self-signed CA Certificate with the http client
http_client = HttpClient(f"https://{host}:{http_port}", ca_cert_path="ca.pem")

with open("ca.pem", "rb") as fh:
    ca_cert = fh.read()

# Using a self-signed CA Certificate with the grpc client
grpc_client = GrpcClient(host, grpc_port, ca_cert=ca_cert)

# Using the http client skipping remote host certificate(s) verification
http_client = HttpClient(f"https://{host}", verify=False)
text = grpc_client.generate_text("flan-t5-small-caikit", text)
print(text)
# 'this is the generated text'

# Using the grpc client skipping remote host certificate(s) verification
grpc_client = GrpcClient(host=host, port=grpc_port, verify=False)
text = grpc_client.generate_text("flan-t5-small-caikit", text)
print(text)
# 'this is the generated text'
