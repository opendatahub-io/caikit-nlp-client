from caikit_nlp_client import GrpcClient, GrpcConfig, HttpClient, HttpConfig

text = "What is the boilign point of Nitrogen?"

# Using the http client
http_client = HttpClient(HttpConfig(host="localhost", port=8080, tls=False))

generated_text = http_client.generate_text("flan-t5-small-caikit", text)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 15, "finish_reason": "EOS_TOKEN", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}

http_client.generate_text(
    "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 20, "finish_reason": "MAX_TOKENS", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}


# using the grpc client
grpc_client = GrpcClient(GrpcConfig(host="localhost", port=8085, insecure=True))
text = http_client.generate_text(
    "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
)
print(text)
# 'this is the generated text'


# using a context manager (grpc only):
# (makes sure that the underlying resources are cleaned)
with GrpcClient("localhost", port=8085) as client:
    text = client.generate_text(
        "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
    )


# streaming implementation, grpc
chunks = []
for message in grpc_client.generate_text_stream("flan-t5-small-caikit", text):
    chunk = message["generated_text"]
    if not message.finish_reason:
        print("Got chunk, not finished")
    chunks.append(chunk)
print(f"generated text: {''.join(chunks)}")
print(f"finish_reason: {message.finish_reason}")

# streaming implementation, http
chunks = []
for message in http_client.generate_text_stream("flan-t5-small-caikit", text):
    chunk = message["generated_text"]
    if message.details.finish_reason == "NOT_FINISHED":
        print("Got chunk, not finished")
    chunks.append(chunk)
print(f"generated text: {''.join(chunks)}")
print(f"finish_reason: {message['details']['finish_reason']}")
