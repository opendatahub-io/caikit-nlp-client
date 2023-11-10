from caikit_nlp_client import GrpcClient, GrpcConfig, HttpClient, HttpConfig

text = "What is the boilign point of Nitrogen?"

# Using the http client
client = HttpClient(HttpConfig(host="localhost", port=8080, tls=False))

generated_text = client.generate_text("flan-t5-small-caikit", text)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 15, "finish_reason": "EOS_TOKEN", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}

client.generate_text("flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20)
print(generated_text)
# {"generated_text": "...", "generated_tokens": 20, "finish_reason": "MAX_TOKENS", "producer_id": {"name": "Text Generation", "version": "0.1.0"}, "input_token_count": 11, "seed": null}


# using the grpc client
client = GrpcClient(GrpcConfig(host="localhost", port=8085, insecure=True))
text = client.generate_text(
    "flan-t5-small-caikit", text, min_new_tokens=20, max_new_tokens=20
)
print(text)
# 'this is the generated text'
