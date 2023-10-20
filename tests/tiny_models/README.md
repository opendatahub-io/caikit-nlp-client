# Tiny Models

These models can be used for spinning up a server for testing the client without consuming too many resources.

This is based on the [test fixtures in caikit-nlp](https://github.com/caikit/caikit-nlp/tree/main/tests/fixtures/tiny_models#tiny-models-for-testing), which uses the [`transformers` utils for creating tiny models](https://github.com/huggingface/transformers/blob/main/utils/create_dummy_models.py).
In order to create a caikit-compatible model to be used with `caikit_nlp`, the following lines of code can be used

```python
import caikit_nlp
model = caikit_nlp.text_generation.TextGeneration.bootstrap("T5ForConditionalGeneration")
model.save("T5ForConditionalGeneration-caikit")
```
