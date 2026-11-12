---
license: apache-2.0
base_model: Qwen/Qwen3-Reranker-4B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
---

# vserifsaglam/Qwen3-Reranker-4B-4bit-MLX

This model [vserifsaglam/Qwen3-Reranker-4B-4bit-MLX](https://huggingface.co/vserifsaglam/Qwen3-Reranker-4B-4bit-MLX) was
converted to MLX format from [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
using mlx-lm version **0.26.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("vserifsaglam/Qwen3-Reranker-4B-4bit-MLX")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
