---
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path:
    - "fmt/olaf-fmt.csv"
---

# Conversion

Use `dataset_format.py` to convert CSV inputs to Llama v2 prompts.
# References

https://huggingface.co/docs/datasets/
https://huggingface.co/blog/llama2