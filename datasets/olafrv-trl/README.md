---
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path:
    - train/*
  - split: test
    path: 
    - test/*
---

# Conversion

Use `dataset_format.py` to convert `raw.csv` to Llama v2 prompt formats.

# References

* https://huggingface.co/docs/datasets/
* https://huggingface.co/blog/llama2