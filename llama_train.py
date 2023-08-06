"""
    Called from Makefile to train a model
    using the data from datasets/olafrv/fmt/
    References:
    - https://huggingface.co/docs/trl/main/en/index
    - ./trl/examples/scripts/sft_trainer.py
"""

from llama_formatter import llama_formatter


def main():
    formatter = llama_formatter()
    formatter.format_dataset_csv(
        input_filepath="datasets/olafrv/raw/olaf-raw.csv",
        output_filepath="datasets/olafrv/fmt/olaf-fmt.csv"
    )


if __name__ == "__main__":
    main()
