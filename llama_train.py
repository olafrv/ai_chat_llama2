from llama_formatter import llama_formatter


def main():
    formatter = llama_formatter()
    formatter.format_dataset_csv(
        input_filepath="datasets/olafrv/raw/olaf-raw.csv",
        output_filepath="datasets/olafrv/fmt/olaf-fmt.csv"
    )

# Fix me!
# To train the model in the hard way?
# Instead of `make train` and HF library?

if __name__ == "__main__":
    main()
