"""
    Called from Makefile to train a model
    using the data from datasets/ directory
"""

from llama_formatter import llama_formatter

from sklearn.model_selection import train_test_split
import pandas as pd
import csv  # Import csv for quoting options

def split_dataset(formatted_filepath, train_filepath, test_filepath, test_size=0.2):
    # Read the formatted dataset
    df = pd.read_csv(formatted_filepath)

    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=test_size)

    # Save the train and test datasets
    train_df.to_csv(train_filepath, index=False, quoting=csv.QUOTE_ALL)
    test_df.to_csv(test_filepath, index=False, quoting=csv.QUOTE_ALL)


def main():
    formatter = llama_formatter()
    raw_filepath = "datasets/olafrv-trl/raw.csv"
    fmt_filepath = "datasets/olafrv-trl/fmt.csv"
    train_filepath = "datasets/olafrv-trl/train/train.csv"
    test_filepath = "datasets/olafrv-trl/test/test.csv"

    # Format the raw dataset
    formatter.format_dataset_csv(
        input_filepath=raw_filepath,
        output_filepath=fmt_filepath
    )

    # Split the formatted dataset into train and test sets
    split_dataset(
        formatted_filepath=fmt_filepath,
        train_filepath=train_filepath,
        test_filepath=test_filepath
    )


if __name__ == "__main__":
    main()
