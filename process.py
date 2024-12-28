import lenskit.crossfold as xf
import argparse
import os
from DataLoader import EbnerdDatasetsLoader
from DataProcessor import PreprocessDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default="ebnerd_demo", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save the processed data")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--duplicate", type=str, default="avg", help="Method to handle duplicates")

    args = parser.parse_args()

    ds_name = args.ds_name
    data_path = args.data_path
    folds = args.folds
    duplicate = args.duplicate

    print("Start to process the dataset")

    if not os.path.exists(f"./raw_data/{ds_name}"):
        EbnerdDatasetsLoader.download_dataset(ds_name)

    df = EbnerdDatasetsLoader.get_data(ds_name, duplicate=duplicate)
    df = PreprocessDataset.generate_internal_ids(df)

    path = f"{data_path}/{ds_name}"
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print(f"Creation of the directory {path} failed")
    else:
        print(f"Successfully created the directory {path}")

    print("Splitting dataset into training set and test set...")
    for i, tp in enumerate(xf.partition_rows(df, folds)):
        print(f"Processing fold: {i + 1}")
        tp.train.to_csv(f"{data_path}/{ds_name}/train_df_{i + 1}.csv", index=False)
        tp.test.to_csv(f"{data_path}/{ds_name}/test_df_{i + 1}.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()