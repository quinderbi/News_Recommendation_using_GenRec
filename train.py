import argparse
import pandas as pd
from DataProcessor import DataProcessor
from Model import FairGANModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default="ebnerd_demo", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save the processed data")
    parser.add_argument("--fold", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--matrix_type", type=str, default="one", help="Type of matrix to construct")

    args = parser.parse_args()

    train_df = pd.read_csv(r'./data/{}/train_df_{}.csv'.format(args.ds_name, args.fold))
    test_df = pd.read_csv(r'./data/{}/test_df_{}.csv'.format(args.ds_name, args.fold))

    train_df = train_df.rename(columns={"user_id":"user","item_id":"item"})
    test_df = test_df.rename(columns={"user_id":"user","item_id":"item"})

    ratings = pd.concat([train_df, test_df])

    if args.matrix_type == "one":
        train = DataProcessor.construct_one_valued_matrix(ratings, train_df, item_based=False)
        test = DataProcessor.construct_one_valued_matrix(ratings, test_df, item_based=False)
    elif args.matrix_type == "real":
        train = DataProcessor.construct_real_matrix(ratings, train_df, item_based=False)
        test = DataProcessor.construct_real_matrix(ratings, test_df, item_based=False)
    elif args.matrix_type == "ratio":
        train = DataProcessor.construct_ratio_valued_matrix(ratings, train_df, item_based=False)
        test = DataProcessor.construct_ratio_valued_matrix(ratings, test_df, item_based=False)
    else:
        # set error message and exit
        print("Invalid matrix type")
        exit()

    print("Train matrix:")

    model = FairGANModel(train)

    model.fit()

    print(model.predict())







if __name__ == "__main__":
    main()