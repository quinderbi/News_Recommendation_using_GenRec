import argparse
import pandas as pd
from DataProcessor import DataProcessor
from Config import Config
from Model import FairGANModel
from Model import DiffModel
import datetime
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default="ebnerd_demo", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save the processed data")
    parser.add_argument("--fold", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--matrix_type", type=str, default="one", help="Type of matrix to construct")
    parser.add_argument("--model", type=str, default="FairGANModel", help="Model to use")
    parser.add_argument("--worker_name", type=str, default="default_worker", help="Worker name")

    args = parser.parse_args()

    ### LOAD DATA ###
    print("Preparing data...")

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
        print("Invalid matrix type")
        exit()

    print("Data Ready.")

    print("The number of users: {}".format(train.shape[0]))
    print("The number of items: {}".format(train.shape[1]))

    ### MODEL ###
    print("preparing model...")
    config = Config[args.model]

    if args.model == "FairGANModel":
        model = FairGANModel(train, config)
    elif args.model == "DiffModel":
        model = DiffModel(train, config)
    else:
        print("Invalid model")
        exit()
    print("Model Ready.")

    ### TRAINING ###
    print("Training model...")
    start_train = datetime.datetime.now()
    model.fit()
    print("Model training completed.")
    finish_train = datetime.datetime.now()

    # hasil = model.predict() # ranting prdiction
    
    ### EVALUATION ###
    print("Evaluating model...")
    result = model.evaluate(test,[5, 10, 20])
    print("Evaluation completed.")

    print("Result: ", result)
    print("Done.")

    print("Saving result...")
    # save csv file
    df_result = pd.DataFrame({
        "data":[args.ds_name],
        "fold":[args.fold],
        "model":[args.model],
        "start_train":[start_train],
        "finish_train":[finish_train],
        "p@5":[result["precision"][0]],
        "p@10":[result["precision"][1]],
        "p@20":[result["precision"][2]],
        "r@5":[result["recall"][0]],
        "r@10":[result["recall"][1]],
        "r@20":[result["recall"][2]],
        "g@5":[result["ndcg"][0]],
        "g@10":[result["ndcg"][1]],
        "g@20":[result["ndcg"][2]],
    })

    if not os.path.exists('./result'):
        os.makedirs('./result')
    # check if file exist
    if os.path.exists(f'./result/{args.worker_name}.csv'):
        df_result.to_csv(f'./result/{args.worker_name}.csv', mode='a', header=False, index=False)
    else:
        df_result.to_csv(f'./result/{args.worker_name}.csv', mode='w', header=True, index=False)

    


if __name__ == "__main__":
    main()