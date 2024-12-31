import argparse
import pandas as pd
from DataProcessor import DataProcessor
from Model import FairGANModel
from Model import DiffModel

CONFIG = {
    "FairGANModel": {
        # "dataset": "ebnerd_demo",
        # "data_path": "ebnerd_demo/",
        'epochs': 10,
        'batch': 64,
        'ranker_gen_layers': [1000],
        'ranker_gen_activation': 'tanh',
        'ranker_gen_dropout': 0.0,
        'ranker_dis_layers': [1000],
        'ranker_dis_activation': 'tanh',
        'ranker_dis_dropout': 0.0,
        'controller_gen_layers': [1000],
        'controller_gen_activation': 'relu',
        'controller_gen_dropout': 0.0,
        'controller_dis_layers': [1000],
        'controller_dis_activation': 'relu',
        'controller_dis_dropout': 0.0,
        'ranker_gen_step': 2,
        'ranker_dis_step': 1,
        'controller_gen_step': 3,
        'controller_dis_step': 1,
        'controlling_fairness_step': 3,
        'ranker_gen_reg': 0.0001,
        'ranker_dis_reg': 0.0,
        'controller_gen_reg': 0.0,
        'controller_dis_reg': 0.0,
        'controlling_fairness_reg': 0.0,
        'alpha': 0.001,
        'lambda': 0.01,
        'ranker_gen_lr': 1e-5,
        'ranker_gen_beta1': 0.9,
        'ranker_dis_lr': 1e-5,
        'ranker_dis_beta1': 0.9,
        'controller_gen_lr': 0.001,
        'controller_gen_beta1': 0.9,
        'controller_dis_lr': 0.001,
        'controller_dis_beta1': 0.9,
        'controlling_fairness_lr': 1e-5,
        'controlling_fairness_beta1': 0.9,
        'ranker_initializer': 'glorot_normal',
        'controller_initializer': 'glorot_normal',
        'debug': False
    },

    "DiffModel": {
        # "data_path": "ebnerd_demo/",
        "lr": 0.01,
        "weight_decay": 0.0,
        "batch_size": 64,
        "epochs": 50,
        "tst_w_val": False,
        "cuda": False,
        # "save_path": './saved_models/',
        "log_name":'log',
        "round": 1,

        # param for model
        "time_type":'cat',
        "dims": [1000],
        "norm": False,
        "emb_size": 100,

        # param for diffusion
        "mean_type": 'x0', # MeanType for diffusion: x0, eps
        "steps": 100,
        "noise_schedule":'linear-var',
        "noise_scale": 0.1,
        "noise_min": 0.001,
        "noise_max": 0.02,
        "sampling_noise": False,
        "sampling_steps": 0,
        "reweight": True,
    }
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, default="ebnerd_demo", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to save the processed data")
    parser.add_argument("--fold", type=int, default=1, help="Number of folds for cross-validation")
    parser.add_argument("--matrix_type", type=str, default="one", help="Type of matrix to construct")
    parser.add_argument("--model", type=str, default="FairGANModel", help="Model to use")

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
        print("Invalid matrix type")
        exit()

    print("The number of users: {}".format(train.shape[0]))
    print("The number of items: {}".format(train.shape[1]))

    config = CONFIG[args.model]

    if args.model == "FairGANModel":
        model = FairGANModel(train, config)
    elif args.model == "DiffModel":
        model = DiffModel(train, config)
    else:
        print("Invalid model")
        exit()


    model.fit()

    # hasil = model.predict()
    
    print(model.evaluate(test,[5, 10, 20]))





if __name__ == "__main__":
    main()