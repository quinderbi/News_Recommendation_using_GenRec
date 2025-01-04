import pandas as pd

ds_name = ["ebnerd_demo", "ebnerd_small"]

print("Watch dataset")
print("==========Raw Data===========")
for ds in ds_name:
    df = pd.read_parquet(f"./raw_data/{ds}/train/history.parquet")

    df = df.explode([
        "impression_time_fixed",
        "scroll_percentage_fixed",
        "article_id_fixed",
        "read_time_fixed"
    ])

    user = df['user_id'].nunique()
    item = df['article_id_fixed'].nunique()
    interaction = len(df)
    sparcity = 1 - (interaction / (user * item))

    print("Dataset: {}".format(ds))
    print("The number of users : {}".format(user))
    print("The number of items : {}".format(item))
    print("The number of interactions : {}".format(interaction))
    print("Sparcity : {}".format(sparcity))
    print("")
print("=============================")
print("")
print("========Processed Data=======")
for ds in ds_name:
    train_df = pd.read_csv(r'./data/{}/train_df_1.csv'.format(ds))
    test_df = pd.read_csv(r'./data/{}/test_df_1.csv'.format(ds))
    
    train_df = train_df.rename(columns={"user_id":"user","item_id":"item"})
    test_df = test_df.rename(columns={"user_id":"user","item_id":"item"})

    ratings = pd.concat([train_df, test_df])

    user = ratings['user'].nunique()
    item = ratings['item'].nunique()
    interaction = len(ratings)
    sparcity = 1 - (interaction / (user * item))

    print("Dataset: {}".format(ds))
    print("The number of users : {}".format(user))
    print("The number of items : {}".format(item))
    print("The number of interactions : {}".format(interaction))
    print("Sparcity : {}".format(sparcity))
    print("")
print("=============================")

