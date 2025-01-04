import numpy as np
import pandas as pd
import zipfile
import requests as request

class EbnerdDatasetsLoader:

    DATASETS = {
        "ebnerd_demo": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip",
        "ebnerd_small": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip",
        "ebnerd_large": "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip"
    }

    @staticmethod
    def download_dataset(dataset_name):
        """
        Download the specified dataset.
        """
        print("Downloading file...")
        response = request.get(EbnerdDatasetsLoader.DATASETS[dataset_name], stream=True)
        if response.status_code == 200:
            with open(dataset_name + ".zip", 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print("File successfully downloaded.")
        else:
            print("Failed to download the file.")
            exit()

        # Step 2: Extract ZIP file
        print("Extracting file...")
        with zipfile.ZipFile(dataset_name + ".zip", "r") as zip_ref:
            zip_ref.extractall(f"./raw_data/{dataset_name}")
            print(f"File extracted to folder: ./raw_data/{dataset_name}")

    @staticmethod
    def get_data(dataset_name, duplicate=None, user_threshold=20, item_threshold=10):
        """
        Open and process the specified dataset.
        
        Parameters:
        - dataset_name: str, name of the dataset
        - duplicate: str, method to handle duplicates ('last', 'first', 'avg')
        - user_threshold: int, minimum number of interactions per user
        - item_threshold: int, minimum number of interactions per item

        Returns:
        - DataFrame: Processed dataset
        """
        df = pd.read_parquet(f"./raw_data/{dataset_name}/train/history.parquet")

        df = df.explode([
            "impression_time_fixed",
            "scroll_percentage_fixed",
            "article_id_fixed",
            "read_time_fixed"
        ])
        df = df.rename(columns={
            "impression_time_fixed": "timestamp",
            "scroll_percentage_fixed": "scroll_percentage",
            "article_id_fixed": "item_id",
            "read_time_fixed": "read_time"
        })

        df = df[["user_id", "item_id", "timestamp", "scroll_percentage", "read_time"]]
        df.dropna(inplace=True)

        if duplicate == "last":
            df = df.drop_duplicates(subset=['item_id', 'user_id'], keep='last')
        elif duplicate == "first":
            df = df.drop_duplicates(subset=['item_id', 'user_id'], keep='first')
        elif duplicate == "avg":
            df = df.groupby(['item_id', 'user_id'], as_index=False).agg({
                'timestamp': 'last',
                'scroll_percentage': 'mean',
                'read_time': 'mean'
            })

        while True:
            df = df[df.groupby('user_id')['user_id'].transform('count') >= user_threshold]
            df = df[df.groupby('item_id')['item_id'].transform('count') >= item_threshold]
            if (df.groupby('user_id')['user_id'].count().min() >= user_threshold and
                df.groupby('item_id')['item_id'].count().min() >= item_threshold):
                break
            if (np.isnan(df.groupby('user_id')['user_id'].count().min()) or
                np.isnan(df.groupby('item_id')['item_id'].count().min())):
                break

        return df