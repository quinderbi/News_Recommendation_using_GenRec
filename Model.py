from Utils.FairGAN import FairGAN
from Utils.DatasetPipeline import DatasetPipeline

def evaluate(ranking, train_data, test_data, topN=[5, 10, 15, 20]):
    ranking[train_data.nonzero()] = -np.inf
    result = {
        "precision": [],
        "recall": [],
        "ndcg": [],
    }

    for N in topN:
        precision = []
        recall = []
        ndcg = []
        for i in range(len(ranking)):
            topN_items = np.argpartition(-ranking[i], N)[:N]
            topN_items = topN_items[np.argsort(-ranking[i][topN_items])]
            _ , actual_history = test_data[i].nonzero()
            topN_label = np.isin(topN_items, actual_history).astype(int)

            precision.append(len(topN_label[topN_label == 1])/N)
            recall.append(len(topN_label[topN_label == 1])/len(actual_history))
            
            dcg = np.sum( topN_label / np.log2(np.arange(2, len(topN_label) + 2)))

            max_relevant = min(len(actual_history), N)
            if max_relevant > 0:
                ideal_label = np.ones(max_relevant)
                idcg = np.sum(ideal_label / np.log2(np.arange(2, max_relevant + 2)))
                ndcg_value = dcg / idcg
            else:
                ndcg_value = 0.0

            ndcg.append(ndcg_value)

        result["precision"].append(np.mean(precision))
        result["recall"].append(np.mean(recall))
        result["ndcg"].append(np.mean(ndcg))

    return result

    

class FairGANModel:
    def __init__(self,data,config):
        self.data = data
        self.config = config
        config["n_items"] = self.data.shape[1]
        self.train_ds = DatasetPipeline(labels=self.data.toarray(), conditions=self.data.toarray()).shuffle(1)
        self.model = FairGAN([], **self.config)


    def fit(self):
        
        self.model.fit(self.train_ds.shuffle(self.data.shape[0]).batch(self.config['batch'], True), epochs=self.config['epochs'], callbacks=[])

    def predict(self):
        return self.model.predict(self.train_ds.batch(self.data.shape[0]))
    
    def evaluate(self,test_data,topN=[5, 10, 20]):
        return evaluate(self.predict(),self.data,test_data,topN);

import torch
import torch.optim as optim
from Utils.DiffUtils.gaussian_diffusion import GaussianDiffusion
from Utils.DiffUtils.DNN import DNN
from Utils.DiffUtils.gaussian_diffusion import ModelMeanType
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import random
import numpy as np
from tqdm import tqdm

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)

class DiffModel:
    def __init__(self,data,config):
        self.data = data

        self.config = config

        random_seed = 1
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        def worker_init_fn(worker_id):
            np.random.seed(random_seed + worker_id)

        self.train_loader = DataLoader(DataDiffusion(torch.FloatTensor(data.A)), batch_size=self.config["batch_size"], \
            pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)

        self.device = torch.device("cuda:0" if self.config["cuda"] else "cpu")

        if self.config["mean_type"] == 'x0':
            mean_type = ModelMeanType.START_X
        elif self.config["mean_type"] == 'eps':
            mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % self.config["mean_type"])

        self.diffusion = GaussianDiffusion(mean_type, self.config["noise_schedule"], \
            self.config["noise_scale"], self.config["noise_min"], self.config["noise_max"], self.config["steps"], self.device).to(self.device)
        
        out_dims = self.config["dims"] + [self.data.shape[1]]
        in_dims = out_dims[::-1]

        self.model = DNN(in_dims, out_dims, self.config["emb_size"], time_type="cat", norm=self.config["norm"]).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])

    def fit(self):

        batch_count = 0
        total_loss = 0.0

        for epoch in range(1, self.config["epochs"] + 1):
            self.model.train()
            # for batch_idx, batch in enumerate(self.train_loader):
            for batch_idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Epoch %d" % epoch, ncols=80):
                batch = batch.to(self.device)
                batch_count += 1
                self.optimizer.zero_grad()
                losses = self.diffusion.training_losses(self.model, batch, self.config["reweight"])
                loss = losses["loss"].mean()
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            # print(f'Runing Epoch {epoch}')
            # print('---'*18)

    def predict(self):
        """
        Generate predictions using the trained model.
        """
        self.model.eval()  # Set model to evaluation mode
        predictions = []   # Initialize an empty list to store predictions

        with torch.no_grad():  # Disable gradient computation
            for batch_idx, batch in enumerate(self.train_loader):
                # Move batch to the appropriate device (e.g., GPU/CPU)
                batch = batch.to(self.device)

                # Generate predictions for the batch
                prediction_batch = self.diffusion.p_sample(
                    self.model,
                    batch,
                    self.config["sampling_steps"],
                    self.config["sampling_noise"]
                )

                # Append the predictions to the list
                predictions.append(prediction_batch)

        # Combine all predictions into a single tensor
        predictions = torch.cat(predictions, dim=0)
        return predictions.cpu().numpy()
    
    def evaluate(self,test_data,topN=[5, 10, 20]):
        return evaluate(self.predict(),self.data,test_data,topN);
