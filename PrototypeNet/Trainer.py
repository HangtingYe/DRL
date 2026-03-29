import torch
import torch.optim as optim
import torch.nn.functional as F
from DataSet.DataLoader import get_dataloader
from PrototypeNet.Model import PN
from utils import aucPerformance, get_logger, F1Performance
import ipdb
import numpy as np

class Trainer(object):
    def __init__(self, model_config: dict):
        self.sche_gamma = model_config['sche_gamma']
        self.device = model_config['device']
        self.learning_rate = model_config['learning_rate']
        self.model = PN(model_config).to(self.device)
        self.train_loader, self.test_loader = get_dataloader(model_config)
        self.model_config = model_config

    def training(self, epochs):
        train_logger = get_logger(f"./models/{self.model_config['dataset_name']}_PN_{self.model_config['preprocess']}_{self.model_config['diversity']}_{self.model_config['plearn']}_{self.model_config['input_info']}_{self.model_config['input_info_ratio']}_{self.model_config['cl']}_{self.model_config['cl_ratio']}_{self.model_config['prototype_num']}_{self.model_config['random_seed']}.log")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.sche_gamma)
        self.model.train()
        print("Training Start.")
        for epoch in range(epochs):
            running_loss = 0.0
            for step, (x_input, y_label) in enumerate(self.train_loader):
                x_input = x_input.to(self.device)
                # decomposition loss
                loss = self.model(x_input).mean()

                # alignment loss
                if self.model_config['input_info'] == True:
                    h = self.model.encoder(x_input)
                    x_tilde = self.model.decoder(h)
                    s_loss = (1-F.cosine_similarity(x_tilde, x_input, dim=-1)).mean() 
                    # s_loss = F.cosine_similarity(x_tilde, x_input, dim=-1).mean() * (-1)
                    loss += self.model_config['input_info_ratio'] * s_loss

                # separation loss
                if self.model_config['cl'] == True:
                    h_ = self.model.phi(x_input)
                    # !!!!!!!!!
                    # h_ = F.softmax(self.model.phi(x_input), dim=1)
                    selected_rows = np.random.choice(h_.shape[0], int(h_.shape[0] * 0.8), replace=False)
                    h_ = h_[selected_rows]

                    matrix = h_ @ h_.T
                    mol = torch.sqrt(torch.sum(h_**2, dim=-1, keepdim=True)) @ torch.sqrt(torch.sum(h_.T**2, dim=0, keepdim=True))
                    matrix = matrix / mol
                    eye = torch.eye(h_.shape[0], device=h_.device)
                    d_loss = ((1 - eye) * matrix).sum() / (h_.shape[0]) / (h_.shape[0])
                    loss += self.model_config['cl_ratio'] * d_loss
                
                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()
            info = 'Epoch:[{}]\t loss={:.4f}\t'
            running_loss = running_loss / len(self.train_loader)
            train_logger.info(info.format(epoch,running_loss))
            torch.save(self.model, f"./models/{self.model_config['dataset_name']}_PN_{self.model_config['preprocess']}_{self.model_config['diversity']}_{self.model_config['plearn']}_{self.model_config['input_info']}_{self.model_config['input_info_ratio']}_{self.model_config['cl']}_{self.model_config['cl_ratio']}_{self.model_config['prototype_num']}_{self.model_config['random_seed']}_{epoch}.pth")
            if self.model_config.get('eval_each_epoch'):
                mse_rauc, mse_ap, mse_f1 = self.evaluate_current()
                print(f"Eval epoch {epoch}: AUC-ROC={mse_rauc:.4f}, AUC-PR={mse_ap:.4f}, f1={mse_f1:.4f}")
        torch.save(self.model, f"./models/{self.model_config['dataset_name']}_PN_{self.model_config['preprocess']}_{self.model_config['diversity']}_{self.model_config['plearn']}_{self.model_config['input_info']}_{self.model_config['input_info_ratio']}_{self.model_config['cl']}_{self.model_config['cl_ratio']}_{self.model_config['prototype_num']}_{self.model_config['random_seed']}.pth")
        print("Training complete.")
        train_logger.handlers.clear()

    def _load_model(self):
        path = f"./models/{self.model_config['dataset_name']}_PN_{self.model_config['preprocess']}_{self.model_config['diversity']}_{self.model_config['plearn']}_{self.model_config['input_info']}_{self.model_config['input_info_ratio']}_{self.model_config['cl']}_{self.model_config['cl_ratio']}_{self.model_config['prototype_num']}_{self.model_config['random_seed']}.pth"
        try:
            model = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            model = torch.load(path, map_location=self.device)
        return model.to(self.device)

    def evaluate(self):
        model = self._load_model()
        was_training = self.model.training
        self.model.eval()
        model.eval()
        mse_score, test_label = [], []
        with torch.no_grad():
            for step, (x_input, y_label) in enumerate(self.test_loader):
                # calculate l2_norm_square in latent space for anomaly score
                x_input = x_input.to(self.device)

                h = model.encoder(x_input)

                weight = self.model.phi(x_input)
                # !!!!!!!!!
                # weight = F.softmax(self.model.phi(x_input), dim=1)
                h_ = weight@model.prototype

                mse = F.mse_loss(h, h_, reduction='none')
                mse_batch = mse.mean(dim=-1, keepdim=True)
                mse_batch = mse_batch.data.cpu()
                mse_score.append(mse_batch)
                test_label.append(y_label)
        if was_training:
            self.model.train()
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1

    def evaluate_current(self):
        was_training = self.model.training
        self.model.eval()
        mse_score, test_label = [], []
        with torch.no_grad():
            for step, (x_input, y_label) in enumerate(self.test_loader):
                x_input = x_input.to(self.device)

                h = self.model.encoder(x_input)

                weight = self.model.phi(x_input)
                # !!!!!!!!!
                # weight = F.softmax(self.model.phi(x_input), dim=1)
                h_ = weight@self.model.prototype

                mse = F.mse_loss(h, h_, reduction='none')
                mse_batch = mse.mean(dim=-1, keepdim=True)
                mse_batch = mse_batch.data.cpu()
                mse_score.append(mse_batch)
                test_label.append(y_label)
        if was_training:
            self.model.train()
        mse_score = torch.cat(mse_score, axis=0).numpy()
        test_label = torch.cat(test_label, axis=0).numpy()
        mse_rauc, mse_ap = aucPerformance(mse_score, test_label)
        mse_f1 = F1Performance(mse_score, test_label)
        return mse_rauc, mse_ap, mse_f1

    def test_score(self):
        model = self._load_model()
        was_training = self.model.training
        self.model.eval()
        model.eval()
        mse_score, test_label = [], []
        with torch.no_grad():
            for step, (x_input, y_label) in enumerate(self.test_loader):
                # calculate l2_norm_square in latent space for anomaly score
                x_input = x_input.to(self.device)

                h = model.encoder(x_input)

                weight = self.model.phi(x_input)
                # !!!!!!!!!
                # weight = F.softmax(self.model.phi(x_input), dim=1)
                h_ = weight@model.prototype

                mse = F.mse_loss(h, h_, reduction='none')
                mse_batch = mse.mean(dim=-1, keepdim=True)
                mse_batch = mse_batch.data.cpu()
                mse_score.append(mse_batch)
                test_label.append(y_label)
        if was_training:
            self.model.train()
        mse_score = torch.cat(mse_score, axis=0).numpy()
        return mse_score

    def train_score(self):
        model = self._load_model()
        was_training = self.model.training
        self.model.eval()
        model.eval()
        mse_score, test_label = [], []
        with torch.no_grad():
            for step, (x_input, y_label) in enumerate(self.train_loader):
                # calculate l2_norm_square in latent space for anomaly score
                x_input = x_input.to(self.device)

                h = model.encoder(x_input)

                weight = self.model.phi(x_input)
                # !!!!!!!!!
                # weight = F.softmax(self.model.phi(x_input), dim=1)
                h_ = weight@model.prototype

                mse = F.mse_loss(h, h_, reduction='none')
                mse_batch = mse.mean(dim=-1, keepdim=True)
                mse_batch = mse_batch.data.cpu()
                mse_score.append(mse_batch)
                test_label.append(y_label)
        if was_training:
            self.model.train()
        mse_score = torch.cat(mse_score, axis=0).numpy()
        return mse_score
