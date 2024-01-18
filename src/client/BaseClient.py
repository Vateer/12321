import copy
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

class BaseClient():
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = "cuda:{}".format(self.args.gpu)

    def set_model(self, model):
        self.model = model
    
    def get_parameter(self):

        return copy.deepcopy(self.para)
        
    
    def set_parameter(self, para):
        self.para = copy.deepcopy(para)

    def train(self):
        self.model.load_state_dict(self.para, strict=True)
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=0)
        loss_func = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        loss_cnt = 0
        for e in range(self.args.local_ep):
            for batch_idx, (datas, labels) in enumerate(self.train_loader):
                datas, labels = datas.to(self.device), labels.to(self.device)
                # self.net.zero_grad() #?
                opt.zero_grad()
                log_probs = self.model(datas)
                # loss = loss_func(log_probs, labels)
                loss = loss_func(log_probs, labels.long())
                total_loss += loss.item()
                loss_cnt += 1
                loss.backward() 
                opt.step()
                # batch_loss.append(loss.item())
        self.para = self.model.state_dict()
        return total_loss/loss_cnt, len(self.train_loader)
    
    def test(self):
        self.model.load_state_dict(self.para, strict=True)
        self.model.eval()
        # test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return accuracy.item(), self.test_loader.dataset.__len__()

    def get_svd(self):
        all_data = self.train_loader.dataset
        data, label = self.train_loader.dataset.tensors
        unique_label = np.unique(label)
        u_temp = []
        for i in unique_label:
            ind = np.where(label == i)
            data_matrix = data[ind]
            data_matrix = data_matrix.reshape(data_matrix.shape[0], -1).T
            k = self.args.svg_k
            if self.args.distribution == "dir":
                pass
            if k > 0:
                u1_temp, sh1_temp, vh1_temp = np.linalg.svd(data_matrix, full_matrices=False)
                u1_temp=u1_temp/np.linalg.norm(u1_temp, ord=2, axis=0) 
                u_temp.append(u1_temp[:,0:k])
        return np.hstack(u_temp)  
    
    def get_pca(self):
        all_data = self.train_loader.dataset
        data, label = self.train_loader.dataset.tensors
        unique_label = np.unique(label)
        u_temp = []
        for i in unique_label:
            ind = np.where(label == i)
            data_matrix = data[ind]
            data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)
            mean_vec = np.mean(data_matrix, axis=0)
            centered_data = data_matrix - mean_vec
            cov_matrix = np.cov(centered_data, rowvar=False)
            k = self.args.pca_k
            
            if k > 0:
                eigvals, eigvecs = np.linalg.eigh(cov_matrix)
                idx = np.argsort(eigvals)[::-1]
                eigvecs = eigvecs[:, idx]
                u_temp.append(eigvecs[:, :k])
        return np.hstack(u_temp)