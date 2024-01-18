from .BaseClient import BaseClient

import numpy as np
import torch
from src.utils import *

class DualFLClient(BaseClient):
    def run(self):
        pass  
    
    def fine_tune_global_model(self):
        self.model.load_state_dict(self.para, strict=True)
        for name, param in self.model.named_parameters():
            if 'fc' not in name:  
                param.requires_grad = False
                
        classifier_params = [param for name, param in self.model.named_parameters() if 'fc' in name]
        base_optimizer = torch.optim.SGD
        opt = SAM(classifier_params, base_optimizer, lr=self.args.lr_now, momentum=self.args.momentum, weight_decay=0)

        self.model.train()
        loss_func = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        loss_cnt = 0
        for e in range(self.args.local_ep):
            for batch_idx, (datas, labels) in enumerate(self.train_loader):
                datas, labels = datas.to(self.device), labels.to(self.device)
                log_probs = self.model(datas)
                loss = loss_func(log_probs, labels.long())
                total_loss += loss.item()
                loss_cnt += 1
                loss.backward()
                opt.first_step(zero_grad=True)
                loss_func(self.model(datas), labels.long()).backward()
                opt.second_step(zero_grad=True)
        self.para = self.model.state_dict()
        return total_loss/loss_cnt, len(self.train_loader)
    

    def fine_tune_global_model_sgd(self):
        self.model.load_state_dict(self.para, strict=True)
        
        for name, param in self.model.named_parameters():
            if 'fc' not in name: 
                param.requires_grad = False
                
        opt = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_now, momentum=self.args.momentum, weight_decay=0)
        loss_func = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        loss_cnt = 0
        for e in range(self.args.local_ep):
            for batch_idx, (datas, labels) in enumerate(self.train_loader):
                datas, labels = datas.to(self.device), labels.to(self.device)
                opt.zero_grad()
                log_probs = self.model(datas)
                loss = loss_func(log_probs, labels.long())
                total_loss += loss.item()
                loss_cnt += 1
                loss.backward() 
                opt.step()
        self.para = self.model.state_dict()
        return total_loss/loss_cnt, len(self.train_loader)
    

    def inference_with_global_model(self, global_para, test_loader = None, fff=0.05, lamb = 0.5):
        self.model.load_state_dict(self.para, strict=True)
        self.model.eval()
        correct = 0
        not_sure_data = []
        not_sure_label = []
        not_sure_output = []

        wrong_cnt = 0
        total = 0

        indices = np.array([])
        if test_loader == None:
            test_loader = self.test_loader
        else:
            lamb = 1
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                entropy = shannon_entropy(output)
                pred = output.argmax(dim=1, keepdim=True)
                total_correct = pred.eq(target.view_as(pred)).sum().item()
                fff = (1 - total_correct / len(test_loader.dataset)) / 2 

                indices = np.argsort(entropy.cpu())
                indices = indices[-int(len(indices)*fff):]
                sure_indices = np.delete(np.arange(len(entropy)), indices) 


                correct += pred.eq(target.view_as(pred))[sure_indices].sum().item()

                not_sure_data.append(data[indices])
                not_sure_label.append(target[indices])
                not_sure_output.append(output[indices])
        
        not_sure_data = torch.cat(not_sure_data, dim=0)
        not_sure_label = torch.cat(not_sure_label, dim=0)
        not_sure_output = torch.cat(not_sure_output, dim=0)

        self.model.load_state_dict(global_para, strict=True)
        self.model.eval()
        with torch.no_grad():
            output = self.model(not_sure_data)
            output = lamb * output + (1 - lamb) * not_sure_output
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(not_sure_label.view_as(pred)).sum().item()

        return correct / len(test_loader.dataset) * 100, len(test_loader.dataset)
