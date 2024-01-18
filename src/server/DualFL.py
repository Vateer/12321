from .BaseServer import BaseServer
from src.utils import *
from src.models import *
import wandb
import torch.utils.data as data_utils
from sklearn.metrics.pairwise import euclidean_distances

class DualFLServer(BaseServer):
    def log_to_cluster(self, subloggers, content):
        for c in subloggers:
            c.save(content)

    def load_auxiliary_dataset(self, auxiliary_datas, auxiliary_labels):
        # 将辅助数据集构造成一个 Dataset 对象
        auxiliary_dataset = data_utils.TensorDataset(torch.from_numpy(auxiliary_datas), torch.from_numpy(auxiliary_labels))
        # 创建一个 DataLoader 对象
        auxiliary_dataloader = data_utils.DataLoader(auxiliary_dataset, batch_size=5, shuffle=False)
        self.loader = auxiliary_dataloader

    def calculate_gradient_sensitivity(self, model):
        model.eval()
        gradient_sensitivity = []
        for batch_idx, (datas, labels) in enumerate(self.loader):
            datas, labels = datas.to(self.args.device), labels.to(self.args.device)
            model.zero_grad()
            output = model(datas)
            loss = F.cross_entropy(output, labels.long())
            loss.backward()
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
            total_gradient = torch.cat(gradients)
            # gradient_norm = torch.norm(total_gradient, 2).item()
            gradient_norm = torch.abs(total_gradient).sum().item()
            gradient_sensitivity.append(gradient_norm)
        return np.array(gradient_sensitivity)


    def build_model_similarity_matrix(self):
        gradient_sensitivities = []
        model_temp = [[] for _ in range(len(self.cluster_ids))]
        data_nums = [[] for _ in range(len(self.cluster_ids))]

        for idx, client in enumerate(self.clients):
            cluster_id = self.client_belong[idx]
            client.set_parameter(self.model_para[cluster_id])
            _loss, data_num = client.train()
            model_temp[cluster_id].append(client.get_parameter())
            data_nums[cluster_id].append(data_num)
            client_model = client.model
            # client_model.load_state_dict(client.get_parameter())
            gradient_sensitivity = self.calculate_gradient_sensitivity(client_model)
            gradient_sensitivities.append(gradient_sensitivity)
        self.cluster_aggregation(model_temp, data_nums)
        gradient_sensitivities = np.vstack(gradient_sensitivities) #?
        I_matrix = euclidean_distances(gradient_sensitivities)
        return I_matrix

    def run(self):
        self.run_cluster_training()

    def test_cluster_model(self, path, logger):
        logger.save("evaluate cluster average model")
        self.cluster_ids = []
        #获得log path下的所有cluster文件夹
        cluster_dirs = [os.path.join(path, i) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]
        #恢复簇成员编号
        for cluster_dir in cluster_dirs:
            with open(os.path.join(cluster_dir, "cluster.txt"), "r") as f:
                self.cluster_ids.append(eval(f.read()))

        #initial model
        self.model = eval("{}{}".format(self.args.model, self.args.dataset.capitalize()))()
        self.model.to("cuda:{}".format(self.args.gpu))
        self.model_para = [self.model.state_dict() for _ in range(len(self.cluster_ids))]

        #initial client
        self.client_belong = [0] * self.args.clients
        for i, cluster_id in enumerate(self.cluster_ids):
            for id in cluster_id:
                self.clients[id].set_model(self.model)
                # self.clients.set_parameter(self.model_para[i])
                self.client_belong[id] = i

        #读取cluster文件内的cluster模型并平均
        model_temp = []
        data_nums = []
        for idx, cluster_dir in enumerate(cluster_dirs):
            model_temp.append([])
            data_nums.append([])
            for file in os.listdir(cluster_dir):
                if file.endswith(".pt"):
                    model_temp[idx].append(torch.load(os.path.join(cluster_dir, file)))
                    data_nums[idx].append(1)
            self.cluster_aggregation(model_temp, data_nums)

        logger.save("cluster number: ", end="")
        logger.save(len(self.cluster_ids))
        logger.save("cluster result: ")
        logger.save(self.cluster_ids)

        #测试簇模型的性能s
        logger.save("evaluation : ")
        total_acc = 0.0
        total_accs = []
        total_nums = []
        total_num = 0
        for i, cluster in enumerate(self.cluster_ids):
            accs = []
            nums = []
            cluster_num = 0
            for id in cluster:
                self.clients[id].set_parameter(self.model_para[i])
                _acc, num = self.clients[id].test()
                # logger.save("client{} : {}%".format(str(id), str(round(_acc, 3))))
                accs.append(_acc)
                nums.append(num)
                cluster_num += num
            acc = 0.0
            for ac, num in zip(accs, nums):
                acc += ac * num / cluster_num
            total_accs.append(acc)
            total_nums.append(cluster_num)
            logger.save("cluster {} : acc = {}%".format(str(i),str(round(acc, 2))))
            total_num += cluster_num
        for acc, num in zip(total_accs, total_nums):
            total_acc += acc * num / total_num
        logger.save("Total average acc : {}%".format(str(round(total_acc, 2))))


    def run_cluster_training(self):
        self.logger = Logger(self.args)
        logger = self.logger
        logger.save(self.args.tag)
        logger.save(str(self.args))
        for arg in vars(self.args):
            logger.save(f"{arg}: {getattr(self.args, arg)}")
        #clustering
        svds = []
        for client in self.clients:
            svds.append(client.get_svd())
        self.A_matrix = calculating_adjacency(self.args, svds)
        self.A_matrix = self.A_matrix / np.linalg.norm(self.A_matrix)
        # self.cluster_ids = hierarchical_clustering(self.A_matrix, self.args.hc_clustering, "average")
        # 0.006 (你可以比一下没有归一化的，还是要optics)
        # self.cluster_ids = hierarchical_clustering(self.A_matrix, self.args.hc_clustering, "average")
        self.cluster_ids = optics_clustering(self.A_matrix, self.args.optics_min_sample, self.args.optics_xi)

        logger.save("cluster number: ", end="")
        logger.save(len(self.cluster_ids))
        logger.save("cluster result: ")
        logger.save(self.cluster_ids)

        subloggers = [SubLogger(self.args, logger.get_log_dir(), "cluster{}.txt".format(str(i))) for i in range(len(self.cluster_ids))]
        for id, cluster in enumerate(self.cluster_ids):
            subloggers[id].save("cluster member: ")
            subloggers[id].save(cluster)

        #initial model
        self.model = eval("{}{}".format(self.args.model, self.args.dataset.capitalize()))()
        self.model.to("cuda:{}".format(self.args.gpu))
        self.model_para = [self.model.state_dict() for _ in range(len(self.cluster_ids))]

        #initial client
        self.client_belong = [0] * self.args.clients
        for i, cluster_id in enumerate(self.cluster_ids):
            for id in cluster_id:
                self.clients[id].set_model(self.model)
                # self.clients.set_parameter(self.model_para[i])
                self.client_belong[id] = i

        loss_record = [[] for _ in range(len(self.cluster_ids))] #记录簇的Loss
        acc_record = [[] for _ in range(len(self.cluster_ids))] #记录簇的Acc
        eacc_record = [[] for _ in range(len(self.cluster_ids))] #记录簇的估计acc
        eacc_minus = []
        logger.save("##########training begin##########")

        recluster_cnt = 0
        for epoch in range(self.args.rounds):
            logger.save("\n===========round {}: ===========".format(epoch + 1))
            wandb_log = {}
            self.log_to_cluster(subloggers=subloggers, content="round{}".format(str(epoch + 1)))
            # for id, cluster_id in enumerate(self.cluster_ids):
            #     logger.save("cluster {}:".format(id))

            if epoch and epoch % self.args.recluster_delta == 0 and epoch != self.args.rounds - 1:
                self.args.lr_now *= (1 - self.args.lr_decay)
                logger.save("Check if need recluster")
                lc_threshold = self.args.lc_threshold  
                need_recluster = False

                # #使用前5轮的数据计算eacc_minus
                # for cluster_id, eaccs in enumerate(eacc_record):
                #     eacc_sum = 0.0
                #     for i, acc in enumerate(eaccs[:5]):
                #         eacc_sum += acc
                #     eacc_minus.append(eacc_sum / 5)
                # for cluster_id, (eaccs, losses) in enumerate(zip(eacc_record, loss_record)):
                #     #使用后1/3的数据计算LC
                #     loss_sum = np.sum(losses)
                #     lc = 0
                #     for i, acc in enumerate(eaccs[-int(len(eaccs)/3):]):
                #         lc += ( acc - eacc_minus[cluster_id] ) / loss_sum
                #     lc /= len(eaccs[-int(len(eaccs)/3):])
                #     if lc < lc_threshold:
                #         need_recluster = True
                #         logger.save(f"Cluster {cluster_id} needs recluster, LC: {lc:.5f}")
                #     else:
                #         logger.save(f"Cluster {cluster_id} is stable, LC: {lc:.5f}")

                #基于后1/3的eacc的方差判断簇是否需要聚类
                for cluster_id, eaccs in enumerate(eacc_record):
                    eaccs = eaccs[-int(len(eaccs)/3):]
                    eacc_std = np.std(eaccs)
                    if eacc_std > self.args.std_threshold:
                        need_recluster = True
                        logger.save(f"Cluster {cluster_id} needs recluster, std: {eacc_std:.5f}")
                    else:
                        logger.save(f"Cluster {cluster_id} is stable, std: {eacc_std:.5f}")

                eacc_minus = []
                if need_recluster:
                    self.args.lr_now = self.args.lr
                    #保存簇模型
                    for i, cluster in enumerate(self.cluster_ids):
                        model_path = os.path.join(logger.get_log_dir(), "cluster{}".format(i))
                        if not os.path.exists(model_path):
                            os.makedirs(model_path, exist_ok=True)
                        torch.save(self.model_para[i], os.path.join(model_path, "epoch{}.pt".format(str(epoch))))

                    recluster_cnt += 1
                    I_matrix = self.build_model_similarity_matrix()
                    I_matrix = I_matrix / np.linalg.norm(I_matrix)
                    M_matrix = 0.5 ** recluster_cnt * self.A_matrix + (1 - 0.5 ** recluster_cnt) * I_matrix
                    M_matrix = M_matrix / np.linalg.norm(M_matrix)
                    self.cluster_ids = optics_clustering(M_matrix, self.args.optics_min_sample, self.args.optics_xi)

                    #更新logger， wandb
                    self.log_to_cluster(subloggers=subloggers, content="recluster")
                    logger.new_cluster(recluster_cnt)
                    subloggers = [SubLogger(self.args, logger.get_log_dir(), "cluster{}.txt".format(str(i))) for i in range(len(self.cluster_ids))]
                    for id, cluster in enumerate(self.cluster_ids):
                        subloggers[id].save("cluster member: ")
                        subloggers[id].save(cluster)
                    logger.save("cluster number: ", end="")
                    logger.save(len(self.cluster_ids))
                    logger.save("cluster result: ")
                    logger.save(self.cluster_ids)
                    loss_record = [[] for _ in range(len(self.cluster_ids))]
                    acc_record = [[] for _ in range(len(self.cluster_ids))] 
                    eacc_record = [[] for _ in range(len(self.cluster_ids))]

                    if self.args.wandb == True:
                        wandb.finish()
                        self.args.wandb_name = self.args.wandb_name[:-1]+str(recluster_cnt)
                        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)

                    #暂存形成的新的簇
                    model_temp = [[] for _ in range(len(self.cluster_ids))]
                    data_nums = [[] for _ in range(len(self.cluster_ids))]
                    for i, cluster in enumerate(self.cluster_ids):
                        for id in cluster:
                            model_temp[i].append(self.model_para[self.client_belong[id]])
                            data_nums[i].append(1)
                            
                    # 更新id
                    for i, cluster in enumerate(self.cluster_ids):
                        for client_id in cluster:
                            self.client_belong[client_id] = i

                    #更新簇模型
                    self.model_para = [self.model.state_dict() for _ in range(len(self.cluster_ids))] #初始化
                    self.cluster_aggregation(model_temp, data_nums)

            #这里主要是要不要按cluster抽
            # selected = np.random.choice(np.array(range(clients_num)), size=int(clients_num * self.args.frac), replace=False)
            selected = np.array([])
            for cluster in self.cluster_ids:
                s = np.random.choice(np.array(range(len(cluster))), size=max(round_with_probability(len(cluster) * self.args.frac), 1), replace=False)
                selected = np.append(selected, np.array(cluster)[s], axis=0)
            selected = selected.astype(np.int_)
            logger.save("selected list = ")
            logger.save(selected,end="\n\n")
            
            # model_temp = [[]]*len(self.cluster_ids)
            # data_nums = [[]]*len(self.cluster_ids)
            # 上面这样有问题，使用*对象会导致list的各个对象都是一样的，需要下面这样
            model_temp = [[] for _ in range(len(self.cluster_ids))]
            data_nums = [[] for _ in range(len(self.cluster_ids))]

            cluster_loss = [0] * len(self.cluster_ids)
            loss_cnt = [0] * len(self.cluster_ids)
            eaccs = [[] for _ in range(len(self.cluster_ids))]
            eacc_num = [[] for _ in range(len(self.cluster_ids))]

            # train
            for c in selected:
                cluster_id = self.client_belong[c]
                self.clients[c].set_parameter(self.model_para[cluster_id])
                #估计acc
                eacc, test_num = self.clients[c].test()
                eaccs[cluster_id].append(eacc)
                eacc_num[cluster_id].append(test_num) 
                _loss, data_num = self.clients[c].train()
                cluster_loss[cluster_id] += _loss
                loss_cnt[cluster_id] += 1
                model_temp[cluster_id].append(self.clients[c].get_parameter())
                data_nums[cluster_id].append(data_num)
                subloggers[self.client_belong[c]].save("Client {}: loss = {}, eacc: {}".format(str(c), str(round(_loss, 3)), str(eaccs[cluster_id][-1])))

            logger.save("training loss : ")
            total_loss = 0.0
            cnt_loss = 0
            for i in range(self.cluster_ids.__len__()):
                if loss_cnt[i]:
                    logger.save("cluster {}: {}\n\n".format(str(i), str(round(cluster_loss[i]/loss_cnt[i], 5))))
                    subloggers[i].save("total loss: {}".format(str(round(cluster_loss[i]/loss_cnt[i], 5))))
                    loss_record[i].append(round(cluster_loss[i]/loss_cnt[i], 5))
                    total_loss += loss_record[i][-1]
                    cnt_loss += 1
                wandb_log["loss_cluster {}".format(str(i))] = loss_record[i][-1]
            wandb_log["loss"] = round(total_loss/cnt_loss, 5)

            # aggregation
            self.cluster_aggregation(model_temp, data_nums)

            logger.save("evaluation : ")
            total_acc = 0.0
            total_accs = []
            total_nums = []
            total_num = 0
            # evaluation_actual_acc
            for i, cluster in enumerate(self.cluster_ids):
                acc = 0.0
                accs = []
                data_nums = []
                data_num_sum = 0
                for id in cluster:
                    self.clients[id].set_parameter(self.model_para[i])
                    _acc, _num = self.clients[id].test()
                    subloggers[i].save("client{} : {}%".format(str(id), str(round(_acc, 3))))
                    # logger.save("client {}: {}".format(str(id), str(_acc.item())))
                    accs.append(_acc)
                    data_nums.append(_num)
                    data_num_sum += _num
                
                for ac, num in zip(accs, data_nums):
                    acc += ac * num / data_num_sum
                total_accs.append(acc)
                total_nums.append(data_num_sum)
                total_num += data_num_sum

                #计算簇的加权 eacc
                eacc_sum = 0.0
                eacc_avg = 0.0
                for n in eacc_num[i]:
                    eacc_sum += n
                for e, n in zip(eaccs[i], eacc_num[i]):
                    eacc_avg += e * n / eacc_sum
                eacc_record[i].append(eacc_avg)

                logger.save("cluster {} : acc = {}%, eacc = {}%".format(str(i),str(round(acc, 2)), eacc_avg))
                wandb_log["acc_cluster {}".format(str(i))] = round(acc, 2)
                wandb_log["eacc_cluster {}".format(str(i))] = eacc_avg
                acc_record[i].append(round(acc, 2))

            for acc, num in zip(total_accs, total_nums):
                total_acc += acc * num / total_num
            logger.save("average acc : {}%".format(str(round(total_acc, 2))))
            wandb_log["acc"] = total_acc
            if self.args.wandb == True:
                wandb.log(wandb_log)
            
            # save model
            if epoch >= self.args.rounds - 10:
                for i, cluster in enumerate(self.cluster_ids):
                    model_path = os.path.join(logger.get_log_dir(), "cluster{}".format(i))
                    if not os.path.exists(model_path):
                        os.makedirs(model_path, exist_ok=True)
                    torch.save(self.model_para[i], os.path.join(model_path, "epoch{}.pt".format(str(epoch))))
        #保存簇成员编号
        for i, cluster in enumerate(self.cluster_ids):
            model_path = os.path.join(logger.get_log_dir(), "cluster{}".format(i))
            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, "cluster.txt"), "w") as f:
                f.write(str(cluster))
        
        self.test_cluster_model(logger.get_log_dir(), logger)
        

    def run_global_model_training(self, log_path=None):
        if not log_path: #没有就表示有self.logger
            log_path = self.logger.get_log_dir()
            #回到父目录
            self.logger._write_buffer_to_file()
            self.logger.path = os.path.dirname(self.logger.path)
        else:
            self.logger = Logger(self.args)
            self.logger.path = os.path.dirname(self.logger.path)
        logger = self.logger
        logger.save(self.args.tag)
        logger.save(str(self.args))
        logger.save("global model training process begin")
        self.cluster_ids = []
        #获得log path下的所有cluster文件夹
        cluster_dirs = [os.path.join(log_path, i) for i in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, i))]
        #恢复簇成员编号
        for cluster_dir in cluster_dirs:
            with open(os.path.join(cluster_dir, "cluster.txt"), "r") as f:
                self.cluster_ids.append(eval(f.read()))

        #initial model
        self.model = eval("{}{}".format(self.args.model, self.args.dataset.capitalize()))()
        self.model.to("cuda:{}".format(self.args.gpu))
        self.model_para = [self.model.state_dict() for _ in range(len(self.cluster_ids))]

        #initial client
        self.client_belong = [0] * self.args.clients
        for i, cluster_id in enumerate(self.cluster_ids):
            for id in cluster_id:
                self.clients[id].set_model(self.model)
                # self.clients.set_parameter(self.model_para[i])
                self.client_belong[id] = i

        #读取cluster文件内的cluster模型并平均
        model_temp = []
        data_nums = []
        for idx, cluster_dir in enumerate(cluster_dirs):
            model_temp.append([])
            data_nums.append([])
            for file in os.listdir(cluster_dir):
                if file.endswith(".pt"):
                    model_temp[idx].append(torch.load(os.path.join(cluster_dir, file)))
                    data_nums[idx].append(1)
            self.cluster_aggregation(model_temp, data_nums)

        logger.save("cluster number: ", end="")
        logger.save(len(self.cluster_ids))
        logger.save("cluster result: ")
        logger.save(self.cluster_ids)

        #测试簇模型的性能
        logger.save("evaluation : ")
        total_acc = 0.0
        total_accs = []
        total_nums = []
        total_num = 0
        for i, cluster in enumerate(self.cluster_ids):
            accs = []
            nums = []
            cluster_num = 0
            for id in cluster:
                self.clients[id].set_parameter(self.model_para[i])
                _acc, num = self.clients[id].test()
                # logger.save("client{} : {}%".format(str(id), str(round(_acc, 3))))
                accs.append(_acc)
                nums.append(num)
                cluster_num += num
            acc = 0.0
            for ac, num in zip(accs, nums):
                acc += ac * num / cluster_num
            total_accs.append(acc)
            total_nums.append(cluster_num)
            logger.save("cluster {} : acc = {}%".format(str(i),str(round(acc, 2))))
            total_num += cluster_num
        for acc, num in zip(total_accs, total_nums):
            total_acc += acc * num / total_num
        logger.save("Total average acc : {}%".format(str(round(total_acc, 2))))

        #将簇模型平均聚合成一个全局模型
        # global_para = {}
        # for model_para in self.model_para:
        #     for key, var in model_para.items():
        #         if key in global_para:
        #             global_para[key] += var.clone()
        #         else:
        #             global_para[key] = var.clone()
        # for key, var in global_para.items():
        #     global_para[key] /= len(self.model_para)
        global_para = self.aggregation(self.model_para, [1] * len(self.model_para))
        
        #测试
        logger.save("evaluation : ")
        accs = []
        total_num = 0
        for client in self.clients:
            client.set_parameter(global_para)
            acc, num = client.test()
            accs.append((acc, num))
            total_num += num
        total_acc = 0.0
        for acc, num in accs:
            total_acc += acc * num / total_num
        logger.save("average acc : {}%".format(str(round(total_acc, 2))))
        
        for epoch in range(self.args.ft_rounds):
            logger.save("\n===========round {}: ===========".format(epoch + 1))
            wandb_log = {}
            
            
            # selected = np.random.choice(np.array(range(self.args.clients)), size=int(self.args.clients * self.args.frac) , replace=False)
            # logger.save("selected list = ")
            # logger.save(selected,end="\n")
            selected = np.array([])
            for cluster in self.cluster_ids:
                s = np.random.choice(np.array(range(len(cluster))), size=max(round_with_probability(len(cluster) * self.args.ft_frac), 1), replace=False)
                selected = np.append(selected, np.array(cluster)[s], axis=0)
            selected = selected.astype(np.int_)
            logger.save("selected list = ")
            logger.save(selected,end="\n\n")

            data_nums = [] 
            paras = []
            #开始训练
            for c in selected:
                self.clients[c].set_parameter(global_para)
                # loss, num = self.clients[c].fine_tune_global_model_sgd()
                loss, num = self.clients[c].fine_tune_global_model()
                data_nums.append(num)
                logger.save("Client {}: loss = {}".format(str(c), str(round(loss, 3))))
                paras.append(self.clients[c].get_parameter())
            
            #聚合
            global_para = self.aggregation(paras, data_nums)

            #测试
            logger.save("evaluation : ")
            accs = []
            total_num = 0
            for client in self.clients:
                client.set_parameter(global_para)
                acc, num = client.test()
                accs.append((acc, num))
                total_num += num
            total_acc = 0.0
            for acc, num in accs:
                total_acc += acc * num / total_num
            logger.save("average acc : {}%".format(str(round(total_acc, 2))))
            wandb_log["acc"] = total_acc
        
        #保存最后的全局模型
        torch.save(global_para, os.path.join(logger.get_log_dir(), "global_model.pt"))

        #保存簇信息
        for i, cluster in enumerate(self.cluster_ids):
            torch.save(self.model_para[i], os.path.join(logger.get_log_dir(), "cluster{}.pt".format(i)))
            #保存簇id信息
            with open(os.path.join(logger.get_log_dir(), "cluster{}.txt".format(i)), "w") as f:
                f.write(str(cluster))

    def inference(self, log_path=None):
        if not log_path:
            log_path = self.logger.get_log_dir()
            #回到父目录
            self.logger._write_buffer_to_file()
        else:
            self.logger = Logger(self.args)
            self.logger.path = os.path.dirname(self.logger.path)
        logger = self.logger
        logger.save(self.args.tag)
        logger.save(str(self.args))
        logger.save("inference process begin")
        self.cluster_ids = []
        #根据log_path恢复簇信息
        for cid in range(100):
            path = os.path.join(log_path, "cluster{}.txt".format(str(cid)))
            if not os.path.exists(path):
                break
            with open(path, "r") as f:
                self.cluster_ids.append(eval(f.read()))
        #恢复簇模型
        self.model_para = []
        for cid in range(len(self.cluster_ids)):
            path = os.path.join(log_path, "cluster{}.pt".format(str(cid)))
            self.model_para.append(torch.load(path))
        #恢复全局模型
        global_para = torch.load(os.path.join(log_path, "global_model.pt"))

        #initial model
        self.model = eval("{}{}".format(self.args.model, self.args.dataset.capitalize()))()
        self.model.to("cuda:{}".format(self.args.gpu))

        #initial client
        self.client_belong = [0] * self.args.clients
        for i, cluster_id in enumerate(self.cluster_ids):
            for id in cluster_id:
                self.clients[id].set_model(self.model)
                # self.clients.set_parameter(self.model_para[i])
                self.client_belong[id] = i
        
        #测试簇模型的性能
        logger.save("evaluation : ")
        total_acc = 0.0
        total_accs = []
        total_nums = []
        total_num = 0

        # wrongs = []
        # cs = []
        # ori_accs = []
        # twrongs = []
        # tcs = []
        # tori_accs = []

        for i, cluster in enumerate(self.cluster_ids):
            accs = []
            nums = []
            cluster_num = 0
            for id in cluster:
                self.clients[id].set_parameter(self.model_para[i])
                _acc, num = self.clients[id].inference_with_global_model(global_para)
                # _acc, num = self.clients[id].test()
                # _acc, num, wrong, c, _ori_acc = self.clients[id].inference_with_global_model(global_para)
                # _acc, num, wrong, c, _ori_acc, twrong, tc = self.clients[id].inference_with_global_model(global_para)
                # wrongs += wrong
                # cs += c
                # wrongs.append(wrong)
                # cs.append(c)
                # ori_accs.append(_ori_acc)
                # twrongs.append(twrong)
                # tcs.append(tc)
                
                # logger.save("client{} : {}%".format(str(id), str(round(_acc, 4))))
                accs.append(_acc)
                nums.append(num)
                cluster_num += num
            acc = 0.0
            for ac, num in zip(accs, nums):
                acc += ac * num / cluster_num
            total_accs.append(acc)
            total_nums.append(cluster_num)
            logger.save("cluster {} : acc = {}%".format(str(i),str(round(acc, 4))))
            total_num += cluster_num

        for acc, num in zip(total_accs, total_nums):
            total_acc += acc * num / total_num
        logger.save("Total average acc : {}%".format(str(round(total_acc, 4))))

        

        
    def inference_global(self, log_path=None, test_loader=None):
        if not log_path:
            log_path = self.logger.get_log_dir()
            #回到父目录
            self.logger._write_buffer_to_file()
        else:
            self.logger = Logger(self.args)
            self.logger.path = os.path.dirname(self.logger.path)
        logger = self.logger
        logger.save(self.args.tag)
        logger.save(str(self.args))
        logger.save("inference process begin")
        self.cluster_ids = []
        #根据log_path恢复簇信息
        for cid in range(100):
            path = os.path.join(log_path, "cluster{}.txt".format(str(cid)))
            if not os.path.exists(path):
                break
            with open(path, "r") as f:
                self.cluster_ids.append(eval(f.read()))
        #恢复簇模型
        self.model_para = []
        for cid in range(len(self.cluster_ids)):
            path = os.path.join(log_path, "cluster{}.pt".format(str(cid)))
            self.model_para.append(torch.load(path))
        #恢复全局模型
        global_para = torch.load(os.path.join(log_path, "global_model.pt"))

        #initial model
        self.model = eval("{}{}".format(self.args.model, self.args.dataset.capitalize()))()
        self.model.to("cuda:{}".format(self.args.gpu))

        #initial client
        self.client_belong = [0] * self.args.clients
        for i, cluster_id in enumerate(self.cluster_ids):
            for id in cluster_id:
                self.clients[id].set_model(self.model)
                # self.clients.set_parameter(self.model_para[i])
                self.client_belong[id] = i
        
        #测试簇模型的性能
        logger.save("evaluation : ")
        total_acc = 0.0
        total_accs = []

        # wrongs = []
        # cs = []
        # ori_accs = []
        # twrongs = []
        # tcs = []
        # tori_accs = []

        for i, cluster in enumerate(self.cluster_ids):
            accs = []
            cluster_num = 0
            for id in cluster:
                self.clients[id].set_parameter(self.model_para[i])
                _acc, num = self.clients[id].inference_with_global_model(global_para, test_loader)
                # _acc, num = self.clients[id].test()
                # _acc, num, wrong, c, _ori_acc = self.clients[id].inference_with_global_model(global_para)
                # _acc, num, wrong, c, _ori_acc, twrong, tc = self.clients[id].inference_with_global_model(global_para)
                # wrongs += wrong
                # cs += c
                # wrongs.append(wrong)
                # cs.append(c)
                # ori_accs.append(_ori_acc)
                # twrongs.append(twrong)
                # tcs.append(tc)
                
                # logger.save("client{} : {}%".format(str(id), str(round(_acc, 4))))
                accs.append(_acc)
                break
            acc = sum(accs) / len(accs)
            total_accs.append(acc)
            logger.save("cluster {} : acc = {}%".format(str(i),str(round(acc, 4))))
        total_acc = sum(total_accs)/len(total_accs)

        logger.save("Total average acc : {}%".format(str(round(total_acc, 4))))
            


        

