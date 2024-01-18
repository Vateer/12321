from torchvision import datasets, transforms
import numpy as np
import torch, argparse
from torch.utils.data import DataLoader, TensorDataset

def make_dataloader(args, data_split, split_from_trainset = False, test_datas = None, test_labels = None):
    ind = np.array(list(range(len(data_split[0]))))
    np.random.shuffle(ind)
    if split_from_trainset == False:
        args.test_frac = 0.0

    split_point = int(len(ind) * (1.0 - args.test_frac))
    

    train_data_tensor = torch.Tensor(data_split[0][ind][:split_point])
    trian_label_tensor = torch.Tensor(data_split[1][ind][:split_point])
    train_dataset = TensorDataset(train_data_tensor, trian_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)

    if split_from_trainset == True:
        test_data_tensor = torch.Tensor(data_split[0][ind][split_point:])
        test_label_tensor = torch.Tensor(data_split[1][ind][split_point:])
        test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    else:
        label_type = np.unique(data_split[1])
        ind = np.array([])
        for label in label_type:
            ind = np.append(ind, np.where(test_labels == label)[0])
        ind = ind.astype(int)
        test_data_tensor = torch.Tensor(test_datas[ind])
        test_label_tensor = torch.Tensor(test_labels[ind])
        test_dataset = TensorDataset(test_data_tensor, test_label_tensor)
        # print(np.unique(np.array(test_label_tensor)))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, test_loader

def partition_data(args, datas, labels, train = True):
    if args.dataset == "mix4":
        return partition_data_mix4(args, datas, labels, train)
    return eval("partition_data_{}".format(args.distribution))(args, datas, labels)

def partition_data_mix4(args, datas, labels, train):
    #按照四个数据集的数量比例将它们分别划分到不同客户端
    data_splits = []
    num_clients = args.clients
    #各个数据集应该划分的客户端数
    data_sum = 0
    for label in labels:
        data_sum += label.sum()
    nums = []
    for label in labels:
        nums.append(int(label.sum()/data_sum*num_clients) + 1 )
    nums[-1] = num_clients - sum(nums[:-1])
    # ["cifar10", "usps", "fmnist", "svhn"]
    nums = [32, 15, 28, 25]
    # print(nums)
    #按照客户端数量均匀划分数据
    for idx, data, label in zip(range(len(datas)), datas, labels):
        data_split = partition_data_iid_mix4(args, data, label, num_clients = nums[idx], train = train)
        data_splits.extend(data_split)
    return data_splits
        
def partition_data_iid_mix4(args, datas, labels, num_clients = 0, train = True):
    data_splits = []
    if num_clients == 0:
        num_clients = args.clients
    num_samples_per_client = 50

    # 将数据集随机打乱
    indices = np.random.permutation(len(labels))
    datas = datas[indices]
    labels = labels[indices]

    #按照类分配，每个类500个

    unique_classes = np.unique(labels)
    for i in range(num_clients):
        data_split = []
        for label in unique_classes:
            indices = np.where(labels == label)[0]
            indices = indices[i*num_samples_per_client:(i+1)*num_samples_per_client]
            if data_split == []:
                data_split = [datas[indices], labels[indices]]
            else:
                data_split[0] = np.append(data_split[0], datas[indices], axis=0)
                data_split[1] = np.append(data_split[1], labels[indices], axis=0)
        data_splits.append(data_split)
    return data_splits
    

def partition_data_iid(args, datas, labels):
    data_splits = []
    num_clients = args.clients
    num_samples_per_client = len(labels) // num_clients

    # 将数据集随机打乱
    indices = np.random.permutation(len(labels))
    datas = datas[indices]
    labels = labels[indices]

    # 按照客户端数量均匀划分数据
    for i in range(num_clients):
        start = i * num_samples_per_client
        end = (i + 1) * num_samples_per_client
        data_splits.append([datas[start:end], labels[start:end]])

    return data_splits

def partition_data_pat(args, datas, labels):
    data_splits = []
    num_clients = args.clients
    class_per_client = args.pat_cls
    balance = False

    data_splits = [[] for _ in range(num_clients)]
    unique_classes = np.unique(labels)
    num_class = unique_classes.__len__()
    idxs = np.array(range(len(labels)))
    idxs_each_class = []
    for i in range(num_class):
        idxs_each_class.append(idxs[labels == i])
    class_num_per_client = [class_per_client] * num_clients
    for i in range(num_class):
        selected = []
        for client_idx in range(num_clients):
            if class_num_per_client[client_idx] > 0:
                selected.append(client_idx)
        selected = selected[:int(np.ceil(num_clients/num_class * class_per_client))] #每个类分配的客户端数量

        sample_num = len(idxs_each_class[i])
        sample_per_client = sample_num / len(selected) #每个客户端分配的平均数量
        if balance == True:
            num_samples = [int(sample_per_client) for _ in range(len(selected))]
        else:
            # num_samples = np.random.randint(max(sample_per_client/10, least_sample/num_class), sample_per_client, len(selected)-1).tolist()
            num_samples = [int(sample_per_client) for _ in range(len(selected))]
            for idx in range(num_samples.__len__()//2):
                delta = np.random.randint(-(num_samples[idx] // 1.5), num_samples[idx] // 1.5)
                num_samples[idx] -= delta
                num_samples[num_samples.__len__() - idx - 1] += delta

        idx = 0
        for client, num in zip(selected, num_samples):
            if data_splits[client] == []:
                data_splits[client] = [datas[idxs_each_class[i][idx:idx+num]], labels[idxs_each_class[i][idx:idx+num]]]
            else:
                data_splits[client][0] = np.append(data_splits[client][0], datas[idxs_each_class[i][idx:idx+num]], axis=0)
                data_splits[client][1] = np.append(data_splits[client][1], labels[idxs_each_class[i][idx:idx+num]], axis=0)
            idx += num
            class_num_per_client[client] -= 1
    
    return data_splits

#按Dirichlet分布划分
def partition_data_dir(args, datas, labels):
    data_splits = []
    least_sample = args.local_bs * 2
    num_clients = args.clients
    alpha = args.dir_alpha

    min_size = 0
    num_samples = labels.shape[0]
    classes = np.unique(labels).__len__()
    unique_classes = np.unique(labels)
    while min_size < least_sample:
        idx_batch = [[] for _ in range(num_clients)]
        for cs in range(len(unique_classes)):
            idx_cs = np.where(labels == cs)[0]
            np.random.shuffle(idx_cs)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p*(len(idx)<num_samples/num_clients) for p,idx in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_cs)).astype(int)[:-1] #获得各个客户端样本的起始坐标
            # np.split(idx_cs,proportions) 将idx_cs按照proportions的下标分割
            idx_batch = [idx + idx2.tolist() for idx,idx2 in zip(idx_batch,np.split(idx_cs,proportions))] 
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for client in range(num_clients):
        data_splits.append([datas[idx_batch[client]], labels[idx_batch[client]]])

    return data_splits

def handle_data(args, join = False):
    if args.dataset == "mix4":
        return handle_data_mix4(args)
    if join:
        return handle_data_join(args)
    else:
        return handle_data_split(args)

def handle_data_mix4(args):
    train_datass = []
    train_labelss = []
    test_datass = []
    test_labelss = []
    arr = ["cifar10", "usps", "fmnist", "svhn"]
    for idx, dataset in enumerate(arr):
        handle_data = eval("handle_{}".format(dataset))
        train_ds, test_ds = handle_data(resize = True)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds.data), shuffle=False)
        for data in train_loader:
            train_datas, train_labels = data

        train_datas = train_datas.cpu().detach().numpy()
        train_labels = train_labels.cpu().detach().numpy()
        #labels集体加上偏移量
        train_labels += idx * 10

        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds.data), shuffle=False)
        for data in test_loader:
            test_datas, test_labels = data

        test_datas = test_datas.cpu().detach().numpy()
        test_labels = test_labels.cpu().detach().numpy()    
        test_labels += idx * 10
        train_datass.append(train_datas)
        train_labelss.append(train_labels)
        test_datass.append(test_datas)
        test_labelss.append(test_labels)
    return train_datass, train_labelss, test_datass, test_labelss



def handle_data_split(args):
    train_ds, test_ds = eval("handle_{}".format(args.dataset))()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds.data), shuffle=False)
    for data in train_loader:
        train_datas, train_labels = data

    train_datas = train_datas.cpu().detach().numpy()
    train_labels = train_labels.cpu().detach().numpy()

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds.data), shuffle=False)
    for data in test_loader:
        test_datas, test_labels = data

    test_datas = test_datas.cpu().detach().numpy()
    test_labels = test_labels.cpu().detach().numpy()

    return train_datas, train_labels, test_datas, test_labels


def handle_data_join(args):
    train_ds, test_ds = eval("handle_{}".format(args.dataset))()
    data_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds.data), shuffle = False)
    for data in data_loader:
        train_ds.data, train_ds.targets = data
    datas, labels = [], []
    datas.extend(train_ds.data.cpu().detach().numpy())
    labels.extend(train_ds.targets.cpu().detach().numpy())
    datas = np.array(datas)
    labels = np.array(labels)

    data_loader = torch.utils.data.DataLoader(test_ds, batch_size=len(test_ds.data), shuffle = False)
    for data in data_loader:
        test_ds.data, test_ds.targets = data
    datas2, labels2 = [], []
    datas2.extend(test_ds.data.cpu().detach().numpy())
    labels2.extend(test_ds.targets.cpu().detach().numpy())
    datas2 = np.array(datas2)
    labels2 = np.array(labels2)

    datas = np.append(datas, datas2, axis=0)
    labels = np.append(labels, labels2, axis=0)

    return datas, labels, None, None
def create_auxiliary_dataset(args, datas, labels):
    if args.dataset == "mix4":
        auxiliary_datas = []
        auxiliary_labels = []
        datass = []
        labelss = []
        for data, label in zip(datas, labels):
            auxiliary_data, auxiliary_label, data, label = _create_auxiliary_dataset(data, label)
            if auxiliary_datas == []:
                auxiliary_datas = auxiliary_data
                auxiliary_labels = auxiliary_label
            else:
                auxiliary_datas = np.append(auxiliary_datas, auxiliary_data, axis=0)
                auxiliary_labels = np.append(auxiliary_labels, auxiliary_label, axis=0)
            datass.append(data)
            labelss.append(label)
        return auxiliary_datas, auxiliary_labels, datass, labelss
    return _create_auxiliary_dataset(datas, labels)

#构造辅助数据集
def _create_auxiliary_dataset(datas, labels):
    unique_labels = np.unique(labels)
    auxiliary_datas = []
    auxiliary_labels = []

    for label in unique_labels:
        # 找到属于当前类别的数据索引
        indices = np.where(labels == label)[0]
        # 随机抽取数据
        num_samples = int(len(indices) * 0.01)  # 0.5%的数据量
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        # 将选中的数据和标签添加到辅助数据集中
        auxiliary_datas.extend(datas[selected_indices])
        auxiliary_labels.extend(labels[selected_indices])
        # 从原始数据集中删除选中的数据
        datas = np.delete(datas, selected_indices, axis=0)
        labels = np.delete(labels, selected_indices, axis=0)

    return np.array(auxiliary_datas), np.array(auxiliary_labels), datas, labels

#返回dataset类
def handle_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    data_train = datasets.MNIST(root = "./data",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.MNIST(root = "./data",
                            transform = transform,
                            train = False, download=True)
    return data_train, data_test

def handle_usps(resize = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    if resize == True:
        #resize 到 32*32
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.Grayscale(num_output_channels=3),  # 将1通道转换为3通道
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # 使用相同的均值和标准差对每个通道进行归一化
                                        ])
    data_train = datasets.USPS(root = "./data",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.USPS(root = "./data",
                            transform = transform,
                            train = False, download=True)
    return data_train, data_test

def handle_cifar10(resize = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    data_train = datasets.CIFAR10(root = "./data",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.CIFAR10(root = "./data",
                            transform = transform,
                            train = False, download=True)
    return data_train, data_test

def handle_cifar100():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0, np.array([63.0, 62.1, 66.7]) / 255.0),
                                    ])
    data_train = datasets.CIFAR100(root = "./data",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.CIFAR100(root = "./data",
                            transform = transform,
                            train = False, download=True)
    return data_train, data_test

def handle_fmnist(resize = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    if resize == True:
        #resize 到 32*32
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.Grayscale(num_output_channels=3),  # 将通道数调整为3
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))  # 使用相同的均值和标准差对每个通道进行归一化
                                        ])
    data_train = datasets.FashionMNIST(root = "./data",
                                transform=transform,
                                train = True,
                                download = True)

    data_test = datasets.FashionMNIST(root = "./data",
                            transform = transform,
                            train = False, download=True)
    return data_train, data_test

def handle_svhn(resize = False):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    if resize == True:
        #resize 到 32*32
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    data_train = datasets.SVHN(root = "./data",
                                transform=transform,
                                split='train',
                                download = True)

    data_test = datasets.SVHN(root = "./data",
                            transform = transform,
                            split='test', download=True)
    return data_train, data_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.dataset = "mnist"
    handle_data(parser)