class BaseServer():
    def __init__(self, args, clients):
        self.args = args
        self.clients = clients
    
    def cluster_aggregation(self, model_temp, data_nums):
        for i, models in enumerate(model_temp):
            if models:
                data_sum = 0.0
                for data in data_nums[i]:
                    data_sum += data
                model_para = {}
                for key, var in models[0].items():
                    model_para[key] = ( data_nums[i][0] / data_sum) * var.clone()
                for j in range(1, len(models)):
                    for key, var in models[j].items():
                        model_para[key] += ( data_nums[i][j] / data_sum) * var.clone()
                self.model_para[i] = model_para #这里要copy，不然
                # self.model_para[i] = copy.deepcopy(model_para)

    def aggregation(self, model_para, data_nums):
        res_model_para = {}
        data_sum = 0
        for data in data_nums:
            data_sum += data
        for key, var in model_para[0].items():
            res_model_para[key] = ( data_nums[0] / data_sum) * var.clone()
        for j in range(1, len(model_para)):
            for key, var in model_para[j].items():
                res_model_para[key] += ( data_nums[j] / data_sum) * var.clone()
        # self.model_para[i] = model_para #这里要copy，不然
        return res_model_para