class Feature:
    def __init__(self, name, dataset):
        self.name = name
        self.count = len(dataset)
        self.mean = sum(dataset) / self.count
        self.std = self.__calc_std(dataset)
        self.min = self.__calc_quartiles(dataset, 0)
        self.q_25 = self.__calc_quartiles(dataset, 25)
        self.q_50 = self.__calc_quartiles(dataset, 50)
        self.q_75 = self.__calc_quartiles(dataset, 75)
        self.max = self.__calc_quartiles(dataset, 100)
    
    def __calc_std(self, dataset):
        sum_squares = 0
        for i in range(len(dataset)):
            sum_squares += (dataset[i] - self.mean) ** 2
        std = sum_squares / (self.count - 1)
        std = std ** 0.5
        return std
    
    def __calc_quartiles(self, dataset, quartile):
        dataset.sort()
        position_min = (float(quartile) / 100) * (self.count - 1)
        position_max_coef = position_min - int(position_min)
        if position_max_coef == 0.0:
            return dataset[int(position_min)]
        position_max = position_min + 1
        position_min_coef = 1 - position_max_coef
        result_min = (dataset[int(position_min)] * position_min_coef)
        result_max = (dataset[int(position_max)] * position_max_coef)
        return result_min + result_max
    
    def return_info(self, info_to_return):
        info_to_return_formatted = '{:>.6f}'.format(info_to_return)
        info = '|'
        info += ' ' * (len(self.name) - len(info_to_return_formatted) + 5)
        info += info_to_return_formatted
        return info

    def get(self, info):
        if info == 'name':
            info = '|  {:>13s}'.format(self.name)
            return info
        if info == 'count':
            return self.return_info(self.count)
        if info == 'mean':
            return self.return_info(self.mean)
        if info == 'std':
            return self.return_info(self.std)
        if info == 'min':
            return self.return_info(self.min)
        if info == 'q_25':
            return self.return_info(self.q_25)
        if info == 'q_50':
            return self.return_info(self.q_50)
        if info == 'q_75':
            return self.return_info(self.q_75)
        if info == 'max':
            return self.return_info(self.max)
