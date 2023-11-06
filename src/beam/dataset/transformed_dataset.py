import torch

from ..utils import check_type
from .universal_dataset import UniversalDataset


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, alg, *args, **kwargs):
        super().__init__()

        if type(dataset) != UniversalDataset:
            dataset = UniversalDataset(dataset)

        self.dataset = dataset
        self.alg = alg
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, ind):

        ind_type = check_type(ind, check_element=False)
        if ind_type.major == 'scalar':
            ind = [ind]

        ind, data = self.dataset[ind]
        dataset = UniversalDataset(data)
        res = self.alg.predict(dataset, *self.args, **self.kwargs)

        return ind, res.values
