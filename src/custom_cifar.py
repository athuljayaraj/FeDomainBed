from torchvision import datasets
import numpy as np

class CustomCIFAR10Loader(datasets.CIFAR10):
     def __init__(self, *args, exclude_list=[], **kwargs):
        super(CustomCIFAR10Loader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return

        if self.train:
            labels = np.array(self.train_labels)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.train_data = self.train_data[mask]
            self.train_labels = labels[mask].tolist()
        else:
            labels = np.array(self.test_labels)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

            self.test_data = self.test_data[mask]
            self.test_labels = labels[mask].tolist()
