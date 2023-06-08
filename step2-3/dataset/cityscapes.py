import os
import torch.utils.data as torch_data
import numpy as np
from torch import from_numpy
from PIL import Image

from dataset import transform as T

eval_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
map_classes = {
    7: "road",  # 1
    8: "sidewalk",  # 2
    9: "parking",
    10: "rail truck",
    11: "building",  # 3
    12: "wall",  # 4
    13: "fence",  # 5
    14: "guard_rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",  # 6
    18: "pole_group",
    19: "light",  # 7
    20: "sign",  # 8
    21: "vegetation",  # 9
    22: "terrain",  # 10
    23: "sky",  # 11
    24: "person",  # 12
    25: "rider",  # 13
    26: "car",  # 14
    27: "truck",  # 15
    28: "bus",  # 16
    29: "caravan",
    30: "trailer",
    31: "train",  # 17
    32: "motorcycle",  # 18
    33: "bicycle"  # 19
}



# Target refers to labels
DATA_DIR = '/content/drive/MyDrive/DATASET/'
IMAGES_DIR = os.path.join(DATA_DIR, 'cityscapes', 'images')
TARGET_DIR = os.path.join(DATA_DIR, 'cityscapes', 'labels')


class Cityscapes(torch_data.Dataset):
    def __init__(self, data, transform=None, target_transform=None, test_transform=None, cl19=False,
                 test_bisenetv2=False, double=False, quadruple=False, dom_gen=None,
                 split_name='heterogeneous'):
        
        self.images = data
        # refer to client_utils (create clients)
        self.true_len = len(self.images['x'])
        self.transform = transform
        self.test_transform = test_transform
        self.target_transform = target_transform
        self.test_bisenetv2 = test_bisenetv2
        self.double = double
        self.quadruple = quadruple
        self.dom_gen = dom_gen
        self.split_name = split_name

        if cl19 and target_transform is None:
            classes = eval_classes
            mapping = np.zeros((256,), dtype=np.int64) + 255
            for i, cl in enumerate(classes):
                mapping[cl] = i
            self.target_transform = lambda x: from_numpy(mapping[x])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the label of segmentation.
        """
        original_index = index

        if original_index >= self.true_len and (self.double or self.quadruple):
            index %= self.true_len

        img = Image.open(os.path.join(IMAGES_DIR, self.images['x'][index]))
        target = Image.open(os.path.join(TARGET_DIR, self.images['y'][index]))
        
        if (self.double and original_index >= self.true_len) or (
                self.quadruple and original_index >= 2 * self.true_len):
            img, target = T.RandomHorizontalFlip(1)(img, target)

        if (self.double and original_index >= self.true_len) or (
                self.quadruple and original_index >= 2 * self.true_len):
                img, target = T.RandomHorizontalFlip(1)(img, target)
        
        original_img = None

        if self.transform is not None:
            if self.test_bisenetv2:
                original_img = img.copy()
                img = self.test_transform(img)
            else:
                img, target = self.transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.test_bisenetv2:
            return (T.Compose(self.test_transform.transforms[-2:])(original_img), img), target  # jump the resize

        return img, target

    def __len__(self):

        if self.double:
            return 2 * self.true_len
        if self.quadruple:
            return 4 * self.true_len
        return self.true_len
