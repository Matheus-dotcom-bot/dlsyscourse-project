import numpy as np
from .autograd import Tensor
from pathlib import Path
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        return np.flip(img, axis=1) if flip_img else img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        pad = self.padding
        ret = np.pad(img, pad)
        H, W, C = ret.shape
        ret = ret[pad+shift_y:H-(pad-shift_y),pad+shift_x:W-(pad-shift_x),pad:C-pad]
        return ret


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        if self.shuffle:
            ordering = np.arange(len(self.dataset))
            np.random.shuffle(ordering)
            self.ordering = np.array_split(
                ordering,
                range(self.batch_size, len(self.dataset), self.batch_size)
            )
        self.__iterator = iter(self.ordering)
        return self

    def __next__(self):
        return [Tensor(x) for x in self.dataset[next(self.__iterator)]]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms
        self.images = self.parse_images(image_filename)
        self.labels = self.parse_labels(label_filename)

    def __getitem__(self, index) -> object:
        data = self.images[index]
        if isinstance(index, int):
            data = self.apply_transforms(data.reshape(28, 28, 1))
        else:
            data = np.array([self.apply_transforms(image.reshape(28, 28, 1))
                             for image in data])
        labels = self.labels[index]
        return data, labels

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def parse_images(f):
        with gzip.open(f, "rb") as fp:
            _, size, width, height = struct.unpack(">4i", fp.read(16))
            s = width * height
            images = [struct.unpack(f">{s}B", fp.read(s)) for _ in range(size)]
        ret = np.array(images, dtype=np.float32)
        ret = (ret - ret.min()) / (ret.max() - ret.min())  # min-max normalization
        return ret

    @staticmethod
    def parse_labels(f):
        with gzip.open(f, "rb") as fp:
            _, size = struct.unpack(">2i", fp.read(8))
            labels = struct.unpack(f">{size}B", fp.read())
        return np.array(labels, np.uint8)


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        self.transforms = transforms
        data = []
        labels = []
        pattern = "data_batch_*" if train else "test_batch"
        for f in sorted(Path(base_folder).glob(pattern)):
            with open(f, "rb") as b:
                batch = pickle.load(b, encoding="latin1")
                data.extend(batch["data"])
                labels.extend(batch["labels"])

        self.X = np.asarray(data).reshape(-1, 3, 32, 32) / 255
        self.y = np.asarray(labels)

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        return self.apply_transforms(self.X[index]), self.y[index]

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return len(self.y)


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if len(self) == self.word2idx.setdefault(word, len(self)):
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return len(self.idx2word)



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        with open(path, "r") as f:
            return [i for line in f.readlines()[:max_lines and max_lines+1]
                    for i in map(self.dictionary.add_word,
                                 line.replace("\n", "<eos>").split())]


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    nbatch = len(data) // batch_size
    return np.array(data[:nbatch*batch_size], dtype=dtype).reshape(batch_size, nbatch).transpose()


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    nbatch, batch_size = batches.shape
    bptt = min(bptt, nbatch - i - 1)
    data = Tensor(batches[i:i+bptt], device=device, dtype=dtype)
    target = Tensor(batches[i+1:i+bptt+1], device=device, dtype=dtype)
    return data, target.reshape((bptt * batch_size,))
