"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""

import os
import tqdm
import pickle
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
import PIL


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self, folder, cache, min_len=32):
        dataset = ImageFolder(folder)
        self.total_frames = 0
        self.lengths = []
        self.images = []

        if cache is not None and os.path.exists(cache):
            with open(cache, 'r') as f:
                self.images, self.lengths = pickle.load(f)
        else:
            for idx, (im, categ) in enumerate(
                    tqdm.tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                shorter, longer = min(im.width, im.height), max(im.width, im.height)
                length = longer // shorter
                if length >= min_len:
                    self.images.append((img_path, categ))
                    self.lengths.append(length)

            if cache is not None:
                with open(cache, 'w') as f:
                    pickle.dump((self.images, self.lengths), f)

        self.cumsum = np.cumsum([0] + self.lengths)
        print "Total number of frames {}".format(np.sum(self.lengths))

    def __getitem__(self, item):
        path, label = self.images[item]
        im = PIL.Image.open(path)
        return im, label

    def __len__(self):
        return len(self.images)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset

        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, target = self.dataset[video_id]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]

        if horizontal:
            i_from, i_to = video.shape[0] * frame_num, video.shape[0] * (frame_num + 1)
            frame = video[:, i_from: i_to, ::]
        else:
            i_from, i_to = video.shape[1] * frame_num, video.shape[1] * (frame_num + 1)
            frame = video[i_from: i_to, :, ::]

        if frame.shape[0] == 0:
            print "video {}. From {} to {}. num {}".format(video.shape, i_from, i_to, item)

        return {"images": self.transforms(frame), "categories": target}

    def __len__(self):
        return self.dataset.cumsum[-1]


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth
        self.transforms = transform if transform is not None else lambda x: x

    def __getitem__(self, item):
        video, target = self.dataset[item]
        video = np.array(video)

        horizontal = video.shape[1] > video.shape[0]
        shorter, longer = min(video.shape[0], video.shape[1]), max(video.shape[0], video.shape[1])
        video_len = longer // shorter

        # videos can be of various length, we randomly sample sub-sequences
        if video_len > self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif video_len >= self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            raise Exception("Length is too short id - {}, len - {}").format(self.dataset[item], video_len)

        frames = np.split(video, video_len, axis=1 if horizontal else 0)
        selected = np.array([frames[s_id] for s_id in subsequence_idx])

        return {"images": self.transforms(selected), "categories": target}

    def __len__(self):
        return len(self.dataset)


class ImageSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transforms = transform

    def __getitem__(self, index):
        result = {}
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], index, axis=0)

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result

    def __len__(self):
        return self.dataset.get_data()[self.dataset.keys[0]].shape[0]


class VideoSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, video_length, every_nth=1, transform=None):
        self.dataset = dataset
        self.video_length = video_length
        self.unique_ids = np.unique(self.dataset.get_data()['video_ids'])
        self.every_nth = every_nth
        self.transforms = transform

    def __getitem__(self, item):
        result = {}
        ids = self.dataset.get_data()['video_ids'] == self.unique_ids[item]
        ids = np.squeeze(np.squeeze(np.argwhere(ids)))
        for k in self.dataset.keys:
            result[k] = np.take(self.dataset.get_data()[k], ids, axis=0)

        subsequence_idx = None
        print result[k].shape[0]

        # videos can be of various length, we randomly sample sub-sequences
        if result[k].shape[0] > self.video_length:
            needed = self.every_nth * (self.video_length - 1)
            gap = result[k].shape[0] - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(start, start + needed, self.video_length, endpoint=True, dtype=np.int32)
        elif result[k].shape[0] == self.video_length:
            subsequence_idx = np.arange(0, self.video_length)
        else:
            print "Length is too short id - {}, len - {}".format(self.unique_ids[item], result[k].shape[0])

        if subsequence_idx:
            for k in self.dataset.keys:
                result[k] = np.take(result[k], subsequence_idx, axis=0)
        else:
            print result[self.dataset.keys[0]].shape

        if self.transforms is not None:
            for k, transform in self.transforms.iteritems():
                result[k] = transform(result[k])

        return result

    def __len__(self):
        return len(self.unique_ids)