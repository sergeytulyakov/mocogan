"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).

Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, MoCoGAN: Decomposing Motion and Content for Video Generation
https://arxiv.org/abs/1707.04993

Usage:
    train.py [options] <dataset> <log_folder>

Options:
    --image_dataset=<path>          specifies a separate dataset to train for images [default: ]
    --image_batch=<count>           number of images in image batch [default: 10]
    --video_batch=<count>           number of videos in video batch [default: 3]

    --image_size=<int>              resize all frames to this size [default: 64]

    --use_infogan                   when specified infogan loss is used

    --use_categories                when specified ground truth categories are used to
                                    train CategoricalVideoDiscriminator

    --use_noise                     when specified instance noise is used
    --noise_sigma=<float>           when use_noise is specified, noise_sigma controls
                                    the magnitude of the noise [default: 0]

    --image_discriminator=<type>    specifies image disciminator type (see models.py for a
                                    list of available models) [default: PatchImageDiscriminator]

    --video_discriminator=<type>    specifies video discriminator type (see models.py for a
                                    list of available models) [default: CategoricalVideoDiscriminator]

    --video_length=<len>            length of the video [default: 16]
    --print_every=<count>           print every iterations [default: 1]
    --n_channels=<count>            number of channels in the input data [default: 3]
    --every_nth=<count>             sample training videos using every nth frame [default: 4]
    --batches=<count>               specify number of batches to train [default: 100000]

    --dim_z_content=<count>         dimensionality of the content input, ie hidden space [default: 50]
    --dim_z_motion=<count>          dimensionality of the motion input [default: 10]
    --dim_z_category=<count>        dimensionality of categorical input [default: 6]
"""

import os
import docopt
import PIL

import functools

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models

from trainers import Trainer

import data


def build_discriminator(type, **kwargs):
    discriminator_type = getattr(models, type)

    if 'Categorical' not in type and 'dim_categorical' in kwargs:
        kwargs.pop('dim_categorical')

    return discriminator_type(**kwargs)


def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    print(args)

    n_channels = int(args['--n_channels'])

    image_transforms = transforms.Compose([
        PIL.Image.fromarray,
        transforms.Scale(int(args["--image_size"])),
        transforms.ToTensor(),
        lambda x: x[:n_channels, ::],
        transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
    ])

    video_transforms = functools.partial(video_transform, image_transform=image_transforms)

    video_length = int(args['--video_length'])
    image_batch = int(args['--image_batch'])
    video_batch = int(args['--video_batch'])

    dim_z_content = int(args['--dim_z_content'])
    dim_z_motion = int(args['--dim_z_motion'])
    dim_z_category = int(args['--dim_z_category'])

    dataset = data.VideoFolderDataset(args['<dataset>'], cache=os.path.join(args['<dataset>'], 'local.db'))
    image_dataset = data.ImageDataset(dataset, image_transforms)
    image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=2, shuffle=True)

    video_dataset = data.VideoDataset(dataset, 16, 2, video_transforms)
    video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=2, shuffle=True)

    generator = models.VideoGenerator(n_channels, dim_z_content, dim_z_category, dim_z_motion, video_length)

    image_discriminator = build_discriminator(args['--image_discriminator'], n_channels=n_channels,
                                              use_noise=args['--use_noise'], noise_sigma=float(args['--noise_sigma']))

    video_discriminator = build_discriminator(args['--video_discriminator'], dim_categorical=dim_z_category,
                                              n_channels=n_channels, use_noise=args['--use_noise'],
                                              noise_sigma=float(args['--noise_sigma']))

    if torch.cuda.is_available():
        generator.cuda()
        image_discriminator.cuda()
        video_discriminator.cuda()

    trainer = Trainer(image_loader, video_loader,
                      int(args['--print_every']),
                      int(args['--batches']),
                      args['<log_folder>'],
                      use_cuda=torch.cuda.is_available(),
                      use_infogan=args['--use_infogan'],
                      use_categories=args['--use_categories'])

    trainer.train(generator, image_discriminator, video_discriminator)
