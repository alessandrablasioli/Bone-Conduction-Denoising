#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

# This Version was authored by Alessandra Blasioli

import logging
import os

import hydra

from denoiser.executor import start_ddp_workers

#from torchsummary import summary
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
logger = logging.getLogger(__name__)


def run(args):

    import torch

    from denoiser import distrib
    from denoiser.data import NoisyCleanSet
    from denoiser.demucs import Demucs, DemucsDouble, DemucsFrequencyDualLSTM, DemucsFrequency, DemucsFrequencyAttention 
    from denoiser.solver import Solver
    distrib.init(args)
    torch.manual_seed(args.seed)# torch also initialize cuda seed if available

    '''  uncomment the correct model:

    baselines no bc '''
    #model = Demucs(**args.demucs, sample_rate=args.sample_rate)
    #model = DemucsFrequency(**args.demucsfrequency, sample_rate=args.sample_rate)
    
    '''  baselines with bc '''
    #model = DemucsDouble(**args.demucsdouble, sample_rate=args.sample_rate)
    model = DemucsFrequencyDualLSTM(**args.demucsfrequencyduallstm, sample_rate=args.sample_rate)

    '''  other models '''
    #model = DemucsFrequencyAttention(**args.demucsfrequencyduallstmattn, sample_rate=args.sample_rate)
    #model = DemucsFrequencyBC(**args.demucsfrequencybc, sample_rate=args.sample_rate)
    
    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return
    print(args.batch_size,distrib.world_size)
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)

    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}

    # Building datasets and loaders

    # edit bone_conduction flag if there are bc inputs
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, **kwargs, bone_conduction=True)
    #print('check 1:', tr_dataset)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:
        cv_dataset = NoisyCleanSet(args.dset.valid, **kwargs, bone_conduction=True)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:
        tt_dataset = NoisyCleanSet(args.dset.test, **kwargs, bone_conduction=True)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
        
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}
    
    if torch.cuda.is_available():
        model.cuda()
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, model, optimizer, args)
    solver.train()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)


@hydra.main(config_path="conf", config_name="config")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)


if __name__ == "__main__":
    main()
