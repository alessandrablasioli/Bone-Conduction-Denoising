Real Time Speech Enhancement in Time and Frequency domain


Starting from the Facebook's work (link) about denoising, we validate the results of their Demucs model with the Valentini dataset.
Then after validating results, we made data preparation of the Vibravox Dataset. In particular, for our work was interesting to use the Air Conduction Microphone
(nome del mic) and the Bone Conduction Microphone (nome del mic), from all the microphones collected by the vibravox team. Then, we mixed in a random way 
the bone and air conduction clean audio with corrispective noises from the speechless noises in the dataset.

After this work of data building, we studied a function that, starting from Air conduction audio, makes up a bone conduction audio. We have build 4 different
versions of this filter. 
The first one is made by studing in a visual way the Frequency curve of the bc and ac audio. Through Bandpass filters and Low pass filters 
we obtained a good result in terms of listening and frequency curve.
Also did 3 versions starting from the work of VibVoice, one using otsu filtering, as their original work and anothe using triangle filtering and the last one with
the difference between the spectrogram of the bc and the spectrogram of ac audio.

The work than proceeds by making multiple versions of the Demucs network, with the objective of inserting the bone conduction as an input to the network to prove the enhancement 
that it may give to the speech enhancement process.

The first versions sticks to the Demucs model, but adding the bc as an input. So, basically duplicating all the parts of the network to have the noisy file and the bone file as input for
the cleaning of the noisy audio. The first try was given with the bone conduction clean as input with same parameters of original demucs. Note that for this experiment and all the followings 
the train set of vibravox was reduced at 6250 files instead of 20.000 to ease the training process.

Then, several versions were implemented:
- Demucs with 2 inputs but this time the bone conduction was noisy
- Demucs but in frequency domain with only nois as input
- Demucs with 2 inputs with noisy bc as input
- Demucs with 2 inputs but with attention instead of 2 LSTM layers
- Demucs in frequency domain with 2 inputs but with attention instead of 2 LSTM layers


The results showed up that the best version is: 



def start_ddp_workers(cfg):

    import torch as th
    import logging
    log = logging.getLogger(__name__)


    #log = hydra.conf.hydra.job_logging
    rendezvous_file = Path(cfg.rendezvous_file)
    if rendezvous_file.exists():
        rendezvous_file.unlink()

    world_size = th.cuda.device_count()
    if not world_size:
        logger.error(
            "DDP is only available on GPU. Make sure GPUs are properly configured with cuda.")
        sys.exit(1)
    logger.info(f"Starting {world_size} worker processes for DDP.")
    with ChildrenManager() as manager:
        for rank in range(world_size):
            kwargs = {}
            argv = list(sys.argv)
            argv += [f"world_size={world_size}", f"rank={rank}"]
            if rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = sp.DEVNULL
                kwargs['stderr'] = sp.DEVNULL
                log.info(f"Rank: {rank}")
                log_filename = getattr(log, "name", "default.log")  # Recupera il nome del log se disponibile
                argv.append(f"hydra.job_logging.handlers.file.filename={log_filename}")
                #argv.append("hydra.job_logging.handlers.file.filename=" + log)
            manager.add(sp.Popen([sys.executable] + argv, cwd=utils.get_original_cwd(), **kwargs))
    sys.exit(int(manager.failed))