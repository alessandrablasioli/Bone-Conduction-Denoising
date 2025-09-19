# This version was authored by Alessandra Blasioli
import json
import logging
import os
import re

from .audio_filter_bc import Audioset # For Synth BC
#from .audio import Audioset
logger = logging.getLogger(__name__)

def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class NoisyCleanSet:
    def __init__(self, json_dir, dataset=None, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, bone_conduction=False):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param bone_conduction: if the network is using the bone conduction audio it must be set to True
        """
        self.dataset = dataset
        noisy_json = os.path.join(json_dir, 'noisy.json')
        
        clean_json = os.path.join(json_dir, 'clean.json')
        
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
            
        with open(clean_json, 'r') as f:
            clean = json.load(f)
        match_files(noisy, clean, matching)
       
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.noisy_set = Audioset(noisy, dataset, **kw)  
        self.clean_set = Audioset(clean, dataset, **kw)

        #for synthetic bc
        
        self.bone_set = Audioset(noisy, dataset, **kw, bone=True)
        assert len(self.clean_set) == len(self.bone_set)
        
        # if using bc audio
        '''
        if bone_conduction:
                bone_json = os.path.join(json_dir, 'bone.json')
                with open(bone_json, 'r') as f:
                    bone = json.load(f)
                match_files(bone, clean, matching)
                self.bone_set = Audioset(bone, dataset, **kw)
                assert len(self.clean_set) == len(self.bone_set)
        '''
        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index], self.bone_set[index]#comment self.bone_set[index] if no bc audio are used 
    
    def __len__(self):
        return len(self.noisy_set)
