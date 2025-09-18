# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math
import time
import torch as th
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn.functional as F
from scipy.signal.windows import get_window

from .resample import downsample2, upsample2
from .utils import capture_init


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    if invers :
        kernel = np.linalg.pinv(kernel).T
    kernel = kernel*window
    kernel = kernel[:, None, :]
    return th.from_numpy(kernel.astype(np.float32)), th.from_numpy(window[None,:,None].astype(np.float32))

'''
Convolutional layer with STFT calculation. Authored by Manuele Rusci (GWT)
'''
class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = th.unsqueeze(inputs, 1)
        inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)
         
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim//2 +1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = th.sqrt(real**2+imag**2) 
            phase = th.atan2(imag, real)
            return mags, phase
        
'''
Convolutional layer with iSTFT calculation. Authored by Manuele Rusci (GWT)
'''
class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', th.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 

        if phase is not None:
            real = inputs*th.cos(phase)
            imag = inputs*th.sin(phase)
            inputs = th.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 

        # this is from th-stft: https://github.com/pseeth/th-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs/(coff+1e-8)
        #outputs = th.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        
        return outputs

'''
Demucs speech enhancement model:
        - one input model
        - time domain
        - the original one from 
                                    Copyright (c) Facebook, Inc. and its affiliates.
                                    All rights reserved.
                                    This source code is licensed under the license found in the
                                    LICENSE file in the root directory of this source tree.
                                    author: adefossez
'''

class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.

    """
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)

        x = x[..., :length]
        return std * x


# All the following variants of the Demucs architecture are authored by Alessandra Blasioli

'''
Demucs in frequency domain
'''

class DemucsFrequency(nn.Module):
    """
    Demucs speech enhancement model in frequency domain.
    
    - hidden: Number of units in the hidden layers, default is 48.
    - depth: Number of layers or the depth of the model, default is 5.
    - kernel_size: Size of the convolutional kernel, default is 1.
    - stride: Stride of the convolution operation, default is 4.
    - causal: If True, applies causal convolutions, default is True.
    - fft_size: Size of the FFT (Fast Fourier Transform), default is 512.
    - normalize: If True, normalizes the input, default is True.
    - floor: Minimum floor value for numerical stability, default is 1e-3.
    - win_len: Length of the window used for STFT analysis, default is 400.
    - win_inc: Step size (hop length) between consecutive windows, default is 100.
    - rnn_units: Number of units in the RNN layers, default is 257.
    - win_type: Type of window function to use, e.g., 'hamming', default is 'hamming'.
    - sample_rate: Sampling rate of the audio signal, default is 16,000 (16 kHz).
    """
    @capture_init
    def __init__(self,
                 hidden=48,
                 depth=5,
                 kernel_size=1,
                 stride=4,
                 causal=True,
                 fft_size=512,
                 normalize=True,
                 floor=1e-3,
                 win_len=400,
                 win_inc=100,
                 rnn_units=257,
                 win_type='hamming',
                 sample_rate=16_000):
        
        super().__init__()

        #parameters
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.fft_size = fft_size
        self.normalize = normalize
        self.floor = floor
        self.hop_length = math.floor(fft_size / 4)
        self.resample = self.hop_length / fft_size
        self.win_len = win_len
        self.win_inc = win_inc 
        self.win_type = win_type
        self.rnn_units = rnn_units
        self.rnn_input_size = (fft_size // 2 ) + 1
        self.sample_rate = sample_rate


        #layers
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        self.lstm = BLSTM(self.rnn_input_size, bi=not causal)
        self.conv = nn.Conv1d(self.rnn_units , ((fft_size // 2 ) + 1), 1)
        self.norm = nn.BatchNorm1d((fft_size // 2 ) + 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        #self.fc = nn.Linear(((fft_size // 2 )  + 1), ((fft_size // 2 )  + 1))
        #self.fc_after_lstm = nn.Linear(((fft_size // 2 )  + 1), ((fft_size // 2 )  + 1))

    def forward(self, mix1):
        length = mix1.size(2)
        #pad
        mag, phase = self.stft(mix1)
        mask = mag
        mask = mask.permute(2, 0, 1)
        #mask = self.fc(mask)
        mask, _ = self.lstm(mask)
        #mask = self.fc_after_lstm(mask)
        mask = mask.permute(1, 2, 0)
        mask = self.conv(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv(mask)
        mask = self.sig(mask)
        estimated = mag * mask
        out = self.istft(estimated, phase=phase)
        szl = out.size(2)
        output = F.pad(out, (0,length-szl),"constant", 0 )
        output = output[..., :length]
        return output


'''

Double input and double LSTM demucs
    - time domain: DemucsDouble
    - frequency domain: DemucsFrequencyDualLSTM

class DemucsDouble(nn.Module):
    
    
    The class Demucs Double simply implements a Demucs with 2 encoders at the beginning, since what 
    we want to do is to give two inputs to the network, an air conduction audio and a bone conductione one.

    We combine the input before the lstm. 



    encoder1 ---
                  concat ----- (lstm , lstm) ---- decoder
                                  
    encoder2 ---
    
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 #resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super(DemucsDouble, self).__init__()
       # if resample not in [1, 2, 4]:
            #raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        #self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        # for air conduction audio
        self.encoder1 = nn.ModuleList()
        self.decoder = nn.ModuleList()


        # for bone conduction audio
        self.encoder2 = nn.ModuleList()



        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder1.append(nn.Sequential(*encode))
            self.encoder2.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        #self.fc = nn.Linear(768, 768)
        
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length )#* self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length)) #/ self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth #// self.resample

    def forward(self, mix1 ,mix2):
        if mix1.dim() == 2:
            mix1 = mix1.unsqueeze(1)
        if mix2.dim() == 2:
                    mix2 = mix2.unsqueeze(1)


        if self.normalize:
            mono1 = mix1.mean(dim=1, keepdim=True)
            std1 = mono1.std(dim=-1, keepdim=True)
            mix1 = mix1 / (self.floor + std1)

            mono2 = mix2.mean(dim=1, keepdim=True)
            std2 = mono2.std(dim=-1, keepdim=True)
            mix2 = mix2 / (self.floor + std2)
        else:
            std1 = std2 = 1



        length = mix1.shape[-1]
        x1 = mix1
        x2 = mix2
        
        x1 = F.pad(x1, (0, self.valid_length(length) - length))
        x2 = F.pad(x2, (0, self.valid_length(length) - length))


        #if self.resample == 2:
            #x1 = upsample2(x1)
            #x2 = upsample2(x2)
        #elif self.resample == 4:
            #x1 = upsample2(x1)
            #x1 = upsample2(x1)
            #x2 = upsample2(x2)
            #x2 = upsample2(x2)
            
        skips1 = []
        for encode in self.encoder1:
            x1 = encode(x1)
            skips1.append(x1)

        skips2 = []
        for encode in self.encoder2:
            x2 = encode(x2)
            skips2.append(x2)
        #print(x1.shape, x2.shape)
        combined = th.cat((x1, x2), dim=2)
        #skip_fc = []
        #put a dense layer here and use it as skip connection 
        #combined = self.fc(combined.permute(0, 2, 1))
        #skip_fc = combined
        combined = combined.permute(2, 0, 1)
        
        combined, _ = self.lstm(combined)
        combined = combined.permute(1, 2, 0)


        #another interesting thing: remove skip of ac and keep only bc skip(skip2)
        
        for decode in self.decoder:
            skip1= skips1.pop(-1)
            skip2 =skips2.pop(-1)
            #skip = th.cat((skip1, skip2), dim=2) 
            #skip = skip_fc.pop(-1)
            
            #skip2 = F.interpolate(skip2, size=combined.size(2), mode='linear')
            skip2 = F.pad(skip2, (0, combined.size(2) - skip2.size(2)))
            combined = combined #+ skip2
            combined = decode(combined)
        #if self.resample == 2:
            #combined = downsample2(combined)
        #elif self.resample == 4:
            #combined = downsample2(combined)
            #combined = downsample2(combined)

        combined = combined[..., :length]
        #print(combined.shape)
        return combined * std1

'''



class DemucsDouble(nn.Module):
    
    '''
    The class Demucs Double simply implements a Demucs with 2 encoders at the beginning, since what 
    we want to do is to give two inputs to the network, an air conduction audio and a bone conductione one.

    We combine the input before the lstm. 



    encoder1 ---
                  concat ----- (lstm , lstm) ---- decoder
                                  
    encoder2 ---
    
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    '''
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super(DemucsDouble, self).__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        # for air conduction audio
        self.encoder1 = nn.ModuleList()
        self.decoder = nn.ModuleList()


        # for bone conduction audio
        self.encoder2 = nn.ModuleList()



        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder1.append(nn.Sequential(*encode))
            self.encoder2.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        #self.fc = nn.Linear(768, 768)
        
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length/ self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth// self.resample

    def forward(self, mix1 ,mix2):
        if mix1.dim() == 2:
            mix1 = mix1.unsqueeze(1)
        if mix2.dim() == 2:
                    mix2 = mix2.unsqueeze(1)


        if self.normalize:
            mono1 = mix1.mean(dim=1, keepdim=True)
            std1 = mono1.std(dim=-1, keepdim=True)
            mix1 = mix1 / (self.floor + std1)

            mono2 = mix2.mean(dim=1, keepdim=True)
            std2 = mono2.std(dim=-1, keepdim=True)
            mix2 = mix2 / (self.floor + std2)
        else:
            std1 = std2 = 1



        length = mix1.shape[-1]
        x1 = mix1
        x2 = mix2
        
        x1 = F.pad(x1, (0, self.valid_length(length) - length))
        x2 = F.pad(x2, (0, self.valid_length(length) - length))


        if self.resample == 2:
            x1 = upsample2(x1)
            x2 = upsample2(x2)
        elif self.resample == 4:
            x1 = upsample2(x1)
            x1 = upsample2(x1)
            x2 = upsample2(x2)
            x2 = upsample2(x2)
            
        skips1 = []
        for encode in self.encoder1:
            x1 = encode(x1)
            skips1.append(x1)

        skips2 = []
        for encode in self.encoder2:
            x2 = encode(x2)
            skips2.append(x2)
        #print(x1.shape, x2.shape)
        combined = th.cat((x1, x2), dim=2)
        #skip_fc = []
        #put a dense layer here and use it as skip connection 
        #combined = self.fc(combined.permute(0, 2, 1))
        #skip_fc = combined
        combined = combined.permute(2, 0, 1)
        
        combined, _ = self.lstm(combined)
        combined = combined.permute(1, 2, 0)


        #another interesting thing: remove skip of ac and keep only bc skip(skip2)
        
        for decode in self.decoder:
            skip1= skips1.pop(-1)
            skip2 =skips2.pop(-1)
            skip = th.cat((skip1, skip2), dim=2) 
            #skip = skip_fc.pop(-1)
            
            #skip2 = F.interpolate(skip2, size=combined.size(2), mode='linear')
            skip2 = F.pad(skip2, (0, combined.size(2) - skip2.size(2)))
            combined = skip
            combined = decode(combined)
        if self.resample == 2:
            combined = downsample2(combined)
        elif self.resample == 4:
            combined = downsample2(combined)
            combined = downsample2(combined)

        combined = combined[..., :length]
        #print(combined.shape)
        return combined * std1


class DemucsFrequencyDualLSTM(nn.Module):
    """
    
    The class DemucsFrequencyDualLSTM implements a frequency domain denoiser inspired by Demucs model
    with 2 inputs, the ac audio and the bc audio.

    - hidden: Number of units in the hidden layers, default is 48.
    - depth: Number of layers or the depth of the model, default is 5.
    - kernel_size: Size of the convolutional kernel, default is 1.
    - stride: Stride of the convolution operation, default is 4.
    - causal: If True, applies causal convolutions, default is True.
    - fft_size: Size of the FFT (Fast Fourier Transform), default is 512.
    - normalize: If True, normalizes the input, default is True.
    - floor: Minimum floor value for numerical stability, default is 1e-3.
    - win_len: Length of the window used for STFT analysis, default is 400.
    - win_inc: Step size (hop length) between consecutive windows, default is 100.
    - rnn_units: Number of units in the RNN layers, default is 257.
    - win_type: Type of window function to use, e.g., 'hamming', default is 'hamming'.
    - sample_rate: Sampling rate of the audio signal, default is 16,000 (16 kHz).
    """
    @capture_init
    def __init__(self,
                 hidden=48,
                 depth=5,
                 kernel_size=1,
                 stride=4,
                 causal=True,
                 fft_size=512,
                 normalize=True,
                 floor=1e-3,
                 win_len=400,
                 win_inc=100,
                 rnn_units=257,
                 win_type='hamming',
                 sample_rate=16_000):
        super().__init__()

        #parameters
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.fft_size = fft_size
        self.normalize = normalize
        self.floor = floor
        self.hop_length = math.floor(fft_size / 4)
        self.resample = self.hop_length / fft_size
        self.win_len = win_len
        self.win_inc = win_inc 
        self.win_type = win_type
        self.rnn_units = rnn_units
        self.rnn_input_size = (fft_size // 2 ) + 1
        self.sample_rate = sample_rate


        #layers
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        self.lstm = BLSTM(self.rnn_input_size, bi=not causal)
        self.conv1 = nn.Conv1d(self.rnn_units , ((fft_size // 2 )  + 1), 1)
        self.conv2 = nn.Conv1d(self.rnn_units , ((fft_size // 2 )  + 1), 1)
        self.norm = nn.BatchNorm1d((fft_size // 2 )  + 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        self.fc_mix1 = nn.Linear(((fft_size // 2 )  + 1), ((fft_size // 2 )  + 1))
        self.fc_mix2 = nn.Linear(((fft_size // 2 )  + 1), ((fft_size // 2 )  + 1))
        self.fc_after_lstm = nn.Linear(((fft_size // 2 )  + 1), ((fft_size // 2 )  + 1))

    
    def forward(self, mix1, mix2):
        alpha = 3.0
        length = mix1.size(2)
        mag1, phase1 = self.stft(mix1)
        mag2, phase2 = self.stft(mix2)
        m = F.pad(mag1, (0, mag2.size(2) - mag1.size(2)))
        mag1_transformed = self.fc_mix1(m.permute(2, 0, 1))  # permute in (freq, batch, time_frame)
        mag2_transformed = self.fc_mix2(mag2.permute(2, 0, 1))
        mags = th.cat((mag1_transformed, mag2_transformed*alpha), dim=0)  # Concatenate features in dim 0 
       
        #mags = (mag1_transformed * mag2_transformed)
        mask, _ = self.lstm(mags)
        mask = self.fc_after_lstm(mask)
        mask = mask.permute(1, 2, 0)  # permute in (batch, channels, freq)
        mask = self.conv1(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv2(mask)
        mask = self.sig(mask)
        mask = mask[:, :, :mag1.size(2)]
        estimated = mag1 * mask
        out = self.istft(estimated, phase=phase1)
        szl = out.size(2)
        output = F.pad(out, (0, length - szl), "constant", 0)
        output = output[..., :length]

        return output
    
''' original version commented bc of ongoing tests
the very first version is withouth the fc layers
def forward(self, mix1, mix2):
        length = mix1.size(2)
        mag1, phase1 = self.stft(mix1)
        mag2, phase2 = self.stft(mix2)
        mag1_transformed = self.fc_mix1(mag1.permute(2, 0, 1))  # permute in (freq, batch, time_frame)
        mag2_transformed = self.fc_mix2(mag2.permute(2, 0, 1))
        mags = th.cat((mag1_transformed, mag2_transformed), dim=0)  # Concatenate features in dim 0 
        mask, _ = self.lstm(mags)
        mask = self.fc_after_lstm(mask)
        mask = mask.permute(1, 2, 0)  # permute in (batch, channels, freq)
        mask = self.conv1(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv2(mask)
        mask = self.sig(mask)
        mask = mask[:, :, :mag1.size(2)]
        estimated = mag1 * mask
        out = self.istft(estimated, phase=phase1)
        szl = out.size(2)
        output = F.pad(out, (0, length - szl), "constant", 0)
        output = output[..., :length]

        return output
'''

'''
Double input and MH Attention demucs
    - time domain: DemucsDoubleAttention
    - frequency domain: DemucsFrequencyAttention

'''

# STILL TO FINISH AND TEST
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = th.zeros(max_len, d_model)
            position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
            div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = th.sin(position * div_term)
            pe[:, 1::2] = th.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].to('cuda')
        return self.dropout(x).to('cuda')
    
class DemucsFrequencyAttention(nn.Module):
    """
    
    """
    @capture_init
    def __init__(self,
                 hidden=48,
                 depth=5,
                 kernel_size=1,
                 stride=4,
                 causal=True,
                 fft_size=510,
                 normalize=True,
                 floor=1e-3,
                 win_len=400,
                 win_inc=100,
                 rnn_units=256,
                 win_type='hamming',
                 sample_rate=16_000,
                 dropout=0.1,
                 num_heads=2):
        super().__init__()

        #parameters
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.fft_size = fft_size
        self.normalize = normalize
        self.floor = floor
        self.num_heads = num_heads
        self.dropout = dropout
        self.hop_length = math.floor(fft_size / 4)
        self.resample = self.hop_length / fft_size
        self.win_len = win_len
        self.win_inc = win_inc 
        self.win_type = win_type
        self.rnn_units = rnn_units
        self.rnn_input_size = (fft_size // 2 ) + 1
        self.sample_rate = sample_rate



        #layers

        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        self.conv = nn.Conv1d(self.rnn_units , ((fft_size // 2 ) + 1), 1)
        self.norm = nn.BatchNorm1d((fft_size // 2 ) + 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')

    def create_custom_mask(self, seq_len, a, b):
        causal_mask = th.tril(th.ones(seq_len, seq_len))
        attention_bias = th.zeros(seq_len, seq_len)
        for i in range(seq_len):
                attention_bias[i, a :b] += 2.0 
        custom_mask = causal_mask + attention_bias
        custom_mask[causal_mask == 0] = float('-inf')
        return custom_mask.to('cuda')
    
    def forward(self, mix1, mix2):
        length = mix1.size(2)
        mag1, phase1 = self.stft(mix1)  
        mag2, phase2 = self.stft(mix2) 
        mags = th.cat((mag1, mag2), dim=2)
        mask = mags.permute(2, 0, 1)


        a = mag1.shape[2] + 1
        b = mag2.shape[2]
        attn_mask = self.create_custom_mask(mask.shape[0], a, b)
        #attn_mask =  torch.tril(torch.ones(mask.shape[0], mask.shape[0])).unsqueeze(0).unsqueeze(0)
        #attn_mask = torch.tril(torch.ones(sequence_length, sequence_length))  # (seq_len, seq_len)

        # Expand it to work across the batch
        #attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
        #attn_mask = attn_mask.expand(batch_size, -1, -1, -1)
        dim_attn = mask.shape[-1] 
        self.attention = th.nn.MultiheadAttention(dim_attn, num_heads=self.num_heads, dropout=self.dropout,bias=True, add_zero_attn=False, kdim=None, vdim=None, batch_first=False,device='cuda', dtype=None)
        mask, _ = self.attention(mask, mask, mask, is_causal=True, attn_mask = th.nn.Transformer.generate_square_subsequent_mask(mask.shape[0]).to('cuda'))
        
        mask = mask.permute(1, 2, 0) 
        #print(mask.shape)
        mask = self.conv(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv(mask)
        mask = self.sig(mask)
        #print(mask.shape)
        mask = mask[:, :, :mag1.size(2)]
        estimated = mag1 * mask
        out = self.istft(estimated, phase=phase1)
        szl = out.size(2)
        output = F.pad(out, (0, length - szl), "constant", 0)
        output = output[..., :length]
        #print(output.shape)
        return output

    
    
'''
def forward(self, mix1, mix2):
        length = mix1.size(2)
        #pad
        mag1, phase1 = self.stft(mix1)
        #print(mag1.shape)
        mag2, phase2 = self.stft(mix2)
        phase_size = phase1.size()
        #print(phase_size)
        a = mag1.shape[2] + 1
        b = mag2.shape[2]
        

        mask = th.cat((mag1,mag2), dim=2)
        _, max_len, d_model = mask.shape
        #print(mag1.shape, mag2.shape, mask.shape)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        mask = self.pe(mask)
        #print(mask.shape)
        #print(attn_mask, attn_mask.shape)
        
        mask = mask.permute(2, 0, 1)
        attn_mask = self.create_custom_mask(mask.shape[0], a, b)
        dim_attn = mask.shape[-1] 
        self.attention = th.nn.MultiheadAttention(dim_attn, num_heads=self.num_heads, dropout=self.dropout,bias=True, add_zero_attn=False, kdim=None, vdim=None, batch_first=False,device='cuda', dtype=None)
        mask, _ = self.attention(mask, mask, mask, is_causal=True, attn_mask = attn_mask)# th.nn.Transformer.generate_square_subsequent_mask(mask.shape[0]).to('cuda'))
        mask = mask.permute(1, 2, 0) 
        mask = self.conv(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv(mask)
        mask = self.sig(mask)
        #print(mask.shape)
        mask = mask[:, :, :mag1.size(2)]
        estimated = mag1 * mask
        out = self.istft(estimated, phase=phase1)
        szl = out.size(2)
        output = F.pad(out, (0,length-szl),"constant", 0 )
        output = output[..., :length]
        #print(output.shape)
        return output
    
'''

    
class DemucsDoubleAttention(nn.Module):
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000,
                 dropout=0.1,
                 num_heads=4):

        super(DemucsDoubleAttention, self).__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        # for air conduction audio
        self.encoder1 = nn.ModuleList()
        self.decoder = nn.ModuleList()


        # for bone conduction audio
        self.encoder2 = nn.ModuleList()



        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder1.append(nn.Sequential(*encode))
            self.encoder2.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)
            
       
        self.attention = th.nn.MultiheadAttention(hidden, num_heads=self.num_heads, dropout=self.dropout,bias=True, add_zero_attn=False, kdim=None, vdim=None, batch_first=False,device='cuda', dtype=None)
        self.fc_mix1 = nn.Linear(hidden,hidden)
        self.fc_mix2 = nn.Linear(hidden,hidden)
        self.fc = nn.Linear(hidden,hidden)

        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix1 ,mix2):
        if mix1.dim() == 2:
            mix1 = mix1.unsqueeze(1)
        if mix2.dim() == 2:
                    mix2 = mix2.unsqueeze(1)


        if self.normalize:
            mono1 = mix1.mean(dim=1, keepdim=True)
            std1 = mono1.std(dim=-1, keepdim=True)
            mix1 = mix1 / (self.floor + std1)

            mono2 = mix2.mean(dim=1, keepdim=True)
            std2 = mono2.std(dim=-1, keepdim=True)
            mix2 = mix2 / (self.floor + std2)
        else:
            std1 = std2 = 1



        length = mix1.shape[-1]
        x1 = mix1
        x2 = mix2
        
        x1 = F.pad(x1, (0, self.valid_length(length) - length))
        x2 = F.pad(x2, (0, self.valid_length(length) - length))


        if self.resample == 2:
            x1 = upsample2(x1)
            x2 = upsample2(x2)
        elif self.resample == 4:
            x1 = upsample2(x1)
            x1 = upsample2(x1)
            x2 = upsample2(x2)
            x2 = upsample2(x2)
            
        skips1 = []
        for encode in self.encoder1:
            x1 = encode(x1)
            skips1.append(x1)

        skips2 = []
        for encode in self.encoder2:
            x2 = encode(x2)
            skips2.append(x2)
        x1 = self.fc_mix1(x1)
        x2 = self.fc_mix2(x2)
        combined = th.cat((x1, x2), dim=2)
        combined = combined.permute(2, 0, 1)
        combined, _ = self.attention(combined, combined,combined, is_causal=True, attn_mask = th.nn.Transformer.generate_square_subsequent_mask(combined.shape[0]).to('cuda'))

        combined = self.fc(combined)
        combined = combined.permute(1, 2, 0)
        for decode in self.decoder:
            skip1= skips1.pop(-1)
            skip2 =skips2.pop(-1)
            skip = th.cat((skip1, skip2), dim=2) 
            combined = combined + skip[..., :combined.shape[-1]]
            combined = decode(combined)
        if self.resample == 2:
            combined = downsample2(combined)
        elif self.resample == 4:
            combined = downsample2(combined)
            combined = downsample2(combined)

        combined = combined[..., :length]
        print((combined*std1).shape)
        return combined * std1



# streaming version by adefossez
def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)   
 
class DemucsStreamer:
    """
    Streaming implementation for Demucs. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - demucs (Demucs): Demucs model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """
    def __init__(self, demucs,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256):
        device = next(iter(demucs.parameters())).device
        self.demucs = demucs
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(demucs.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = demucs.valid_length(1) + demucs.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = demucs.total_stride * num_frames
        self.resample_in = th.zeros(demucs.chin, resample_buffer, device=device)
        self.resample_out = th.zeros(demucs.chin, resample_buffer, device=device)

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = th.zeros(demucs.chin, 0, device=device)

        bias = demucs.decoder[0][2].bias
        weight = demucs.decoder[0][2].weight
        chin, chout, kernel = weight.shape
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        self._weight = weight.permute(1, 2, 0).contiguous()

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.lstm_state = None
        self.conv_state = None
        pending_length = self.pending.shape[1]
        padding = th.zeros(self.demucs.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        demucs = self.demucs
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = demucs.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != demucs.chin:
            raise ValueError(f"Expected {demucs.chin} channels, got {chin}")

        self.pending = th.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if demucs.normalize:
                mono = frame.mean(0)
                variance = (mono**2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (demucs.floor + math.sqrt(self.variance))
            padded_frame = th.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, stride - resample_buffer:stride]
            frame = padded_frame

            if resample == 4:
                frame = upsample2(upsample2(frame))
            elif resample == 2:
                frame = upsample2(frame)
            frame = frame[:, resample * resample_buffer:]  # remove pre sampling buffer
            frame = frame[:, :resample * self.frame_length]  # remove extra samples after window

            out, extra = self._separate_frame(frame)
            padded_out = th.cat([self.resample_out, out, extra], 1)
            self.resample_out[:] = out[:, -resample_buffer:]
            if resample == 4:
                out = downsample2(downsample2(padded_out))
            elif resample == 2:
                out = downsample2(padded_out)
            else:
                out = padded_out

            out = out[:, resample_buffer // resample:]
            out = out[:, :stride]

            if demucs.normalize:
                out *= math.sqrt(self.variance)
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, stride:]

        self.total_time += time.time() - begin
        if outs:
            out = th.cat(outs, 1)
        else:
            out = th.zeros(chin, 0, device=wav.device)
        return out

    def _separate_frame(self, frame):
        demucs = self.demucs
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * demucs.resample
        x = frame[None]
        for idx, encode in enumerate(demucs.encoder):
            stride //= demucs.stride
            length = x.shape[2]
            if idx == demucs.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - demucs.kernel_size) // demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - demucs.kernel_size - demucs.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = th.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = demucs.lstm(x, self.lstm_state)
        x = x.permute(1, 2, 0)
        # In the following, x contains only correct samples, i.e. the one
        # for which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra = None
        for idx, decode in enumerate(demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -demucs.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -demucs.stride:]
            else:
                extra[..., :demucs.stride] += next_state[-1]
            x = x[..., :-demucs.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :demucs.stride] += prev
            if idx != demucs.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]


#modify test function
def test():
    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.demucs",
        description="Benchmark the streaming Demucs implementation, "
                    "as well as checking the delta with the offline implementation.")
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--resample", default=4, type=int)
    parser.add_argument("--hidden", default=48, type=int)
    parser.add_argument("--sample_rate", default=16000, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-t", "--num_threads", type=int)
    parser.add_argument("-f", "--num_frames", type=int, default=1)
    args = parser.parse_args()
    if args.num_threads:
        th.set_num_threads(args.num_threads)
    sr = args.sample_rate
    sr_ms = sr / 1000
    demucs = Demucs(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
    x = th.randn(1, int(sr * 4)).to(args.device)
    out = demucs(x[None])[0]
    streamer = DemucsStreamer(demucs, num_frames=args.num_frames)
    out_rt = []
    frame_size = streamer.total_length
    with th.no_grad():
        while x.shape[1] > 0:
            out_rt.append(streamer.feed(x[:, :frame_size]))
            x = x[:, frame_size:]
            frame_size = streamer.demucs.total_stride
    out_rt.append(streamer.flush())
    out_rt = th.cat(out_rt, 1)
    model_size = sum(p.numel() for p in demucs.parameters()) * 4 / 2**20
    initial_lag = streamer.total_length / sr_ms
    tpf = 1000 * streamer.time_per_frame
    print(f"model size: {model_size:.1f}MB, ", end='')
    print(f"delta batch/streaming: {th.norm(out - out_rt) / th.norm(out):.2%}")
    print(f"initial lag: {initial_lag:.1f}ms, ", end='')
    print(f"stride: {streamer.stride * args.num_frames / sr_ms:.1f}ms")
    print(f"time per frame: {tpf:.1f}ms, ", end='')
    print(f"RTF: {((1000 * streamer.time_per_frame) / (streamer.stride / sr_ms)):.2f}")
    print(f"Total lag with computation: {initial_lag + tpf:.1f}ms")


if __name__ == "__main__":
    test()



#------------experimens 
'''


class DemucsFrequencyBC(nn.Module):
   
    @capture_init
    def __init__(self,
                 hidden=48,
                 depth=5,
                 kernel_size=1,
                 stride=4,
                 causal=True,
                 fft_size=512,
                 normalize=True,
                 floor=1e-3,
                 win_len=400,
                 win_inc=100,
                 rnn_units=257,
                 win_type='hamming',
                 sample_rate=16_000):
        super().__init__()

        #parameters
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.fft_size = fft_size
        self.normalize = normalize
        self.floor = floor
        self.hop_length = math.floor(fft_size / 4)
        self.resample = self.hop_length / fft_size
        self.win_len = win_len
        self.win_inc = win_inc 
        self.win_type = win_type
        self.rnn_units = rnn_units
        self.rnn_input_size = (fft_size // 2 ) + 1
        self.sample_rate = sample_rate


        #layers
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
        self.lstm = BLSTM(self.rnn_input_size, bi=not causal)
        self.conv = nn.Conv1d(self.rnn_units , ((fft_size // 2 ) + 1), 1)
        self.norm = nn.BatchNorm1d((fft_size // 2 ) + 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_size, self.win_type, 'real')
    
    def forward(self, mix1, mix2):
        length = mix1.size(2)
        mag1, phase1 = self.stft(mix1)  
        mag2, phase2 = self.stft(mix2) 
        mags = th.cat((mag2, mag1), dim=2)
        mask = mags.permute(2, 0, 1)
        mask, _ = self.lstm(mask)
        mask = mask.permute(1, 2, 0) 
        mask = self.conv(mask)
        mask = self.norm(mask)
        mask = self.relu(mask)
        mask = self.conv(mask)
        mask = self.sig(mask)
        mask = mask[:, :, :mag1.size(2)]
        estimated = mag1 * mask
        out = self.istft(estimated, phase=phase1)
        szl = out.size(2)
        output = F.pad(out, (0, length - szl), "constant", 0)
        output = output[..., :length]

        return output   
        '''