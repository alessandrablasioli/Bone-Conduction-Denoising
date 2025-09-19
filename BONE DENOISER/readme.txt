Real Time Speech Enhancement in Time and Frequency domain


Starting from Facebook’s work on audio denoising (Demucs), we first validated the performance of the Demucs model using the Valentini dataset.
After this validation step, we prepared the Vibravox dataset for our experiments.
Among the various microphones available in Vibravox, we focused on the Shure WH20XLR air-conduction microphone and the AKG C411 bone-conduction microphone.
We randomly mixed clean bone- and air-conduction recordings with matching noise samples taken from the dataset’s “speechless” noise segments.

With this custom data in place, we explored a method to synthesize bone-conduction audio directly from air-conduction recordings.
We developed four versions of this transformation filter:

Manual filter design – Based on a visual comparison of the frequency responses of bone- and air-conduction signals.
Using a combination of band-pass and low-pass filters, we achieved good results both in listening tests and in matching the target frequency curve.

2–4. VibVoice-inspired approaches – Following ideas from VibVoice, we implemented three alternatives:

            - an Otsu-based filtering method (as in the original VibVoice work),
            
            - a triangle-filtering method, and
            
            - a spectrogram-difference method, which models the difference between bone and air spectrograms.
            
Ultimately, we retained only the Otsu-based filter and the manually designed filter as our synthetic bone-conduction generators.

Next, we modified the Demucs architecture to incorporate bone-conduction input, aiming to test whether this additional information improves speech enhancement.
Our first variant duplicated the network’s input path to accept both the noisy air-conduction signal and the corresponding bone-conduction signal.

We then implemented several additional versions:

        - Demucs with two inputs (noisy air + bone)
        
        - Demucs operating in the frequency domain
        
        - Demucs operating in the frequency domain with two inputs (noisy air + bone)

Planned future work includes:

        - Demucs with two inputs and an attention mechanism replacing the dual LSTM layers
        
        - A frequency-domain Demucs with two inputs and attention in place of the LSTMs

The results showed up that: 


|   Model                    |   PESQ Improvement (%)  |   STOI Improvement (%)   |
|----------------------------|-------------------------|--------------------------|
|  DemucsDouble              | 5.39                    | 0.86                     |
|  DemucsFrequencyDualLSTM   | 3.61                    | 0.87                     |



