# Voice_extracted
## Background
   When you would like to extract someone's clear voice from a noisy video(with BGM or anything), this project can fully fulfill what you need;
   You only need to upload the video and wait it be changed into a clear audio with merely the voice from the person
## How to run it?
1. Make sure you have imported the packages
2. run Voice_extract.py
## Tutorials
1. Change the input file path
2. run Voice_extract.py
## Principle
1. **Video to Audio**: Use the ffmpy3 library to extract audio from the uploaded video.
2. **Voice Activity Detection (VAD)**: Use the Silero VAD model to detect voice segments and merge them.
3. **Vocal Separation**: Use the Demucs algorithm to separate the vocals from the background and retain only the vocals.
4. **Noise Reduction**: Apply multiple noise reduction methods to the separated vocals, including spectral subtraction, Librosa, and DeepFilterNet.
   - reduce_noise_with_librosa: Use the Librosa library to perform noise reduction through spectral subtraction.
   - reduce_noise_with_noisereduce: Use the Noisereduce library for noise reduction.
   - reduce_noise_with_deepfilternet: Use DeepFilterNet for deep learning-based noise reduction.
   - denoise_audio_with_advanced_spectral_subtraction: Use advanced spectral subtraction for audio noise reduction.
