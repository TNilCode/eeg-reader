'''
EEG Data Filtering by Taha Nilforooshan, 2024, 
This script demonstrates how to filter EEG data using a bandpass filter and a notch filter.

MIT License

Copyright (c) 2024 TAHA NILFOROOSHAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt

# EEG Data Parameters
n_samples = 1024  # Number of EEG data points per channel
n_channels = 3  # Number of EEG channels
fs = 256  # Sampling frequency in Hz

# Time vector
t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)

# Generate synthetic EEG data (NOT NEEDED IN ACTUAL APPLICATION, JUST FOR TESTING)
np.random.seed(0)  # Seed for reproducibility
eeg_data = np.zeros((n_channels, n_samples))
for i in range(n_channels):
    # Alpha wave: 10 Hz
    alpha_signal = np.sin(2 * np.pi * 10 * t)
    # Beta wave: 20 Hz
    beta_signal = np.sin(2 * np.pi * 20 * t)
    # Noise
    noise = np.random.normal(0, 0.5, n_samples)
    # Combine signals and noise
    eeg_data[i, :] = alpha_signal + beta_signal + noise

# Filter parameters
fs = 256  # Sample rate of the EEG data
lowcut = 8.0  # Low cutoff frequency for the bandpass filter (Hz)
highcut = 30.0  # High cutoff frequency for the bandpass filter (Hz)
notch_freq = 60.0  # Frequency to be notched out (Hz)
quality_factor = 30.0  # Quality factor for notch filter

# Notch filter
def apply_notch_filter(data, freq, fs, quality_factor):
    w0 = freq / (fs / 2)  # Normalized Frequency
    b, a = iirnotch(w0, quality_factor)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Bandpass filter, uses a 5th order butterworth filter.
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Applying the filters
eeg_data_notched = np.apply_along_axis(apply_notch_filter, 1, eeg_data, notch_freq, fs, quality_factor)
eeg_data_filtered = np.apply_along_axis(apply_bandpass_filter, 1, eeg_data_notched, lowcut, highcut, fs)

print(eeg_data_filtered)  # This will be the filtered EEG data ready for further processing

