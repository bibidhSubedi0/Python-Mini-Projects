
import numpy as np
import matplotlib.pyplot as plt
# Time vector with more samples
t = np.linspace(0, 10, 10000)  # 10 seconds sampled at 5000 points
signal = np.sin(10*t)

# Compute FFT
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))

# Only consider positive frequencies
positive_frequencies = frequencies[frequencies >= 0]
positive_fft = np.abs(fft_result[frequencies >= 0])

# Limit to frequencies up to 10 Hz
freq_limit = 10
limited_indices = positive_frequencies <= freq_limit
limited_frequencies = positive_frequencies[limited_indices]
limited_fft = positive_fft[limited_indices]

# Plot the signal and its Fourier Transform
plt.figure(figsize=(12, 6))

# Time-domain plot
plt.subplot(1, 2, 1)
plt.plot(t, signal)
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Frequency-domain plot
plt.subplot(1, 2, 2)
plt.stem(limited_frequencies, limited_fft)
plt.title('Frequency Domain Signal (Frequencies up to 10 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
