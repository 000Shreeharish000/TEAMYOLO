import numpy as np
import pywt
import os
import scipy.signal as signal
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Noisy and Ground Truth Data
def load_data(noisy_folder, clean_folder):
    noisy_data = []
    clean_data = []

    for root, _, files in os.walk(noisy_folder):
        for file in files:
            if file.endswith('.npy'):
                noisy_data.append(np.load(os.path.join(root, file)))

    for root, _, files in os.walk(clean_folder):
        for file in files:
            if file.endswith('.npy'):
                clean_data.append(np.load(os.path.join(root, file)))

    if not noisy_data or not clean_data:
        raise ValueError("Error: No .npy files found in one or both folders!")

    # Ensure all signals have the same length
    min_length = min([len(sig) for sig in noisy_data + clean_data])
    noisy_data = np.array([sig[:min_length] for sig in noisy_data])
    clean_data = np.array([sig[:min_length] for sig in clean_data])

    return noisy_data, clean_data

# PSNR Calculation
def psnr(original, denoised):
    mse = mean_squared_error(original.flatten(), denoised.flatten())
    if mse == 0:
        return 100  # Perfect denoising
    max_pixel = np.max(original)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# 1. Wavelet Denoising
def wavelet_denoising(signal, wavelet='db4', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)[:len(signal)]

# 2. Butterworth Low-Pass Filter
def butter_lowpass_filter(signal_data, cutoff=30, fs=256, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, signal_data)

# 3. Autoencoder-Based Denoising
def build_autoencoder(input_shape):
    encoder = keras.Sequential([
        keras.layers.Input(shape=input_shape),  # specify the input shape
        keras.layers.Flatten(),  # flatten the data from (19, 500) to (19*500,)
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu')
    ])
    decoder = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(input_shape[0] * input_shape[1], activation='linear'),  # output shape should match flattened size
        keras.layers.Reshape(input_shape)  # reshape back to (19, 500)
    ])
    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def denoise_autoencoder(model, noisy_data):
    return model.predict(noisy_data)

# Apply and Evaluate Methods
def apply_denoising_methods(noisy_data, clean_data):
    psnr_scores = {}

    # Wavelet Denoising
    wavelet_denoised = np.array([wavelet_denoising(sig) for sig in noisy_data])
    psnr_scores['Wavelet'] = psnr(clean_data, wavelet_denoised)

    # Butterworth Filtering
    butter_denoised = np.array([butter_lowpass_filter(sig) for sig in noisy_data])
    psnr_scores['Butterworth'] = psnr(clean_data, butter_denoised)

    # Autoencoder Denoising
    autoencoder = build_autoencoder((noisy_data.shape[1],))
    
    # Early stopping to prevent endless training
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    autoencoder.fit(noisy_data, clean_data, epochs=10, batch_size=32, verbose=1, callbacks=[early_stopping])
    autoencoder_denoised = denoise_autoencoder(autoencoder, noisy_data)
    psnr_scores['Autoencoder'] = psnr(clean_data, autoencoder_denoised)

    return psnr_scores, wavelet_denoised, butter_denoised, autoencoder_denoised

# Corrected Folder Paths
noisy_folder = "C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/noisy_train_data"
clean_folder ="C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/train_data"

# Load Data
noisy_data, clean_data = load_data(noisy_folder, clean_folder)

# Run Denoising Methods
psnr_results, wavelet_denoised, butter_denoised, autoencoder_denoised = apply_denoising_methods(noisy_data, clean_data)

# Print PSNR Scores
print("\nPSNR Scores for Different Methods:")
for method, score in psnr_results.items():
    print(f"{method}: {score:.2f} dB")

# Plot Examples
plt.figure(figsize=(12, 6))
plt.plot(clean_data[0], label='Original', linestyle='dashed')
plt.plot(noisy_data[0], label='Noisy', alpha=0.6)
plt.plot(wavelet_denoised[0], label='Wavelet Denoised')
plt.plot(butter_denoised[0], label='Butterworth Denoised')
plt.plot(autoencoder_denoised[0], label='Autoencoder Denoised')
plt.legend()
plt.title("Comparison of Denoising Methods")
plt.show()
