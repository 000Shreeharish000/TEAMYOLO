import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tqdm import tqdm  # For the progress bar during training

# 1. Load Data from .npy files
def load_data_from_npy(noisy_folder, clean_folder):
    noisy_data, clean_data = [], []

    for root, dirs, files in os.walk(noisy_folder):
        for file in files:
            if file.endswith('.npy'):
                noisy_data.append(np.load(os.path.join(root, file)))

    for root, dirs, files in os.walk(clean_folder):
        for file in files:
            if file.endswith('.npy'):
                clean_data.append(np.load(os.path.join(root, file)))

    if not noisy_data or not clean_data:
        raise ValueError("Error: No files found in one or both folders!")

    noisy_data, clean_data = np.array(noisy_data), np.array(clean_data)

    print(f"✅ Loaded {len(noisy_data)} noisy and {len(clean_data)} clean samples")
    print(f"🔹 Shape: Noisy {noisy_data.shape}, Clean {clean_data.shape}")

    if noisy_data.shape != clean_data.shape:
        raise ValueError("❌ Mismatch: Noisy and Clean data have different shapes!")

    return noisy_data, clean_data

# 2. Generator Model
def build_generator(latent_dim, n_classes):
    model = keras.Sequential([
        keras.layers.Input(shape=(latent_dim + n_classes,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(19 * 500, activation='tanh'),
        keras.layers.Reshape((19, 500))  
    ])
    return model

# 3. Discriminator Model
def build_discriminator(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# 4. Conditional GAN (cGAN)
def build_cgan(generator, discriminator):
    discriminator.trainable = False
    return keras.Sequential([generator, discriminator])

# 5. Train cGAN with Progress Bar
def train_cgan(generator, discriminator, cgan, clean_data, n_classes, latent_dim, epochs=5000, batch_size=32, save_interval=500):
    half_batch = batch_size // 2

    for epoch in tqdm(range(epochs), desc="Training Progress"):  # ✅ Progress bar added
        # Train Discriminator
        idx = np.random.randint(0, clean_data.shape[0], half_batch)
        real_data = clean_data[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.randn(half_batch, latent_dim)
        random_classes = np.random.randint(0, n_classes, half_batch)
        random_classes_one_hot = keras.utils.to_categorical(random_classes, num_classes=n_classes)

        combined_input = np.concatenate([noise, random_classes_one_hot], axis=-1)
        fake_data = generator.predict(combined_input, verbose=0)  # ✅ Silent mode

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((half_batch, 1)))

        d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])  # ✅ Convert NumPy to scalar

        # Train Generator
        noise = np.random.randn(batch_size, latent_dim)
        random_classes = np.random.randint(0, n_classes, batch_size)
        random_classes_one_hot = keras.utils.to_categorical(random_classes, num_classes=n_classes)

        combined_input = np.concatenate([noise, random_classes_one_hot], axis=-1)
        g_loss = cgan.train_on_batch(combined_input, np.ones((batch_size, 1)))

        # Save Checkpoints
        if epoch % save_interval == 0:
            generator.save(f'generator_epoch_{epoch}.h5')
            discriminator.save(f'discriminator_epoch_{epoch}.h5')

    print("✅ Training Complete!")

# 6. Analyze and Generate Synthetic Data
def generate_synthetic_data(generator, batch_size, latent_dim, n_classes):
    noise = np.random.randn(batch_size, latent_dim)
    random_classes = np.random.randint(0, n_classes, batch_size)
    random_classes_one_hot = keras.utils.to_categorical(random_classes, num_classes=n_classes)

    combined_input = np.concatenate([noise, random_classes_one_hot], axis=-1)
    synthetic_data = generator.predict(combined_input)

    return synthetic_data

# Path to the models and dataset
noisy_folder = "C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/noisy_train_data"
clean_folder = "C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/train_data"
latent_dim = 100
n_classes = 4  

# Load Data
noisy_data, clean_data = load_data_from_npy(noisy_folder, clean_folder)

# Normalize Clean Data
clean_data = (clean_data - np.min(clean_data)) / (np.max(clean_data) - np.min(clean_data))
clean_data = 2 * clean_data - 1  

# Generator Models for Different Epochs
epochs = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
synthetic_data_by_epoch = {}

# Analyze Synthetic Data Generation for Each Epoch
for epoch in epochs:
    model_path = f'generator_epoch_{epoch}.h5'
    
    if os.path.exists(model_path):  # Check if the model file exists
        generator = load_model(model_path)
        synthetic_data_by_epoch[epoch] = generate_synthetic_data(generator, 100, latent_dim, n_classes)
    else:
        print(f"Warning: Model file for epoch {epoch} not found at {model_path}")

# Visualization of Synthetic Data Compared to Real Data
def plot_comparison(synthetic_data, real_data, epoch):
    # Take the first sample from synthetic data and real data for visualization
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # Adjust the layout for vertical orientation
    axs[0].plot(synthetic_data[0].flatten(), label="Synthetic")
    axs[0].set_title(f'Synthetic Data Epoch {epoch}')
    axs[0].legend()

    axs[1].plot(real_data[0].flatten(), label="Real")
    axs[1].set_title(f'Real Data')
    axs[1].legend()

    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

# Visualizing for the first epoch (you can loop through other epochs similarly)
for epoch, synthetic_data in synthetic_data_by_epoch.items():
    print(f"Visualizing Epoch {epoch}")
    plot_comparison(synthetic_data, clean_data, epoch)
