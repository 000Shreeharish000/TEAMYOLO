import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # ✅ Progress bar for training output

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
        layers.Input(shape=(latent_dim + n_classes,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(19 * 500, activation='tanh'),
        layers.Reshape((19, 500))  # Reshape to desired output size
    ])
    return model

# 3. Discriminator Model (for WGAN)
def build_discriminator(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(1)  # No activation (for WGAN)
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
        random_classes_one_hot = to_categorical(random_classes, num_classes=n_classes)

        combined_input = np.concatenate([noise, random_classes_one_hot], axis=-1)
        fake_data = generator.predict(combined_input, verbose=0)  # ✅ Silent mode

        # WGAN: Real and Fake Losses
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, -np.ones((half_batch, 1)))  # WGAN: -1 for fake data
        
        # No need for indexing; both are scalars
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Train Generator
        noise = np.random.randn(batch_size, latent_dim)
        random_classes = np.random.randint(0, n_classes, batch_size)
        random_classes_one_hot = to_categorical(random_classes, num_classes=n_classes)

        combined_input = np.concatenate([noise, random_classes_one_hot], axis=-1)
        g_loss = cgan.train_on_batch(combined_input, -np.ones((batch_size, 1)))  # WGAN: -1 for generator loss

        # Save Checkpoints
        if epoch % save_interval == 0:
            generator.save(f'generator_epoch_{epoch}.h5')
            discriminator.save(f'discriminator_epoch_{epoch}.h5')

    print("✅ Training Complete!")

# 6. Setup Parameters & Train

noisy_folder = "C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/noisy_train_data"
clean_folder = "C:/Users/Shree Harish V/Desktop/Advitiya Hack/EEG task7 cGAN/EEG_Data/train_data"
latent_dim = 100
n_classes = 4  

noisy_data, clean_data = load_data_from_npy(noisy_folder, clean_folder)

clean_data = (clean_data - np.min(clean_data)) / (np.max(clean_data) - np.min(clean_data))
clean_data = 2 * clean_data - 1  # Scale to [-1, 1]

generator = build_generator(latent_dim, n_classes)
discriminator = build_discriminator(clean_data.shape[1:])
cgan = build_cgan(generator, discriminator)

# Use Adam optimizer with lower learning rate for better stability
adam_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

discriminator.compile(loss='mean_squared_error', optimizer=adam_optimizer)  # WGAN uses MSE loss
cgan.compile(loss='mean_squared_error', optimizer=adam_optimizer)

train_cgan(generator, discriminator, cgan, clean_data, n_classes, latent_dim, epochs=5000, batch_size=32, save_interval=500)
