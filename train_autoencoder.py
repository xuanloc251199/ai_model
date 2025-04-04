import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from load_utkface_dataset import load_data

IMG_SIZE = 128
LATENT_DIM = 256
AGE_CLASSES = 3

def build_encoder():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    z = layers.Dense(LATENT_DIM)(x)
    return models.Model(inp, z, name="encoder")

def build_decoder():
    z_input = Input(shape=(LATENT_DIM,))
    age_input = Input(shape=(AGE_CLASSES,))
    x = layers.Concatenate()([z_input, age_input])
    x = layers.Dense(16 * 16 * 128, activation='relu')(x)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    out = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return models.Model([z_input, age_input], out, name="decoder")

# Load dữ liệu
X, y_onehot, label_encoder = load_data()

# Xây dựng mô hình
encoder = build_encoder()
decoder = build_decoder()

img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
age_input = Input(shape=(AGE_CLASSES,))
z = encoder(img_input)
recon = decoder([z, age_input])
autoencoder = models.Model([img_input, age_input], recon)

autoencoder.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint("model_autoencoder_best.h5", save_best_only=True)
]

autoencoder.fit(
    [X, y_onehot], X,
    batch_size=32,
    epochs=1000,
    validation_split=0.1,
    callbacks=callbacks
)

autoencoder.save("model_autoencoder_final.h5")
print("Đã huấn luyện và lưu mô hình hoàn tất.")
