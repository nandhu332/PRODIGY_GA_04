import os
import tensorflow as tf

# === Configuration ===
DATA_DIR   = os.path.join("data", "facades")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
TEST_DIR   = os.path.join(DATA_DIR, "test")

IMG_SIZE    = 256
BATCH_SIZE  = 1
EPOCHS      = 5    # Run only 5 epochs for quick demo
BUFFER_SIZE = 400
LAMBDA      = 100

# === Data Loading & Preprocessing ===
def load_image_pair(image_file):
    img = tf.io.read_file(image_file)
    img = tf.image.decode_jpeg(img)
    w = tf.shape(img)[1] // 2
    inp = img[:, :w, :]
    tar = img[:, w:, :]
    inp = tf.image.resize(inp, [IMG_SIZE, IMG_SIZE])
    tar = tf.image.resize(tar, [IMG_SIZE, IMG_SIZE])
    inp = (inp / 127.5) - 1
    tar = (tar / 127.5) - 1
    return inp, tar

def make_dataset(folder):
    ds = tf.data.Dataset.list_files(os.path.join(folder, "*.jpg"))
    ds = ds.map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(BUFFER_SIZE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Models ===
from models.generator import build_generator
from models.discriminator import build_discriminator

generator = build_generator()
discriminator = build_discriminator()

# === Optimizers & Losses ===
gen_opt  = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(disc_pred, gen_out, target):
    adv = bce(tf.ones_like(disc_pred), disc_pred)
    l1  = tf.reduce_mean(tf.abs(target - gen_out))
    return adv + LAMBDA * l1

def disc_loss(d_real, d_fake):
    real = bce(tf.ones_like(d_real), d_real)
    fake = bce(tf.zeros_like(d_fake), d_fake)
    return real + fake

# === Training Step ===
@tf.function
def train_step(inp, tar):
    with tf.GradientTape(persistent=True) as tape:
        fake = generator(inp, training=True)
        d_real = discriminator([inp, tar], training=True)
        d_fake = discriminator([inp, fake], training=True)
        g_loss = gen_loss(d_fake, fake, tar)
        d_loss = disc_loss(d_real, d_fake)
    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
    return g_loss, d_loss

# === Training Loop ===
def fit(epochs):
    train_ds = make_dataset(TRAIN_DIR)
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs} …")
        for inp, tar in train_ds:
            g, d = train_step(inp, tar)
        print(f"... Finished Epoch {epoch}/{epochs}. Gen loss: {g:.4f}, Disc loss: {d:.4f}")
        if epoch % 5 == 0:
            gen_path = f"./checkpoints/gen_epoch{epoch}"
            disc_path = f"./checkpoints/disc_epoch{epoch}"
            generator.save(gen_path)
            discriminator.save(disc_path)
            print(f"Saving generator to {gen_path}")
            print(f"Saving discriminator to {disc_path}")

# === Main ===
if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Using TRAIN_DIR: {TRAIN_DIR}")
    fit(EPOCHS)
