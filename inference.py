import tensorflow as tf
import numpy as np
from PIL import Image
from models.generator import build_generator

# Rebuild model architecture & load weights
generator = build_generator()
generator.load_weights(tf.train.latest_checkpoint('./checkpoints/'))

def preprocess(img_path):
    img = Image.open(img_path).convert('RGB').resize((256,256))
    arr = (np.array(img).astype('float32') / 127.5) - 1
    return tf.expand_dims(arr, 0)

def postprocess(tensor):
    img = tensor[0].numpy()
    img = ((img + 1) * 127.5).astype('uint8')
    return Image.fromarray(img)

def translate(input_path, output_path):
    inp = preprocess(input_path)
    fake = generator(inp, training=False)
    img = postprocess(fake)
    img.save(output_path)
    print(f"Saved translated image to {output_path}")

if __name__ == "__main__":
    import sys
    translate(sys.argv[1], sys.argv[2])
