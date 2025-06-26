import gradio as gr
from inference import preprocess, postprocess, generator

def infer(image):
    # input: PIL Image, size arbitrary
    img = image.resize((256,256))
    inp = preprocess(image)
    fake = generator(inp, training=False)
    return postprocess(fake)

iface = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="pix2pix Translator",
    description="Upload an image; get its pix2pix translation.",
)
if __name__ == "__main__":
    iface.launch()
