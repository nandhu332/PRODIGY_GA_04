# Pix2Pix cGAN Image-to-Image Translation

This repository implements a conditional GAN (pix2pix) for paired image-to-image translation. It includes:

* **Data Download & Preparation**: `download_data.py` downloads and extracts the Facades dataset.
* **Training**: `train.py` defines the U-Net generator, PatchGAN discriminator, loss functions, and a training loop that saves checkpoints every 5 epochs.
* **Inference**: `inference.py` loads the latest generator SavedModel and provides command-line and programmatic image translation.
* **Web Interface**: `app.py` launches a Gradio UI for uploading an image and getting a translated output in the browser.

---

## 📁 Repository Structure

```
pix2pix_app/
├── data/
│   └── facades/                # Extracted dataset: train/ and test/ subfolders
├── download_data.py            # Downloads & extracts facades.tar.gz into data/
├── models/
│   ├── generator.py            # U-Net generator definition
│   └── discriminator.py        # PatchGAN discriminator definition
├── checkpoints/                # SavedModel checkpoints (gen_epochX/ & disc_epochX/)
├── train.py                    # Data loader, train loop, checkpoint saving
├── inference.py                # Load model & translate images via CLI
├── app.py                      # Gradio web app for interactive translation
└── README.md                   # This file
```

## ⚙️ Prerequisites

* Python 3.8+
* Windows/macOS/Linux
* (Optional) GPU with CUDA for faster training

## 📦 Setup

1. **Clone the repo**

   ```bash
   git clone <repo_url>
   cd pix2pix_app
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows (CMD):
   venv\Scripts\activate
   # Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install tensorflow matplotlib numpy pillow gradio
   ```

## 🗂️ Download & Prepare Dataset

Run:

```bash
python download_data.py
```

This will download the Facades dataset and extract it to `data/facades/` with `train/` and `test/` subfolders.

## 🚂 Training

1. Ensure `data/facades/train` and `data/facades/test` exist.
2. Run training for 5 epochs (quick demo):

   ```bash
   python train.py
   ```
3. Check that `checkpoints/gen_epoch5/` and `checkpoints/disc_epoch5/` are created.
4. For full training, edit `EPOCHS` in `train.py` and rerun.

## 🔍 Inference (CLI)

Translate a single image via command-line:

```bash
python inference.py <input_image.jpg> <output_image.png>
```

## 🌐 Web App (Gradio)

Launch the interactive interface:

```bash
python app.py
```

Navigate to `http://localhost:7860` in your browser, upload an image, and see the translation.

## 🔧 Hyperparameters & Customization

* **IMG\_SIZE**: Resolution for resizing (default 256×256)
* **BATCH\_SIZE**: Number of image pairs per batch
* **EPOCHS**: Total training epochs
* **LAMBDA**: Weight for L1 loss in generator objective
* **Learning Rate**: Default `2e-4`, `beta_1=0.5` for Adam

Feel free to experiment with different datasets, network depths, and augmentations!

---

© 2025 Pix2Pix App by Nandhu A. Feel free to fork and contribute!
