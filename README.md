Facial GAN Training Suite
=========================

This repository hosts an end-to-end pipeline for experimenting with facial image generation using Generative Adversarial Networks (GANs). Our goal is to train and compare a progression of models—Vanilla GAN, StyleGAN, and StyleGAN2-ADA—to understand their qualitative and quantitative differences on curated face datasets.

Project Structure
-----------------
- `config.py` / `configs/`: Load dataset and experiment settings stored in TOML files.
- `data_preprocess.py`: Downloads face datasets from Kaggle and normalizes every image to 256×256 PNG (RGB).
- `utils/helper.py`: Shared utilities, including Kaggle API authentication.
- `raw_data/`: Destination for the raw Kaggle downloads.

Environment Setup
-----------------
1. Use Python 3.10 or newer.
2. Provide a Kaggle API key in the `.env` file:

   ```bash
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_key
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Data Acquisition & Preprocessing
--------------------------------
1. Review or edit the dataset list under `dataset.face_data` in `configs/datasets.toml`.
2. Adjust `DEFAULT_SIZE`, output format, or other preprocessing options in `data_preprocess.py` if needed.
3. Run the preprocessing script to standardize all datasets:

   ```bash
   python data_preprocess.py
   ```

   Processed 256×256 PNG images will be written to `data/processed/<dataset_name>/`.

Training Roadmap
----------------
- **Vanilla GAN**: Start with a DCGAN-style baseline on CelebA to validate the data pipeline and training loop.
- **StyleGAN**: Implement or adapt a style-based generator to push toward higher fidelity face synthesis.
- **StyleGAN2-ADA**: Incorporate Adaptive Discriminator Augmentation for improved stability on limited or imbalanced data.

All models will consume the same preprocessed datasets; shared experiment parameters (batch size, learning rate, schedulers, etc.) will be added to `configs/` as we iterate. Training scripts will be organized under a dedicated `train/` package.
