Facial GAN Training Suite
=========================

This repository hosts an end-to-end pipeline for experimenting with facial image generation using Generative Adversarial Networks (GANs). Our goal is to train and compare a progression of models—Vanilla GAN, StyleGAN, and StyleGAN2-ADA—to understand their qualitative and quantitative differences on curated face datasets.

Project Structure
-----------------
- `config.py` / `configs/`: Load dataset and experiment settings stored in TOML files.
- `data_preprocess.py`: Downloads face datasets from Kaggle and normalizes every image to 224×224 JPEG (RGB).
  CelebA/FaceScrub/PubFig identities are shuffled with a fixed random seed and remapped to integer folders, and the FFHQ subset is deterministically down-sampled to 30k images.
- `utils/helper.py`: Shared utilities, including Kaggle API authentication.
- `raw_data/`: Destination for the raw Kaggle downloads.
- `data/`: Preprocessed splits written as `<dataset>/<identity_id>/<image>.jpg` (datasets without identities keep their original folder structure).

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
2. (CelebA only) Place the official `identity_CelebA.txt` file under `raw_data/celeba-dataset/` so identities can be remapped.
3. PubFig will be auto-extracted from `pubfig83.v1.tgz` on the first run; ensure the archive is present in `raw_data/pubfig83/`.
4. Adjust `DEFAULT_SIZE`, output format, sampling cap, or seed in `data_preprocess.py` if needed.
5. Run the preprocessing script to standardize all datasets:

   ```bash
   python data_preprocess.py
   ```

   Processed 224×224 JPEG images will be written to `data/<dataset_name>/...`.
   For datasets with identity labels (CelebA identity file, FaceScrub, PubFig), integers starting at 0 replace name-based subfolders (deterministic shuffle with random seed 42 per dataset).
   FFHQ keeps its flat structure but is limited to a reproducible 30k-image subset (seed configurable).

Training Roadmap
----------------
- **Vanilla GAN**: Start with a DCGAN-style baseline on CelebA to validate the data pipeline and training loop.
- **StyleGAN**: Implement or adapt a style-based generator to push toward higher fidelity face synthesis.
- **StyleGAN2-ADA**: Incorporate Adaptive Discriminator Augmentation for improved stability on limited or imbalanced data.

All models will consume the same preprocessed datasets; shared experiment parameters (batch size, learning rate, schedulers, etc.) will be added to `configs/` as we iterate. Training scripts will be organized under a dedicated `train/` package.

Milestones
----------
- [ ] Draft baseline GAN training script and run initial experiments
- [ ] Integrate experiment tracking (TensorBoard, Weights & Biases, etc.)
- [ ] Implement training pipelines for StyleGAN and StyleGAN2-ADA
- [ ] Publish checkpoints and sample galleries for each model

Contributing
------------
This project evolves alongside ongoing experiments. Please open an issue for feature requests or bug reports. Pull requests are welcome—include a short description of your setup and findings to streamline reviews.
