Facial GAN Pipeline
===================

This project prepares large-scale face datasets and trains a sequence of Generative Adversarial Network models (Vanilla GAN, StyleGAN, StyleGAN2-ADA). The repository focuses on reproducible preprocessing, identity-normalised directory layouts, and consistent experiment configuration.

Repository Layout
-----------------
- `config.py` / `configs/`: Centralised TOML configuration (dataset list, target image size, future hyper-parameters).
- `data_preprocess.py`: End-to-end pipeline that downloads, extracts, identity-normalises, resizes, and samples each dataset.
- `utils/helper.py`: Shared helpers (e.g., Kaggle authentication).
- `raw_data/`: Mirrors downloaded archives and raw folders for each dataset.
- `data/`: Contains processed outputs (`<dataset>/<identity>/<image>.jpg` or flat structure when identities are unavailable).
- `requirements.txt`: Minimal dependencies (`kaggle`, `kagglehub`, `Pillow`, `tqdm`).

Getting Started
---------------
1. **Python**: Use Python 3.10+ and create an isolated environment.
2. **Credentials**: Add Kaggle API keys to `.env`:

   ```bash
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_key
   ```

3. **Dependencies**: Install requirements.

   ```bash
   pip install -r requirements.txt
   ```

4. **Dataset Config**: Adjust `configs/datasets.toml`.
   - `dataset.face_data`: List of Kaggle dataset identifiers (downloaded automatically).
   - `dataset.size`: Target square resolution (e.g., 64, 128, 224). Defaults to 224 when omitted.

5. **Auxiliary Files**: Place additional metadata if available.
   - `raw_data/celeba-dataset/identity_CelebA.txt` for identity mapping.
   - `raw_data/pubfig83/pubfig83.v1.tgz` (extracted automatically on first run).

6. **Preprocess Everything**: Run the preprocessing script and monitor tqdm progress per dataset.

   ```bash
   python data_preprocess.py
   ```

   By default the output resolution follows `dataset.size`, images are stored as JPEG, identities are remapped to numeric indices, and FFHQ is down-sampled to a deterministic 30k subset. All randomness is driven by a configurable seed (default 42).

Preprocessing Details
---------------------
- **Downloads**: Uses `kagglehub` with environment credentials; archives land in `raw_data/<dataset_name>/`.
- **Extraction**: PubFig archives are safely unpacked; other datasets rely on their native layout.
- **Identity Handling**:
  * CelebA leverages `identity_CelebA.txt` to map filenames to original IDs.
  * FaceScrub and PubFig use folder names; both are merged and re-indexed.
  * Identities are shuffled deterministically and assigned zero-based numeric folders under `data/<dataset>/`.
- **Sampling & Size**:
  * All images are centre-cropped and resized to the configured square resolution using LANCZOS.
  * FFHQ is randomly sampled to 30,000 images (seeded).
  * Non-identity datasets respect optional caps while preserving directory hierarchy.
- **Progress & Logging**: Each dataset displays a tqdm bar and finishes with a summary that reports processed/skipped counts.

Running Specific Datasets
------------------------
- Use `preprocess_dataset(dataset_name, seed=42, max_images=None)` within Python to customise seed or sampling limit per dataset.
- `process_celeba(seed=42)` remains available for quick CLI-style usage.

Next Steps
----------
- Implement GAN training loops (Vanilla, StyleGAN, StyleGAN2-ADA) consuming the processed data.
- Integrate experiment tracking (TensorBoard / Weights & Biases).
- Add evaluation scripts (FID, identity preservation metrics) and publish sample outputs.

Contributing
------------
Issues and pull requests are welcome. Please describe the dataset/seed settings used when reporting results so others can reproduce them exactly.
