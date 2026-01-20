# DeepDream-4D: Phase 1 - The Semantic Bridge

Project Goal: End-to-end reconstruction of dream worlds from EEG signals.
Current Status: Phase 1 MVP (EEG -> CLIP -> Stable Diffusion Image).

## Project Structure

This project is designed with modularity in mind ("Lego-like" architecture) to support future extensions to 4D.

- `src/models.py`: Contains `EEGEncoder` (1D-CNN + Attention + PCA) and `DreamAdapter`.
- `src/pipeline.py`: Contains `EEG2DreamPipeline` for training and generation.
- `src/data.py`: Dummy data generation (simulating THINGS-EEG).
- `main.py`: Execution script that runs the pipeline (Environment Setup -> Model Init -> Train -> Generate).

## How to Run

### Option 1: Google Colab (Recommended for GPU)
To train the model properly and generate videos without waiting hours, use the provided notebook:
1. Upload `DeepDream_4D_Colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Change Runtime to **GPU** (T4 is free).
3. Run all cells.

### Option 2: Local CPU/GPU
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   *Note: On CPU, video generation will be extremely slow. The script automatically reduces inference steps to 2 for CPU demos.*

## Key Features Implemented

1. **EEG Encoder**:
   - **1D-CNN**: Extracts temporal features from 32-channel EEG.
   - **Residual Attention**: Focuses on signal peaks, suppressing noise.
   - **Subspace Projection**: Learnable linear layer acting as PCA to filter high-frequency noise.

2. **Dream Adapter**:
   - Maps EEG embeddings (Vector) to CLIP Text Space (Sequence `77x768`) for Stable Diffusion conditioning.

3. **Training Strategy**:
   - **Loss**: `CosineSimilarityLoss` between EEG features and CLIP Image Embeddings.
   - **Frozen SD**: UNet and VAE are frozen; only the Encoder/Adapter are trained.

## Future Roadmap (Phase 2 & 3)

See `main.py` output for pseudo-code on integrating **Stable Video Diffusion** for 4D generation.
