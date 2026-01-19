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

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   ```bash
   python main.py
   ```
   *Note: On the first run, it attempts to download Stable Diffusion v1-5. If internet is slow or restricted (e.g. without HF Token), it will automatically fallback to a DEBUG mode using mock layers to demonstrate the tensor flow and training logic.*

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
