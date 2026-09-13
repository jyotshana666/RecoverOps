# Production Model Weights Directory

This directory stores trained model checkpoints for the RecoverOps backend.

---

## Default Local Model Location
- **Path**: `models/best.pt`
- **Environment Variable Override**: `MODEL_PATH=/path/to/custom_weights.pt`

---

## Instructions for Deploying Trained Weights from Kaggle:
1. Complete the 50-epoch (or smoke test) training run in Kaggle.
2. Download the artifact `best.pt` from Kaggle:
   `/kaggle/working/RecoverOps/experiments/baseline/runs/rtdetr_l_baseline/weights/best.pt`
3. Place `best.pt` directly in this directory:
   `c:\Work\Projects\RecoverOps\models\best.pt`
4. Start the FastAPI backend:
   ```bash
   uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Verify model readiness:
   ```bash
   curl http://localhost:8000/health
   ```
