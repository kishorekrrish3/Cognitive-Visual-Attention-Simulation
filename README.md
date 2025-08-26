# Visual Attention Simulation (Cognitive Vision System) — Streamlit

A simplified cognitive AI simulation that demonstrates:
- **Bottom-up saliency** (Spectral Residual)
- **Foveated vision** (sharp center, blurry periphery)
- **Scanpath with Inhibition of Return (IoR)**

## Features
- Upload any image (JPG/PNG).
- Compute saliency and visualize heatmap overlay.
- Step through fixations or run all fixations at once.
- Foveated rendering around the current fixation.
- Download scanpath as JSON.

## Install & Run
```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Launch
streamlit run app.py
```

## How it works
1. **Spectral Residual Saliency**:
   - Compute FFT of a resized grayscale image.
   - Subtract a local average in the log spectrum (residual).
   - Inverse FFT → saliency → normalize.

2. **Fixation Policy + IoR**:
   - Pick the maximum of (saliency - visited_mask).
   - Apply a Gaussian **Inhibition of Return** around the fixation so the next fixation explores elsewhere.

3. **Foveation**:
   - Blend progressively blurred versions of the image in concentric bands around the fixation (approximation of retinal acuity).

## Project Structure
```
visual_attention_streamlit/
├── app.py
├── requirements.txt
└── README.md
```

## Notes
- This is intentionally simple (no deep networks) but still captures core cognitive ideas.
- You can extend it with videos, motion saliency, or top-down cues if needed.