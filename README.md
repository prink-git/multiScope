# MultiScope — Multimodal Image-Text Retrieval System

A production-quality deep learning system for computing similarity between images and text descriptions using OpenAI's CLIP model.

## Features

- 🖼️ **Image Encoding**: Convert images to 512-dim normalized embeddings
- 📝 **Text Encoding**: Convert text to matching vector space
- 🔍 **Similarity Scoring**: Compute cosine similarity between image-text pairs
- 🎨 **Interactive Dashboard**: Streamlit UI for real-time retrieval
- ⚡ **GPU Support**: Automatic CUDA detection and fallback to CPU

## Installation

1. Clone or download the project:
   ```bash
   cd MultiScope
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add sample images to `data/images/` folder

## Quick Start

### Run Main Demo
```bash
python -m app.main
```

### Run Dashboard
```bash
streamlit run dashboard.py
```

Then open the URL shown in your terminal (or Streamlit Cloud app URL after deployment).

## Project Structure

```
MultiScope/
├── app/
│   ├── clip_model.py       # CLIP model manager (singleton)
│   ├── image_encoder.py    # Image → embedding
│   ├── text_encoder.py     # Text → embedding
│   ├── similarity.py       # Cosine similarity
│   ├── utils.py            # Helper functions
│   └── main.py             # CLI demo
├── data/images/            # Sample images directory
├── dashboard.py            # Streamlit UI
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Usage Examples

### Python API

```python
from app.image_encoder import encode_image
from app.text_encoder import encode_text
from app.similarity import compute_similarity

# Encode
image_emb = encode_image("data/images/sample.jpg")
text_emb = encode_text("a dog running")

# Compare
score = compute_similarity(image_emb, text_emb)
print(f"Similarity: {score:.4f}")
```

### Batch Processing

```python
from app.text_encoder import encode_text
from app.similarity import compute_similarity

texts = ["dog", "cat", "bird"]
text_embeddings = encode_text(texts)
similarities = compute_similarity(image_emb, text_embeddings)
```

## Model Details

- **Architecture**: CLIP ViT-B/32 (Vision Transformer)
- **Model ID**: `openai/clip-vit-base-patch32`
- **Embedding Dimension**: 512
- **Normalization**: L2 (cosine similarity)
- **Source**: Hugging Face Transformers

## Device Support

- ✅ CUDA GPU (auto-detected)
- ✅ CPU fallback

Check device:
```python
from app.clip_model import get_clip_manager
manager = get_clip_manager()
print(manager.get_device())
```

## Best Practices

1. **Normalize Embeddings**: Always use L2 normalization before similarity
2. **Batch Processing**: Process multiple texts at once for efficiency
3. **Error Handling**: Validate image paths and text input
4. **Memory**: Model uses ~1.5GB VRAM; CPU requires ~3GB RAM

## Performance

- Image encoding: ~50-100ms (GPU), ~200-300ms (CPU)
- Text encoding: ~20-50ms (GPU), ~100-150ms (CPU)
- Similarity computation: <1ms

Built as a production-quality multimodal retrieval system using PyTorch and Hugging Face Transformers.
