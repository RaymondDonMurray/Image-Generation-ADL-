# Auto-Regressive Image Generation

This project implements an end-to-end generative model pipeline for synthesizing game scene images. The system learns to generate novel 100×150 pixel SuperTuxKart images by first compressing images into discrete tokens, then modeling the statistical relationships between these tokens using a transformer-based architecture.

## Overview

The project demonstrates a complete generative modeling pipeline inspired by modern language models, but applied to image synthesis. Instead of predicting the next word in a sentence, the model predicts the next visual "token" in an image.

**Key Components:**
1. **Patch-Level Auto-Encoder** - Compresses images into compact latent representations
2. **Binary Spherical Quantization (BSQ)** - Converts continuous features into discrete tokens
3. **Autoregressive Transformer** - Learns patterns in token sequences to enable generation
4. **Image Generation** - Samples novel images token-by-token from the learned distribution

## Theoretical Background

### Auto-Encoders and Dimensionality Reduction

An auto-encoder learns to compress high-dimensional data (like images) into a lower-dimensional latent space, then reconstruct the original data from this compressed representation. By training the network to minimize reconstruction error, the encoder learns to capture the most important features while discarding redundant information.

In this implementation, images are divided into patches (5×5 pixel blocks), and each patch is independently encoded into a learned feature vector. This patch-based approach reduces computational complexity while preserving local visual structure.

**Why patches?** Raw pixel-level modeling would require learning relationships between 45,000 values (100×150×3). Patch-based encoding reduces this to just 600 tokens (20×30 grid), making the problem tractable while maintaining visual fidelity.

### Quantization and Discrete Representations

Continuous latent spaces present a challenge for generative modeling - there are infinitely many possible values. Binary Spherical Quantization solves this by:

1. Projecting each latent vector onto a unit sphere (L2 normalization)
2. Discretizing each dimension to either +1 or -1 (binarization)
3. Creating a finite vocabulary of 2^10 = 1,024 distinct tokens

This discretization transforms the image generation problem into a sequence modeling problem, similar to text generation in language models.

### Autoregressive Models and Transformers

An autoregressive model generates data one element at a time, where each new element depends on all previously generated elements. The joint probability of a sequence is factorized as:

```
P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × P(x₃|x₁,x₂) × ... × P(xₙ|x₁,...,xₙ₋₁)
```

The transformer architecture excels at this task through its self-attention mechanism, which allows each position to attend to all previous positions. By using causal masking, the model ensures that predictions at position *i* only depend on positions 0 through *i-1*, maintaining the autoregressive property.

**Key Innovation:** The same architectural principles that power GPT for text generation apply equally well to image token sequences. The model learns visual patterns and structures just as language models learn grammar and semantics.

## Implementation Approach

### 1. Patch Auto-Encoder

**High-Level Algorithm:**
```
Encoder:
  1. Divide image into non-overlapping patches (5×5 pixels each)
  2. Apply learned linear transformation to each patch
  3. Process patches with convolutional layers to enable inter-patch communication
  4. Output: Grid of latent feature vectors (20×30×128)

Decoder:
  1. Process latent features with convolutional layers
  2. Apply learned linear transformation to reconstruct patches
  3. Reassemble patches into complete image
  4. Output: Reconstructed image (100×150×3)
```

The encoder-decoder architecture uses 3×3 convolutions to allow neighboring patches to influence each other, capturing spatial relationships beyond individual patch boundaries. GELU activations provide smooth non-linearities throughout the network.

### 2. Binary Spherical Quantization

**High-Level Algorithm:**
```
Encoding:
  1. Project latent vector to lower dimension (128 → 10)
  2. Normalize to unit sphere (ensures all codes have equal magnitude)
  3. Binarize each dimension using differentiable sign function
  4. Convert binary code to integer index (0-1023)

Decoding:
  1. Convert integer index to binary code (+1/-1 values)
  2. Project back to original latent dimension (10 → 128)
  3. Pass to image decoder for reconstruction
```

The normalization step is crucial - it ensures that all tokens lie on a hypersphere, giving them equal importance and preventing mode collapse during training. The straight-through estimator allows gradients to flow through the non-differentiable sign operation during training.

### 3. Autoregressive Transformer

**High-Level Algorithm:**
```
Forward Pass:
  1. Flatten 2D token grid (20×30) to 1D sequence (600 tokens)
  2. Embed integer tokens into continuous vectors (128-dim)
  3. Add positional encodings (learnable position information)
  4. Shift sequence by one position (output predicts next token)
  5. Apply causal self-attention (prevent looking at future tokens)
  6. Process through 3-layer transformer
  7. Project to vocabulary-sized logits (1024 dimensions)
  8. Reshape back to 2D grid

Training:
  - Optimize cross-entropy loss between predicted and actual next tokens
  - Gradients flow through entire sequence simultaneously (teacher forcing)
```

The causal masking is implemented via an attention mask that sets future positions to negative infinity before the softmax operation, effectively blocking information flow from future to past.

### 4. Image Generation

**High-Level Algorithm:**
```
Generation (Sampling):
  1. Initialize empty token grid (20×30 zeros)
  2. For each position in raster-scan order (top-left to bottom-right):
     a. Run model forward pass with current partial tokens
     b. Extract probability distribution for current position
     c. Sample next token from categorical distribution
     d. Fill in sampled token
  3. Decode final token grid to image using BSQ decoder and auto-encoder
```

Each generation requires 600 forward passes (one per token), making it computationally expensive but ensuring proper autoregressive generation. The stochastic sampling introduces diversity - each run produces different images.

## Results

### Training Configuration

**Patch Auto-Encoder:**
- Patch size: 5×5 pixels
- Latent dimension: 128
- Training epochs: 5 (default)
- Optimizer: AdamW (learning rate: 1e-3)
- Loss function: MSE (L2 reconstruction)

**BSQ Auto-Encoder:**
- Patch size: 5×5 pixels (20×30 token grid)
- Latent dimension: 128
- Codebook bits: 10 (1,024 possible tokens)
- Training epochs: 5 (default)
- Same optimizer and loss as above

**Autoregressive Model:**
- Model dimension: 128
- Vocabulary size: 1,024 tokens
- Transformer layers: 3
- Attention heads: 4
- Feedforward dimension: 512
- Positional embeddings: Learned
- Training epochs: 5 (default)
- Loss: Cross-entropy (reported in bits)

### Generated Images

The model successfully generates novel game scenes with recognizable structure. While images are blurry due to the lossy quantization, the model captures:
- Color distributions (sky, ground, objects)
- Spatial layout (horizon lines, object placement)
- Scene composition (foreground/background separation)

**Sample Generations:**

![Generated Image 0](generated/generation_0.png)
![Generated Image 1](generated/generation_1.png)
![Generated Image 2](generated/generation_2.png)
![Generated Image 5](generated/generation_5.png)
![Generated Image 7](generated/generation_7.png)
![Generated Image 9](generated/generation_9.png)

**Comparison with Different Training Configurations:**

With positional embeddings (better spatial structure):

![With Positional Encoding 1](gen_pos_1.png)
![With Positional Encoding 2](gen_pos_2.png)
![With Positional Encoding 3](gen_pos_3.png)

Without positional embeddings (less coherent):

![Without Positional Encoding 1](gen_nopos_1.png)
![Without Positional Encoding 2](gen_nopos_2.png)
![Without Positional Encoding 3](gen_nopos_3.png)

After one epoch only (early training):

![One Epoch 1](gen_one_1.png)
![One Epoch 2](gen_one_2.png)
![One Epoch 3](gen_one_3.png)

**Observations:**
- Positional embeddings significantly improve spatial coherence and structure
- Early training (1 epoch) produces noisy, random-looking patterns
- Full training (5 epochs) captures scene-level structure and color relationships
- Each sample shows unique variations due to stochastic sampling

### Quality Analysis

**Strengths:**
- Successfully learns abstract visual patterns from training data
- Generates diverse samples through probabilistic sampling
- Captures color palettes characteristic of SuperTuxKart environments
- Models spatial relationships (e.g., sky typically in upper regions)
- Demonstrates autoregressive generation working on visual data

**Limitations:**
- Blurry outputs due to 10-bit quantization bottleneck
- Limited fine detail preservation
- Occasional artifacts at patch boundaries
- Small model size constrains representational capacity

**Why the blur?** The BSQ quantization creates a lossy compression (1,024 possible values per patch vs. ~16 million in RGB space). This information bottleneck forces the model to learn abstract features rather than memorize exact textures, resulting in smooth but imprecise reconstructions.

## Technical Architecture Summary

**Pipeline Flow:**
```
Training Phase:
Image (100×150×3) → Patches → Encoder → BSQ Quantization → Decoder → Reconstructed Image
                                         ↓
                                    Integer Tokens
                                         ↓
                              Saved to tokenized dataset

Autoregressive Training:
Integer Tokens → Embedding → Transformer → Next Token Predictions

Generation Phase:
Start Token → Transformer (600× iterations) → Token Sequence → BSQ Decoder → Generated Image
```

**Key Hyperparameters:**
- Patch size: 5×5 pixels (20×30 token grid per image)
- Latent dimension: 128
- Codebook bits: 10 (1,024 unique tokens)
- Transformer depth: 3 layers
- Attention heads: 4
- Batch size: 64
- Learning rate: 1e-3 (AdamW)

## Conceptual Insights

### Connection to Language Models

This image generation approach mirrors text generation in large language models:

| **Text Generation** | **Image Generation (This Project)** |
|---------------------|-------------------------------------|
| Characters/Words | Pixels |
| Tokenization (BPE) | BSQ Quantization |
| Vocabulary (50k tokens) | Codebook (1k tokens) |
| 1D sequence | 2D sequence (flattened to 1D) |
| GPT predicts next word | Model predicts next token |
| Autoregressive sampling | Autoregressive sampling |

The fundamental insight: Any data that can be tokenized can be modeled autoregressively. The transformer doesn't "know" it's processing images vs. text - it simply learns patterns in discrete sequences.

### Why This Matters

**Generative Modeling:** Understanding how to factorize complex distributions (images) into simpler conditional distributions (next token given previous tokens) is fundamental to modern AI.

**Compression:** This model is also a learned image compressor. The tokenization step represents each image as 600 integers (0-1023), achieving significant compression compared to raw pixels.

**Scalability:** While this project uses a small model on limited data, the same principles scale to:
- DALL-E (text-to-image generation with billions of parameters)
- Imagen Video (autoregressive video generation)
- Any domain with sequential structure

## Project Structure

```
.
├── homework/
│   ├── ae.py                    # Patch auto-encoder implementation
│   ├── bsq.py                   # Binary Spherical Quantization
│   ├── autoregressive.py        # Transformer model
│   ├── generation.py            # Sampling routines
│   ├── train.py                 # Training loop with Lightning
│   ├── tokenize.py              # Dataset tokenization script
│   └── data.py                  # PyTorch DataLoaders
├── generated/                   # Generated image samples
├── checkpoints/                 # Trained model weights
├── data/
│   ├── train/                   # Training images
│   └── valid/                   # Validation images
└── logs/                        # TensorBoard training logs
```

## Reflections

This project demonstrates that the same architectural principles powering modern language models can be applied to visual data. The key innovation is the tokenization step - by converting continuous images into discrete tokens, we transform generation into a sequence modeling problem that transformers handle naturally.

The quality limitations stem from fundamental tradeoffs:
- **Compression ratio** - More aggressive compression (fewer bits) reduces quality
- **Model capacity** - Larger models learn richer patterns but require more compute
- **Training data** - More diverse data improves generalization

Despite these constraints, the model successfully learns to generate plausible game scenes, validating the autoregressive approach to image synthesis.

## Future Directions

**Architectural Improvements:**
- Hierarchical token structures (coarse-to-fine generation)
- Conditioning on text prompts (text-to-image generation)
- Latent diffusion instead of autoregressive decoding (faster sampling)

**Scaling:**
- Larger codebooks (12-14 bits for higher fidelity)
- Deeper transformers (6-12 layers)
- More training data and longer training

**Applications:**
- Game asset generation and augmentation
- Data augmentation for computer vision tasks
- Understanding autoregressive visual modeling

---

*This project was completed as part of an Advanced Deep Learning course, implementing techniques from modern generative modeling research.*
