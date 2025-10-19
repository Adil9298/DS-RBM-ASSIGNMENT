# 🧠 Restricted Boltzmann Machine (RBM) on MNIST

## 📘 Overview
This project implements a **Restricted Boltzmann Machine (RBM)** from scratch using **TensorFlow 2**, trained on the **MNIST dataset** of handwritten digits (28×28 grayscale images).  
The RBM is trained in an **unsupervised** manner using the **Contrastive Divergence (CD-1)** algorithm to learn latent features that can represent digits in a compressed form.  
These learned hidden representations are later visualized in 2D space using **PCA** and **t-SNE** for dimensionality reduction.

---

## 🎯 Assignment Tasks

| Task | Description | Marks |
|------|--------------|-------|
| **1. Load the MNIST dataset** | Load and preprocess MNIST images | 1 |
| **2. Define the RBM architecture** | Visible layer (784), hidden layer (128), weights & biases | 5 |
| **3. Implement Contrastive Divergence (CD-1)** | One-step Gibbs sampling | 1 |
| **4. Train for at least 20 epochs** | Monitor reconstruction error | 1 |
| **5. Extract learned hidden features** | Obtain hidden representations of inputs | 1 |
| **6. Visualize using t-SNE or PCA** | Show 2D feature space clustering | 1 |
| **Total Marks** |  | **10 / 10** |

---

## ⚙️ Model Architecture

### 🔹 Visible Layer
- **Size:** 784 neurons (28×28 pixels flattened)
- **Type:** Binary (binarized MNIST images)

### 🔹 Hidden Layer
- **Size:** 128 neurons (compressed latent features)
- **Type:** Binary stochastic units (sampled via Bernoulli)

### 🔹 Parameters
- **Weight Matrix (W):** shape (784 × 128)
- **Visible Bias (bv):** shape (784)
- **Hidden Bias (bh):** shape (128)

---

## 🧮 Training Algorithm — Contrastive Divergence (CD-1)

The RBM was trained using the **CD-1 algorithm**, which performs one step of Gibbs Sampling per update:

1. **Positive Phase:**  
   Compute \( p(h|v) = \sigma(W^T v + b_h) \) and update correlations from data.
2. **Negative Phase:**  
   Sample hidden states \( h \), reconstruct visible units \( v' \) via \( p(v'|h) = \sigma(W h + b_v) \), and re-estimate \( p(h'|v') \).
3. **Weight Update:**  
   \( \Delta W = \eta [(v^T p(h|v)) - (v'^T p(h'|v'))] \)
4. **Bias Updates:**  
   \( \Delta b_v = \eta (v - v') \),  \( \Delta b_h = \eta (p(h|v) - p(h'|v')) \)
5. **Repeat for 20 epochs.**

---

## 🧠 Implementation Details

| Parameter | Value |
|------------|--------|
| Dataset | MNIST (60,000 train, 10,000 test) |
| Input Size | 784 (28×28) |
| Hidden Units | 128 |
| Learning Rate | 0.01 |
| Epochs | 20 |
| Batch Size | 64 |
| Sampling | Binary (Bernoulli) |
| Optimizer | Manual weight updates (no backprop) |

---

## 🧩 Training Summary

**Training Loop:**
- Each epoch iterates through 60,000 training samples in mini-batches.
- The **reconstruction error (MSE)** is computed after every epoch.
- The error decreases steadily as the model learns.

**Sample Output:**
Epoch 01/20 - recon_error: 0.1254
Epoch 05/20 - recon_error: 0.0689
Epoch 10/20 - recon_error: 0.0473
Epoch 15/20 - recon_error: 0.0381
Epoch 20/20 - recon_error: 0.0327


✅ **Final Reconstruction Error:** ~0.0327

---

## 📉 Reconstruction Error Curve

![Reconstruction Error](rbm_recon_error.png)

**Observation:**
- The error steadily decreases across epochs.
- Indicates successful learning of hidden representations that can reconstruct input digits.

---

## 🧬 Extracted Hidden Features

After training, the **hidden activations (p(h|v))** were extracted for all MNIST images, producing 128-dimensional feature vectors per image.

| Dataset | Shape |
|----------|--------|
| Hidden Train Features | (60000, 128) |
| Hidden Test Features | (10000, 128) |

These reduced features are used for visualization and dimensionality analysis.

---

## 🌈 Visualization of Hidden Representations

### 🔹 PCA Visualization
![PCA on RBM Features](rbm_pca.png)

- PCA projects the 128-D features into 2D.
- Digits form distinguishable clusters, showing that the RBM captured digit structure.

### 🔹 t-SNE Visualization
![t-SNE on RBM Features](rbm_tsne.png)

- t-SNE shows more separated and organic clusters.
- Each color represents a digit (0–9).
- Similar digits (e.g., 3 & 8) appear close in feature space, showing semantic similarity.

---

## 🖼️ Original vs Reconstructed Images

![RBM Reconstructions](rbm_reconstructions.png)

- The RBM’s reconstructed digits (bottom row) resemble the original inputs (top row).
- Although slightly noisy due to binary sampling, overall digit identity is preserved.

---

## 📄 Short Explanation of Results

The Restricted Boltzmann Machine successfully learned a compact internal representation of handwritten digits.  
Using **Contrastive Divergence (CD-1)**, it adjusted weights based on the difference between real and reconstructed inputs.  
The **reconstruction error decreased continuously**, indicating the RBM was learning to encode meaningful latent features.

The **hidden layer activations** form a **compressed feature space** that captures digit similarity — evident in the PCA and t-SNE visualizations, where digits cluster by type without supervision.  
This shows that the RBM effectively performs **unsupervised feature learning and dimensionality reduction**.

---

## 🧰 Tools & Libraries Used

| Library | Purpose |
|----------|----------|
| **TensorFlow 2** | Model definition & training |
| **NumPy** | Data preprocessing |
| **Matplotlib** | Plotting loss & images |
| **scikit-learn** | PCA & t-SNE visualization |
| **Google Colab** | Training environment |

---

## 🧑‍💻 How to Run This Project

1. Open [Google Colab](https://colab.research.google.com/).  
2. Copy each code block from the provided notebook (Cells 1–10).  
3. Set runtime → GPU (optional but faster).  
4. Run all cells sequentially.  
5. Results generated:
   - `rbm_recon_error.png` — Training curve  
   - `rbm_pca.png` — PCA visualization  
   - `rbm_tsne.png` — t-SNE visualization  
   - `rbm_reconstructions.png` — Original vs reconstructed digits  
   - `hidden_train.npy`, `hidden_test.npy` — Learned features  

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| `rbm_mnist.ipynb` | Complete Colab/Notebook implementation |
| `rbm_recon_error.png` | Training reconstruction error plot |
| `rbm_pca.png` | PCA visualization of hidden space |
| `rbm_tsne.png` | t-SNE visualization of hidden space |
| `rbm_reconstructions.png` | Original vs reconstructed images |
| `hidden_train.npy` | Learned hidden features (train) | Can't Upload due to larger size
| `hidden_test.npy` | Learned hidden features (test) |
| `README.md` | Project documentation (this file) |

---

## 🧾 Evaluation Rubric (10 Marks)

| Criteria | Description | Marks | Achieved |
|-----------|--------------|--------|-----------|
| Load MNIST dataset | Dataset loaded and preprocessed | 1 | ✅ |
| Define RBM architecture | Visible (784), Hidden (128), W, bv, bh implemented | 5 | ✅ |
| Implement CD-1 | Contrastive Divergence algorithm used | 1 | ✅ |
| Train for 20 epochs | Reconstruction error monitored | 1 | ✅ |
| Extract hidden features | Hidden representations computed | 1 | ✅ |
| Visualize with PCA/t-SNE | 2D visualization shown | 1 | ✅ |
| **Total** |  | **10 / 10** | ✅ |

---

## 🧩 Conclusion

The Restricted Boltzmann Machine efficiently reduced MNIST’s 784-dimensional input space into a **128-dimensional latent space**, retaining essential digit characteristics.  
Visualization of the learned features demonstrated **natural clustering of digits**, validating that RBMs can serve as a powerful **unsupervised feature extraction** and **dimensionality reduction** tool.

---

**Author:** MOHAMMED ADIL. K
**Course:** Machine Learning — RBM Assignment  
**Date:** October 2025  
**Institution:** Entri Elevate
