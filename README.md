# Reliability- and Calibration-Aware Hierarchical Empathy Classification from Text Using Transformer Models

An end-to-end NLP framework built to classify empathy from text while prioritizing prediction confidence, trust, and uncertainty estimation. This repository compares a standard flat classifier with a multi-stage hierarchical architecture built on top of DistilBERT.

##  Key Features
* **Unified Empathy Taxonomy:** Merges two large-scale emotion datasets (**Empathetic Dialogues** and **GoEmotions**) into a mapped 5-class empathy label space: *Compassion, Caring, Sadness, Excitement, and Neutral*.
* **Hierarchical Architecture:** Implements a two-stage prediction process (Binary Presence Detection ➡️ Fine-grained Classification) to effectively mitigate neutral-class dominance.
* **Confidence Calibration:** Integrates post-hoc Temperature Scaling optimized via SciPy to minimize Expected Calibration Error (ECE).
* **Uncertainty Estimation:** Employs Monte Carlo (MC) Dropout at inference time to quantify prediction variance and flag ambiguous data.

---

##  System Architecture

1. **Text Tokenization:** Subword tokenization using `DistilBertTokenizerFast` with uniform sequence padding/truncation (Max Length: 128 tokens).
2. **Backbone Encoder:** Shared pre-trained `DistilBERT` base (6 layers, 768 hidden dimensions).
3. **Classification Heads:**
   * **Presence Head:** Linear layer $(768 \rightarrow 2)$ determining if empathy is present or absent (Neutral).
   * **Type Head:** Linear layer $(768 \rightarrow 5)$ identifying the specific empathetic class.
4. **Confidence Gating & Loss:** Multiplicative probability gating during inference ($P_{\text{presence}} \times P_{\text{type}}$) combined with a custom hierarchy-aware distance penalty loss function.

---

##  Experimental Results

Fine-tuning was executed over 3 epochs using the AdamW optimizer (Learning Rate: 2e-5, Batch Size: 16) on a Tesla T4 GPU. 

### Performance Comparison

| Model Architecture | Macro F1-Score | Baseline ECE | Post-Calibration ECE ($T=1.50$) | Calibration Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Flat Classifier** | 0.68 | 0.102 | 0.078 | 24% |
| **Hierarchical Classifier** | **0.72** | **0.077** | **0.043** | **58%** |

### Class-Wise Metrics (Hierarchical Model)
* **Neutral:** F1-Score: `0.80` (Highest performance due to majority distribution)
* **Sadness:** F1-Score: `0.75` (Distinct linguistic markers)
* **Caring / Compassion:** F1-Score: `0.70` / `0.69`
* **Excitement:** F1-Score: `0.65` (Rarest class)

---

##  Environment & Dependencies

* Python 3.12+
* PyTorch 2.0+
* Hugging Face Transformers (4.30+)
* Scikit-Learn
* SciPy
* NumPy & Pandas
