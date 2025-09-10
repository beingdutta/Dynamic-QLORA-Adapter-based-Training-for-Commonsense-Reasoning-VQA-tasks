# 🧩 Reasoning-Aware Modular MoE for Visual Question Answering  

This repository implements a **Reasoning-Aware Mixture-of-Experts (MoE)** framework for **Visual Question Answering (VQA)**. The model leverages **LoRA adapters** and a **lightweight router** to efficiently handle diverse reasoning tasks such as **spatial, quantity, physical, and social reasoning**, using a compact **Qwen-VL-2B backbone**.  

---

## 📖 Overview  

Large vision–language models excel at VQA but require enormous compute and memory. Our approach makes reasoning more **modular and efficient** by:  

- Splitting questions into **reasoning categories** (Spatial, Quantity, Social, Physical).  
- Training **LoRA adapters**, each specialized for one reasoning type.  
- Using a **router network** (trained on SBERT embeddings) to dynamically select the right expert at inference time.  
- Running inference through the **Qwen-VL-2B backbone**, augmented with the chosen adapter.  

---

## 🏗️ Architecture  

<div align="center">
  <img src="docs/architecture.png" width="700"/>
</div>  

1. **Input**: Image + Question (classified into reasoning type).  
2. **Embedding + Encoder**: SBERT + Transformer encoder extracts textual representations.  
3. **Router**: Routes the sample to the most relevant reasoning adapter.  
4. **LoRA Adapters**: Specialized modules (Spatial, Quantity, Social, Physical).  
5. **Backbone**: Qwen-VL-2B processes fused representations.  
6. **Output**: Generates the final answer.  

---

## 📊 Datasets  

We train and evaluate on **canonical VQA benchmarks**:  
- **TDIUC** (~500K QA pairs, broad reasoning tasks)  
- **Visual7W** (~327K QA pairs, spatial/commonsense inference)  
- **DAQUAR** (~12K QA pairs, indoor object location/attributes)  

Each QA pair is automatically annotated into one of four **reasoning categories** using **LLaMA-3.2 (11B)** and **LLaMA-3.1 (8B)**.  

---

## ⚙️ Training Workflow  

1. **LoRA Fine-Tuning**  
   - Train one adapter per reasoning category.  
   - Base model (Qwen-VL-2B) remains frozen.  
   - Reduces trainable parameters to **<1%**.  

2. **Router Training**  
   - Lightweight classifier on SBERT embeddings.  
   - Routes questions to the right reasoning adapter.  

3. **Inference**  
   - Router selects adapter dynamically.  
   - Qwen-VL-2B backbone + adapter → generates the answer.  

---

## 📈 Results  

- **Quantity & Spatial adapters** outperform the base backbone.  
- **Router-based MoE** achieves competitive accuracy while staying **memory-efficient**.  
- **Naïve blending (X-LoRA)** underperforms, showing that careful routing is more effective.  

### Example Metrics (Prototype Test Set)  
- **Accuracy**: ~XX%  
- **BERTScore-F1**: ~YY%  
- **BLEU-1**: ~ZZ%  
- **ROUGE-L**: ~AA%  
- **Cosine Similarity (>0.71 threshold)**: ~BB%  

---

## 🚀 Quick Start  

### Installation  

```bash
git clone https://github.com/yourusername/custom-moe-vqa.git
cd custom-moe-vqa
pip install -r requirements.txt
