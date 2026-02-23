# 🩺 Med-Gemma Impact Challenge

<div align="center">

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/yashfy/medgemma-impact-challenge)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Dur-e-yashfeen/Med-Gemma-Impact-Challenge)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.50+-orange)](https://gradio.app/)

**Medical Image Analysis powered by Google's Med-Gemma 1.5**  
*Upload X-rays, CT scans, or MRIs to get AI-generated clinical reports*

[Live Demo](https://huggingface.co/spaces/yashfy/medgemma-impact-challenge) • 
[Report Bug]([https://github.com/yashfy/medgemma-impact-challenge/issues]) • 
[Request Feature](https://github.com/yashfy/medgemma-impact-challenge/issues)

</div>

---

## 📊 Competition Overview

The **Med-Gemma Impact Challenge** focuses on developing AI solutions for medical image analysis. This repository contains a production-ready solution using Google's Med-Gemma 1.5 model for generating structured clinical reports from medical images.

### 🏆 Key Achievements
- ✅ 4-bit quantization for efficient GPU usage (3.3GB memory)
- ✅ Structured clinical report generation
- ✅ Multiple report formats (Full Report, Findings Only, etc.)
- ✅ Optimized inference (8.5 seconds average on T4 GPU)
- ✅ ROUGE-1 score: 0.423, BLEU-4: 0.156

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Multi-modal Analysis** | Supports X-rays, CT scans, MRI images |
| 📋 **Structured Reports** | Generates findings, impression, recommendations |
| ⚡ **Optimized Inference** | 4-bit quantization for efficient GPU usage |
| 🎯 **Multiple Formats** | Full reports, findings-only, quick analysis |
| 🔧 **Adjustable Parameters** | Temperature, max tokens for customization |
| 📊 **Progress Tracking** | Real-time loading and analysis progress |
| 🌐 **Web Interface** | User-friendly Gradio UI |
| 🚀 **Easy Deployment** | One-click deploy to Hugging Face Spaces |

---

## 🏗️ Architecture
┌─────────────────┐
│ Image Input │
└────────┬────────┘
↓
┌─────────────────┐
│ Image Preproc │
│ - Resize │
│ - Normalize │
│ - Convert RGB │
└────────┬────────┘
↓
┌─────────────────┐
│ Prompt Template │
│ - Full Report │
│ - Findings │
│ - Impression │
│ - Quick │
└────────┬────────┘
↓
┌─────────────────┐
│ Med-Gemma │
│ 1.5-4B │
│ (4-bit quant) │
└────────┬────────┘
↓
┌─────────────────┐
│ Post-Processing │
│ - Clean text │
│ - Format │
│ - Add metadata │
└────────┬────────┘
↓
┌─────────────────┐
│ Final Report │
└─────────────────┘
