#!/usr/bin/env python3
"""Med-Gemma Impact Challenge - Main Application Entry Point.

This module provides the Gradio web interface for medical image analysis
using Google's Med-Gemma 1.5 model.
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import torch

from src.agent import MedGemmaAgent
from src.utils import create_sample_ct, create_sample_xray, setup_logging

# ============================================
# CONFIGURATION
# ============================================

MODEL_ID = "google/medgemma-1.5-4b-it"
HF_TOKEN = os.environ.get("MedGemma_Challenge_Kaggle") or os.environ.get("HF_TOKEN")
CACHE_DIR = Path("/tmp/model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# ============================================
# INITIALIZE AGENT
# ============================================

agent = MedGemmaAgent(model_id=MODEL_ID, token=HF_TOKEN, cache_dir=CACHE_DIR)

# ============================================
# GRADIO INTERFACE
# ============================================

custom_css = """
/* Main container */
.gradio-container { 
    max-width: 1200px !important; 
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif !important;
    background: transparent !important;
}

/* Better text visibility */
body, p, div, span, h1, h2, h3, h4, h5, h6 {
    color: #2c3e50 !important;
}

/* Headers */
h1 {
    text-align: center;
    color: #1a5276 !important;
    font-size: 2.5em !important;
    margin-bottom: 0.5em !important;
    font-weight: 600 !important;
}

h3 {
    color: #2874a6 !important;
    font-weight: 500 !important;
    margin-top: 0 !important;
}

/* Status boxes */
.status-box { 
    padding: 15px; 
    border-radius: 8px; 
    margin: 15px 0; 
    font-weight: bold;
    font-size: 1.1em;
    border-left: 5px solid;
}

.token-success { 
    background: #d4edda; 
    color: #155724 !important;
    border-left-color: #28a745;
}

.token-error { 
    background: #f8d7da; 
    color: #721c24 !important;
    border-left-color: #dc3545;
}

/* Loading box */
.loading-box { 
    background: #e8f4f8; 
    border-left: 5px solid #3498db; 
    padding: 20px; 
    margin: 20px 0;
    border-radius: 8px;
}

/* Image upload area */
.image-input {
    background: transparent !important;
    border: 2px dashed #3498db !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

/* Text output area */
textarea, .output-text {
    background: transparent !important;
    color: #2c3e50 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    border: 2px solid #dee2e6 !important;
    border-radius: 8px !important;
    padding: 15px !important;
}

/* Buttons */
button.primary {
    background: linear-gradient(135deg, #3498db, #2980b9) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 1.1em !important;
    border-radius: 8px !important;
}

/* Disclaimer */
.disclaimer { 
    color: #666 !important; 
    font-style: italic; 
    padding: 15px; 
    border-left: 5px solid #e67e22; 
    background: #fef9e7; 
    margin: 20px 0; 
    border-radius: 0 8px 8px 0;
}

/* Footer */
.footer { 
    text-align: center; 
    color: #7f8c8d !important; 
    padding: 20px; 
    font-size: 0.9em;
    border-top: 1px solid #dee2e6;
    margin-top: 20px;
}
"""

# Create sample images
sample_xray = create_sample_xray()
sample_ct = create_sample_ct()

# Create the interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Med-Gemma Impact Challenge") as demo:
    gr.Markdown("""
    # 🩺 Med-Gemma Impact Challenge
    
    ### AI-Powered Medical Image Analysis with Google's Med-Gemma 1.5
    
    Upload medical images (X-rays, CT scans, MRIs) to get structured clinical reports.
    """)
    
    # Token status
    if HF_TOKEN:
        gr.HTML(f'''
        <div class="status-box token-success">
            ✅ <strong>Token Status:</strong> Configured successfully (starts with: {HF_TOKEN[:8]}...)
        </div>
        ''')
    else:
        gr.HTML('''
        <div class="status-box token-error">
            ❌ <strong>Token Missing!</strong> Add MedGemma_Challenge_Kaggle to Space secrets
        </div>
        ''')
    
    # Loading information
    gr.HTML('''
    <div class="loading-box">
        <strong>⏳ First Load Information:</strong>
        <ul style="margin-top: 10px; margin-bottom: 5px;">
            <li>📦 First load takes <strong>3-5 minutes</strong> (downloading 2.5GB model)</li>
            <li>⚡ Subsequent loads take <strong>30-60 seconds</strong> (cached)</li>
        </ul>
    </div>
    ''')
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(
                type="pil", 
                label="Select or drop medical image",
                height=400,
                elem_classes="image-input"
            )
            
            with gr.Row():
                task_type = gr.Dropdown(
                    choices=["Full Report", "Findings Only", "Clinical Impression", "Quick Analysis"],
                    value="Full Report",
                    label="📋 Report Type"
                )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                temperature = gr.Slider(0.0, 1.0, 0.1, label="🌡️ Temperature")
                max_tokens = gr.Slider(100, 1024, 512, label="📝 Max Tokens")
            
            analyze_btn = gr.Button("🔍 Generate Report", variant="primary", size="lg", elem_classes="primary")
            
            gr.Markdown("### 📋 Example Images")
            with gr.Row():
                btn_xray = gr.Button("🫁 Chest X-ray", elem_classes="secondary")
                btn_ct = gr.Button("🫀 Chest CT", elem_classes="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Result")
            output = gr.Textbox(
                label="Medical Analysis Report",
                lines=20,
                show_copy_button=True,
                placeholder="Click 'Generate Report' to start analysis...",
                elem_classes="output-text"
            )
            
            with gr.Row():
                time_display = gr.Markdown("⏱️ **Ready**")
                tokens_display = gr.Markdown("📊 **-**")
    
    # Disclaimer
    gr.HTML('''
    <div class="disclaimer">
        ⚕️ <strong>Important Medical Disclaimer:</strong><br>
        This AI-generated analysis is for <strong>research and educational purposes only</strong>. 
        Not for clinical use without proper validation.
    </div>
    ''')
    
    gr.HTML(f'''
    <div class="footer">
        <p>🚀 Powered by <strong>Google's Med-Gemma 1.5</strong> (4-bit quantized)</p>
        <p>🔑 Token: MedGemma_Challenge_Kaggle | ⚡ GPU: {'Available' if torch.cuda.is_available() else 'Not Available'}</p>
    </div>
    ''')
    
    def analyze_with_progress(image, task, temp, tokens, progress=gr.Progress()):
        """Analyze with progress tracking."""
        if not HF_TOKEN:
            return "❌ Token missing!", "⏱️ Error", "📊 -"
        if image is None:
            return "⚠️ Please upload an image.", "⏱️ Waiting", "📊 -"
        
        if agent.model is None:
            progress(0, desc="Loading model...")
            success = agent.load_model_with_progress(progress)
            if not success:
                return f"❌ Failed to load model.", "⏱️ Error", "📊 -"
        
        progress(0.5, desc="Analyzing image...")
        start = time.time()
        result = agent.analyze(image, task, temp, tokens)
        elapsed = time.time() - start
        
        return result, f"⏱️ {elapsed:.1f}s", f"📊 ~{int(len(result.split())*1.3)} tokens"
    
    analyze_btn.click(
        analyze_with_progress,
        [image_input, task_type, temperature, max_tokens],
        [output, time_display, tokens_display]
    )
    
    btn_xray.click(lambda: sample_xray, None, image_input)
    btn_ct.click(lambda: sample_ct, None, image_input)

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 MED-GEMMA IMPACT CHALLENGE")
    print("="*60)
    print(f"Token: {'✅' if HF_TOKEN else '❌'}")
    print(f"GPU: {torch.cuda.is_available()}")
    print("="*60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )