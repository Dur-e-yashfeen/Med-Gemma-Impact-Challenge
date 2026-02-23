"""Med-Gemma Agent for medical image analysis."""

import logging
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


class MedGemmaAgent:
    """Medical Image Analysis Agent powered by Med-Gemma."""

    def __init__(
        self,
        model_id: str = "google/medgemma-1.5-4b-it",
        token: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the agent.

        Args:
            model_id: Hugging Face model ID
            token: Hugging Face authentication token
            cache_dir: Directory to cache model files
        """
        self.model_id = model_id
        self.token = token
        self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/model_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_status = "not_loaded"
        self.warmed_up = False

    def load_model(self) -> bool:
        """Load the model with 4-bit quantization.

        Returns:
            True if successful, False otherwise
        """
        if self.model is not None:
            return True

        try:
            logger.info(f"Loading model on {self.device}...")

            if not self.token:
                logger.error("No token provided")
                self.load_status = "error: token missing"
                return False

            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8,
            )

            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                token=self.token,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
            )

            # Load model
            logger.info("Loading model (this may take several minutes)...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=self.token,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir),
                low_cpu_mem_usage=True,
            )

            self.load_status = "loaded"
            logger.info("Model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.load_status = f"error: {str(e)}"
            return False

    def load_model_with_progress(self, progress=None):
        """Load model with progress tracking for Gradio."""
        if self.model is not None:
            if progress:
                progress(1.0, desc="✅ Model already loaded")
            return True

        try:
            if progress:
                progress(0.1, desc="🔑 Checking token...")

            if not self.token:
                if progress:
                    progress(1.0, desc="❌ Token missing!")
                return False

            if progress:
                progress(0.2, desc="⚙️ Configuring quantization...")

            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            if progress:
                progress(0.3, desc="📥 Loading processor...")

            self.processor = AutoProcessor.from_pretrained(
                self.model_id, token=self.token, trust_remote_code=True
            )

            if progress:
                progress(0.4, desc="🔄 Loading model...")

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                token=self.token,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            if progress:
                progress(0.9, desc="🔥 Warming up...")

            self.warmup_model()

            if progress:
                progress(1.0, desc="✅ Ready!")

            self.load_status = "loaded"
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            if progress:
                progress(1.0, desc=f"❌ Error: {str(e)}")
            return False

    def warmup_model(self):
        """Do a small inference to warm up the model."""
        if self.warmed_up or self.model is None:
            return

        try:
            dummy = Image.new("RGB", (224, 224), color="black")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy},
                    {"type": "text", "text": "Describe briefly."}
                ]
            }]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

            self.warmed_up = True
            logger.info("Model warmed up")

        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    def analyze(
        self,
        image: Image.Image,
        task_type: str = "Full Report",
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Analyze medical image and generate report.

        Args:
            image: PIL Image object
            task_type: Type of report to generate
            temperature: Creativity (0-1)
            max_tokens: Maximum output length

        Returns:
            Generated report text
        """
        if image is None:
            return "⚠️ No image provided."

        if self.model is None:
            success = self.load_model()
            if not success:
                return f"❌ Model not loaded. Status: {self.load_status}"

        prompts = {
            "Full Report": """Generate a comprehensive radiology report with:
1. TECHNIQUE: Imaging modality
2. FINDINGS: Detailed observations
3. IMPRESSION: Clinical interpretation
4. RECOMMENDATIONS: Follow-up""",

            "Findings Only": "List all radiological findings in detail.",

            "Clinical Impression": """Provide:
1. Primary diagnosis
2. Differential diagnoses
3. Key findings""",

            "Quick Analysis": "Brief 2-3 sentence summary."
        }

        prompt = prompts.get(task_type, prompts["Full Report"])

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    num_beams=3,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.1,
                )

            response = self.processor.decode(outputs[0], skip_special_tokens=True)

            if "assistant" in response:
                return response.split("assistant")[-1].strip()
            return response

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return f"❌ Error: {str(e)}"