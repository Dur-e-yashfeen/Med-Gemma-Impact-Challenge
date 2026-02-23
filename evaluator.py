"""Evaluation metrics for competition submission."""

from typing import Dict, List, Optional

import evaluate
import numpy as np
import re


class CompetitionEvaluator:
    """Evaluate model predictions using competition metrics."""

    def __init__(self):
        """Initialize evaluator with metrics."""
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        try:
            self.meteor = evaluate.load("meteor")
        except:
            self.meteor = None

    def compute_metrics(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute all competition metrics.

        Args:
            predictions: List of model predictions
            references: List of ground truth texts

        Returns:
            Dictionary of metric scores
        """
        # Clean texts
        predictions = [self._clean_text(p) for p in predictions]
        references = [[self._clean_text(r)] for r in references]

        results = {}

        # ROUGE scores
        rouge_results = self.rouge.compute(
            predictions=predictions, references=[r[0] for r in references]
        )
        results.update(rouge_results)

        # BLEU score
        bleu_results = self.bleu.compute(
            predictions=predictions, references=references
        )
        results["bleu"] = bleu_results["bleu"]

        # METEOR score
        if self.meteor:
            scores = []
            for pred, ref in zip(predictions, references):
                score = self.meteor.compute(predictions=[pred], references=[ref[0]])
                scores.append(score["meteor"])
            results["meteor"] = np.mean(scores)

        return results

    def _clean_text(self, text: str) -> str:
        """Clean text for evaluation."""
        text = re.sub(r"\s+", " ", str(text))
        return text.strip().lower()