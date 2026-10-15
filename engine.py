import os
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class VerbatimCompactorEngine:
    def __init__(self, model_dir: str = "./onnx_compactor"):
        model_path = os.path.join(model_dir, "model_quantized.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing ONNX artifacts in {model_dir}")

        # Optimize for single-vCPU free instances
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"]
        )

        # Fast Rust Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128)

    def _extract_spans(self, text: str) -> list[tuple[int, int]]:
        """
        Splits text into (start_idx, end_idx) tuples.
        Captures sentence boundary punctuation alongside trailing whitespace/newlines.
        """
        pattern = re.compile(r'[^.!?\n]+(?:[.!?]+|\n+|$)(?:\s*)')
        return [(m.start(), m.end()) for m in pattern.finditer(text) if m.group().strip()]

    def _score_pairs(self, query: str, sentences: list[str], batch_size: int = 64) -> np.ndarray:
        scores = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            pairs = [(query, s) for s in batch]

            encoded = self.tokenizer.encode_batch(pairs)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

            ort_inputs = {
                self.session.get_inputs()[0].name: input_ids,
                self.session.get_inputs()[1].name: attention_mask
            }
            logits = self.session.run(None, ort_inputs)[0].flatten()
            scores.extend(logits)

        return np.array(scores)

    def compact(self, raw_context: str, current_query: str, target_reduction: float = 0.50) -> dict:
        if not raw_context.strip():
            return {"compacted_context": raw_context, "reduction_ratio": 0.0}

        spans = self._extract_spans(raw_context)
        if not spans:
            return {"compacted_context": raw_context, "reduction_ratio": 0.0}

        sentence_texts = [raw_context[s:e].strip() for s, e in spans]
        scores = self._score_pairs(current_query, sentence_texts)

        # Percentile cutoff: target_reduction = 0.60 drops lowest 60%
        threshold = np.percentile(scores, target_reduction * 100)

        # Splicing exact original character slices
        surviving_slices = [
            raw_context[start:end] 
            for (start, end), score in zip(spans, scores) 
            if score >= threshold
        ]

        compacted_text = "".join(surviving_slices)
        orig_len = len(raw_context)
        comp_len = len(compacted_text)

        return {
            "compacted_context": compacted_text,
            "original_chars": orig_len,
            "compacted_chars": comp_len,
            "reduction_ratio": round(1.0 - (comp_len / orig_len if orig_len > 0 else 0), 4),
            "sentences_total": len(spans),
            "sentences_kept": len(surviving_slices)
        }