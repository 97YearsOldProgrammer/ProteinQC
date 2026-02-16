"""Model components for ProteinQC.

CaLMEncoder: Pure PyTorch CaLM BERT encoder (Pre-LN, RoPE, SDPA).
Classification heads: Linear, MLP, Gated for binary RNA classification.
norm_convert: LayerNorm -> RMSNorm conversion utility.
"""
