"""MLX Llama 3.1 8B backend with LoRA for GRPO training.

Wraps mlx-lm for text generation and log-probability extraction.
Adapted from MLX-GRPO (github.com/Doriandarko/MLX-GRPO) patterns.

Key MLX APIs used:
- mlx_lm.load / mlx_lm.generate for model loading and sampling
- nn.value_and_grad for autodiff through LoRA parameters
- nn.log_softmax for token-level log probability computation
- linear_to_lora_layers for LoRA adapter injection
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers


class MLXBackend:
    """MLX Llama 3.1 8B backend with LoRA for GRPO training.

    Args:
        model_name: HuggingFace model ID for mlx-lm (auto-downloads).
        lora_rank: LoRA adapter rank.
        lora_layers: Number of transformer layers to apply LoRA to (from the end).
        lora_dropout: Dropout rate for LoRA layers.
        lora_scale: LoRA scaling factor (alpha/rank).
    """

    def __init__(
        self,
        model_name: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        lora_rank: int = 16,
        lora_layers: int = 8,
        lora_dropout: float = 0.0,
        lora_scale: float = 20.0,
    ):
        print(f"  Loading policy model: {model_name}", flush=True)
        self.model, self.tokenizer = mlx_load(model_name)

        # Freeze base, apply LoRA to last N layers
        self.model.freeze()
        linear_to_lora_layers(
            self.model,
            num_layers=lora_layers,
            config={"rank": lora_rank, "dropout": lora_dropout, "scale": lora_scale},
        )

        trainable_flat = nn.utils.tree_flatten(self.model.trainable_parameters())
        n_trainable = sum(v.size for _, v in trainable_flat)
        all_flat = nn.utils.tree_flatten(self.model.parameters())
        n_total = sum(v.size for _, v in all_flat)
        print(
            f"  LoRA applied: {n_trainable:,} trainable / {n_total:,} total params "
            f"({100 * n_trainable / n_total:.2f}%)",
            flush=True,
        )

        # Reference model (frozen copy for KL divergence)
        print(f"  Loading reference model: {model_name}", flush=True)
        self.ref_model, _ = mlx_load(model_name)
        self.ref_model.freeze()

        self._model_name = model_name
        self._lora_rank = lora_rank
        self._lora_layers = lora_layers

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temp: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Generate a completion from chat messages.

        Args:
            messages: List of {"role": ..., "content": ...} chat messages.
            max_tokens: Maximum tokens to generate.
            temp: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text string.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _format_messages_fallback(messages)

        sampler = make_sampler(temp=temp, top_p=top_p)
        response = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return response

    def tokenize_chat(self, messages: list[dict]) -> list[int]:
        """Convert chat messages to token IDs."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_str = _format_messages_fallback(messages)
        return self.tokenizer.encode(prompt_str)

    def compute_log_probs(
        self, prompt_tokens: list[int], completion_tokens: list[int]
    ) -> mx.array:
        """Compute sum(log P(completion | prompt)) via teacher-forced forward pass.

        Concatenates prompt + completion tokens, runs forward pass, then
        extracts log probabilities at completion token positions only.

        Args:
            prompt_tokens: Token IDs for the prompt.
            completion_tokens: Token IDs for the completion to score.

        Returns:
            Scalar mx.array: sum of log probabilities over completion tokens.
        """
        return _compute_log_probs_impl(
            self.model, prompt_tokens, completion_tokens
        )

    def compute_ref_log_probs(
        self, prompt_tokens: list[int], completion_tokens: list[int]
    ) -> mx.array:
        """Same as compute_log_probs but using the frozen reference model."""
        return _compute_log_probs_impl(
            self.ref_model, prompt_tokens, completion_tokens
        )

    def get_optimizer(self, lr: float = 2e-5) -> optim.Adam:
        """Create AdamW optimizer for LoRA parameters."""
        return optim.AdamW(learning_rate=lr)

    def save_adapter(self, path: Path) -> None:
        """Save LoRA adapter weights to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat_weights = dict(nn.utils.tree_flatten(self.model.trainable_parameters()))
        mx.savez(str(path), **flat_weights)
        print(f"  Adapter saved: {path}", flush=True)

    def load_adapter(self, path: Path) -> None:
        """Load LoRA adapter weights from .npz file."""
        path = Path(path)
        weights = mx.load(str(path))
        self.model.load_weights(list(weights.items()), strict=False)
        print(f"  Adapter loaded: {path}", flush=True)


# ---------------------------------------------------------------------------
# GRPO loss computation (MLX-native)
# ---------------------------------------------------------------------------

def compute_grpo_loss(
    model,
    prompt_tokens: list[int],
    completion_tokens_list: list[list[int]],
    advantages: mx.array,
    old_log_probs: mx.array,
    ref_log_probs: mx.array,
    beta: float = 0.05,
    clip_eps: float = 0.2,
) -> tuple[mx.array, dict]:
    """Compute GRPO loss for a group of completions.

    Args:
        model: Policy model (with LoRA).
        prompt_tokens: Shared prompt token IDs.
        completion_tokens_list: List of completion token ID sequences (one per group member).
        advantages: Per-completion advantage values [G].
        old_log_probs: Log probs from the generation step [G].
        ref_log_probs: Pre-computed ref model log probs [G] (outside autograd scope).
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon.

    Returns:
        (loss, metrics_dict) where metrics_dict has policy_loss, kl_loss, kl_mean
        as mx.array (call .item() after mx.eval).
    """
    current_lps = []
    for comp_tokens in completion_tokens_list:
        current_lps.append(
            _compute_log_probs_impl(model, prompt_tokens, comp_tokens)
        )

    current_lp = mx.stack(current_lps)
    ref_lp = ref_log_probs

    # PPO-style clipped ratio
    ratio = mx.exp(current_lp - old_log_probs)
    clipped = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    policy_reward = mx.minimum(ratio * advantages, clipped * advantages)

    # Reverse KL estimator: r - log(r) - 1 where r = ref/policy
    log_ratio_kl = ref_lp - current_lp
    kl = mx.exp(log_ratio_kl) - log_ratio_kl - 1.0

    policy_loss = -mx.mean(policy_reward)
    kl_loss = beta * mx.mean(kl)
    loss = policy_loss + kl_loss

    metrics = {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "kl_mean": mx.mean(kl),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_log_probs_impl(
    model, prompt_tokens: list[int], completion_tokens: list[int]
) -> mx.array:
    """Teacher-forced log prob computation for a single prompt+completion."""
    full_tokens = prompt_tokens + completion_tokens
    input_ids = mx.array(full_tokens, dtype=mx.int32)[None, :]  # [1, seq_len]

    logits = model(input_ids)  # [1, seq_len, vocab]
    log_probs = nn.log_softmax(logits, axis=-1)

    # Vectorized gather: positions that predict completion tokens
    prompt_len = len(prompt_tokens)
    positions = mx.arange(prompt_len - 1, prompt_len - 1 + len(completion_tokens))
    target_ids = mx.array(completion_tokens, dtype=mx.int32)
    selected = log_probs[0, positions, target_ids]  # [len(completion_tokens)]

    return mx.sum(selected)


def _format_messages_fallback(messages: list[dict]) -> str:
    """Simple fallback if tokenizer lacks apply_chat_template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
        elif role == "user":
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
        elif role == "assistant":
            parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)
