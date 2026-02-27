"""GRPO v2 trainer: pre-baked evidence, single generation, fixed grad accum.

Fixes three critical bugs from v1:
1. Grad accum: accumulates with _add_trees(), ONE optimizer.update() per step
2. Lazy evaluation: mx.eval() per-group inside accumulation loop
3. Token concat: eliminated — single completion_text with stored tokens

Architecture:
  BakedEvidence -> build_evidence_prompt -> LLM generates ONE response
  -> parse <reasoning>/<classification>/<confidence> -> binary reward -> GRPO
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .baked_data import BakedEvidence
from .episode_v2 import EpisodeV2
from .logger import AssessmentLogger, GRPOLogger, create_memory_watcher
from .mlx_backend import MLXBackend, compute_grpo_loss
from .prompt_v2 import build_evidence_prompt, parse_structured_output


class GRPOTrainerV2:
    """Single-generation GRPO trainer on pre-baked evidence.

    Args:
        backend: MLX Llama backend with LoRA.
        train_data: Pre-baked evidence for training.
        test_data: Pre-baked evidence for testing.
        output_dir: Directory for logs, checkpoints, episodes.
        group_size: Episodes per sequence for GRPO advantage estimation.
        grad_accum: Number of sequences to accumulate gradients over.
        lr: Learning rate for AdamW.
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon.
        warmup_ratio: Fraction of steps for LR warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        max_tokens: Max tokens per LLM generation.
        seed: Random seed.
        log_every: Log training metrics every N optimizer steps.
        save_every: Save adapter checkpoint every N optimizer steps.
        assess_every: Run test assessment every N optimizer steps.
    """

    def __init__(
        self,
        backend: MLXBackend,
        train_data: list[BakedEvidence],
        test_data: list[BakedEvidence],
        output_dir: Path,
        group_size: int = 4,
        grad_accum: int = 4,
        lr: float = 2e-5,
        beta: float = 0.05,
        clip_eps: float = 0.2,
        warmup_ratio: float = 0.05,
        max_grad_norm: float = 1.0,
        max_tokens: int = 300,
        seed: int = 42,
        log_every: int = 10,
        save_every: int = 1000,
        assess_every: int = 500,
    ):
        self.backend = backend
        self.train_data = train_data
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.group_size = group_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.beta = beta
        self.clip_eps = clip_eps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.max_tokens = max_tokens
        self.log_every = log_every
        self.save_every = save_every
        self.assess_every = assess_every
        self.rng = random.Random(seed)

        # Directories
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir = self.output_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = backend.get_optimizer(lr=lr)

        # Loggers
        self.grpo_logger = GRPOLogger(self.output_dir)
        self.assess_logger = AssessmentLogger(self.output_dir)

        # Tracking
        self._best_acc = 0.0
        self._global_step = 0

    def run_episode(self, evidence: BakedEvidence) -> EpisodeV2:
        """Run one single-generation episode from pre-baked evidence.

        Builds prompt with all evidence, generates ONE response, parses
        structured output, computes binary reward.
        """
        messages = build_evidence_prompt(evidence)
        prompt_tokens = self.backend.tokenize_chat(messages)

        completion_text = self.backend.generate(
            messages, max_tokens=self.max_tokens, temp=0.8
        )
        completion_tokens = self.backend.tokenizer.encode(completion_text)

        reasoning, prediction, confidence = parse_structured_output(completion_text)
        reward = 1.0 if prediction == evidence.label else 0.0

        return EpisodeV2(
            sequence_id=evidence.sequence_id,
            label=evidence.label,
            prompt_tokens=tuple(prompt_tokens),
            completion_text=completion_text,
            completion_tokens=tuple(completion_tokens),
            reasoning=reasoning,
            prediction=prediction,
            confidence=confidence,
            reward=reward,
        )

    def run_group(self, evidence: BakedEvidence) -> list[EpisodeV2]:
        """Run group_size episodes for a single sequence."""
        return [self.run_episode(evidence) for _ in range(self.group_size)]

    def compute_group_advantages(self, episodes: list[EpisodeV2]) -> list[float]:
        """Within-group advantage normalization."""
        if not episodes:
            return []
        rewards = [ep.reward for ep in episodes]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var_r ** 0.5

        if std_r < 1e-8:
            return [0.0] * len(episodes)
        return [(r - mean_r) / std_r for r in rewards]

    def _lr_schedule(self, step: int, total_steps: int) -> float:
        """Cosine decay with linear warmup."""
        warmup_steps = int(total_steps * self.warmup_ratio)
        if step < warmup_steps:
            return self.lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, epochs: int = 2) -> dict:
        """Full GRPO training loop with fixed gradient accumulation."""
        n_train = len(self.train_data)
        steps_per_epoch = n_train // self.grad_accum
        total_steps = steps_per_epoch * epochs

        print(f"\n{'='*60}", flush=True)
        print(f"  GRPO v2 Training: {epochs} epochs, {n_train} sequences", flush=True)
        print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}", flush=True)
        print(f"  Group size: {self.group_size}, Grad accum: {self.grad_accum}", flush=True)
        print(f"  LR: {self.lr}, Beta: {self.beta}, Clip: {self.clip_eps}", flush=True)
        print(f"  Max tokens: {self.max_tokens}", flush=True)
        print(f"  Output: {self.output_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)

        mem_watcher = create_memory_watcher(self.output_dir)
        mem_watcher.start()

        try:
            for epoch in range(epochs):
                self._train_epoch(epoch, epochs, steps_per_epoch, total_steps)
        finally:
            mem_watcher.stop()
            self.grpo_logger.close()
            self.assess_logger.close()

        self.backend.save_adapter(self.ckpt_dir / "adapter_final.npz")

        return {
            "epochs": epochs,
            "total_steps": self._global_step,
            "best_accuracy": self._best_acc,
        }

    def _train_epoch(
        self, epoch: int, total_epochs: int, steps_per_epoch: int, total_steps: int
    ) -> None:
        """Run one training epoch with proper gradient accumulation."""
        data = list(self.train_data)
        self.rng.shuffle(data)

        seq_idx = 0
        epoch_correct = 0
        epoch_total = 0

        for step_in_epoch in range(steps_per_epoch):
            self._global_step += 1
            current_lr = self._lr_schedule(self._global_step, total_steps)
            self.optimizer.learning_rate = current_lr

            step_result = self._train_step(data, seq_idx)
            seq_idx = step_result["next_seq_idx"]
            epoch_correct += step_result["n_correct"]
            epoch_total += step_result["n_total"]

            # Logging
            if step_result["n_groups"] > 0 and self._global_step % self.log_every == 0:
                acc = epoch_correct / max(epoch_total, 1)
                self.grpo_logger.log_step(
                    epoch=epoch,
                    step=self._global_step,
                    total_steps=total_steps,
                    reward_mean=step_result["reward_mean"],
                    reward_std=step_result["reward_std"],
                    accuracy=acc,
                    policy_loss=step_result["avg_policy_loss"],
                    kl_loss=step_result["avg_kl_loss"],
                    kl_mean=step_result["avg_kl_mean"],
                    total_loss=step_result["avg_loss"],
                    lr=current_lr,
                    gen_sec=step_result["gen_sec"],
                    update_sec=step_result["update_sec"],
                )

            # Periodic assessment
            if self._global_step % self.assess_every == 0:
                self._run_assessment(epoch, self._global_step)

            # Checkpoint
            if self._global_step % self.save_every == 0:
                path = self.ckpt_dir / f"adapter_step_{self._global_step}.npz"
                self.backend.save_adapter(path)

            # Log ALL episodes with step metadata
            if step_result.get("all_episodes"):
                ep_path = self.episode_dir / f"epoch_{epoch}.jsonl"
                with open(ep_path, "a") as fh:
                    for ep in step_result["all_episodes"]:
                        record = ep.to_dict()
                        record["step"] = self._global_step
                        record["epoch"] = epoch
                        fh.write(json.dumps(record) + "\n")

        print(
            f"  Epoch {epoch} complete: {epoch_correct}/{epoch_total} "
            f"({100 * epoch_correct / max(epoch_total, 1):.1f}%)",
            flush=True,
        )

    def _train_step(self, data: list[BakedEvidence], seq_idx: int) -> dict:
        """One optimizer step with proper gradient accumulation.

        FIX 1: Accumulates gradients with _add_trees(), ONE optimizer.update()
        FIX 2: mx.eval() per-group to avoid lazy evaluation bomb
        FIX 3: Uses stored prompt_tokens/completion_tokens (no re-tokenization)
        """
        # Phase 1: Generate episodes for grad_accum sequences
        gen_start = time.time()
        all_episodes: list[EpisodeV2] = []
        all_advantages: list[float] = []
        n_correct = 0
        n_total = 0

        for _ in range(self.grad_accum):
            if seq_idx >= len(data):
                break
            evidence = data[seq_idx]
            seq_idx += 1

            group = self.run_group(evidence)
            advantages = self.compute_group_advantages(group)
            all_episodes.extend(group)
            all_advantages.extend(advantages)

            for ep in group:
                n_total += 1
                if ep.is_correct:
                    n_correct += 1

        gen_sec = time.time() - gen_start

        if not all_episodes:
            return {
                "next_seq_idx": seq_idx, "n_correct": 0, "n_total": 0,
                "n_groups": 0, "gen_sec": gen_sec, "update_sec": 0,
                "reward_mean": 0, "reward_std": 0,
                "avg_loss": 0, "avg_policy_loss": 0,
                "avg_kl_loss": 0, "avg_kl_mean": 0,
            }

        # Phase 2: Compute old + ref log probs (before update, outside autograd)
        old_lps = []
        ref_lps = []
        for ep in all_episodes:
            prompt_tok = list(ep.prompt_tokens)
            comp_tok = list(ep.completion_tokens)
            old_lps.append(self.backend.compute_log_probs(prompt_tok, comp_tok))
            ref_lps.append(self.backend.compute_ref_log_probs(prompt_tok, comp_tok))
        # FIX 2: Evaluate log probs eagerly
        _mlx_eval(*old_lps, *ref_lps)

        # Phase 3+4: Accumulate gradients across groups, ONE optimizer update
        update_start = time.time()
        acc_grads = None
        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl_loss = 0.0
        total_kl_mean = 0.0
        n_groups = 0

        for g_start in range(0, len(all_episodes), self.group_size):
            g_end = min(g_start + self.group_size, len(all_episodes))
            g_episodes = all_episodes[g_start:g_end]
            g_advantages = all_advantages[g_start:g_end]
            g_old_lps = old_lps[g_start:g_end]
            g_ref_lps = ref_lps[g_start:g_end]

            if len(g_advantages) < 2:
                continue

            # Use stored tokens directly (FIX 3)
            prompt_tok = list(g_episodes[0].prompt_tokens)
            comp_tokens_list = [list(ep.completion_tokens) for ep in g_episodes]
            adv_arr = mx.array(g_advantages)
            old_lp_arr = mx.stack(g_old_lps)
            ref_lp_arr = mx.stack(g_ref_lps)

            def loss_fn(model):
                return compute_grpo_loss(
                    model, prompt_tok, comp_tokens_list,
                    adv_arr, old_lp_arr, ref_lp_arr,
                    beta=self.beta, clip_eps=self.clip_eps,
                )

            (loss, metrics), grads = nn.value_and_grad(
                self.backend.model, loss_fn
            )(self.backend.model)

            # FIX 2: Evaluate per-group to avoid lazy evaluation bomb
            _mlx_eval(loss, metrics["policy_loss"], metrics["kl_loss"], metrics["kl_mean"])
            _mlx_eval(*_collect_arrays(grads))

            total_loss += loss.item()
            total_policy_loss += metrics["policy_loss"].item()
            total_kl_loss += metrics["kl_loss"].item()
            total_kl_mean += metrics["kl_mean"].item()
            n_groups += 1

            # FIX 1: Accumulate gradients instead of separate updates
            if acc_grads is None:
                acc_grads = grads
            else:
                acc_grads = _add_trees(acc_grads, grads)

        if n_groups > 0 and acc_grads is not None:
            # Average accumulated gradients
            acc_grads = _scale_tree(acc_grads, mx.array(1.0 / n_groups))
            _mlx_eval(*_collect_arrays(acc_grads))

            # Clip and apply ONE update
            acc_grads, _ = _clip_grad_norm(acc_grads, self.max_grad_norm)
            self.optimizer.update(self.backend.model, acc_grads)
            _mlx_eval(self.backend.model.parameters(), self.optimizer.state)

        update_sec = time.time() - update_start

        # Compute reward stats
        rewards = [ep.reward for ep in all_episodes]
        r_mean = sum(rewards) / len(rewards) if rewards else 0
        r_std = (sum((r - r_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0

        return {
            "next_seq_idx": seq_idx,
            "n_correct": n_correct,
            "n_total": n_total,
            "n_groups": n_groups,
            "gen_sec": gen_sec,
            "update_sec": update_sec,
            "reward_mean": r_mean,
            "reward_std": r_std,
            "avg_loss": total_loss / max(n_groups, 1),
            "avg_policy_loss": total_policy_loss / max(n_groups, 1),
            "avg_kl_loss": total_kl_loss / max(n_groups, 1),
            "avg_kl_mean": total_kl_mean / max(n_groups, 1),
            "all_episodes": all_episodes,
        }

    def _run_assessment(self, epoch: int, step: int, max_samples: int = 200) -> None:
        """Run test-set assessment on a random subsample."""
        if len(self.test_data) > max_samples:
            samples = self.rng.sample(self.test_data, max_samples)
        else:
            samples = list(self.test_data)

        correct = 0
        predictions = []
        labels = []

        for evidence in samples:
            ep = self.run_episode(evidence)
            if ep.is_correct:
                correct += 1
            pred_val = 1 if ep.prediction == "coding" else 0
            label_val = 1 if evidence.label == "coding" else 0
            predictions.append(pred_val)
            labels.append(label_val)

        n = len(samples)
        acc = correct / max(n, 1)

        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = (tp * tn - fp * fn) / max(denom, 1e-8)

        self.assess_logger.log(
            epoch=epoch, step=step,
            accuracy=acc, precision=prec, recall=rec, f1=f1, mcc=mcc,
            n_correct=correct, n_total=n,
        )

        if acc > self._best_acc:
            self._best_acc = acc
            self.backend.save_adapter(self.ckpt_dir / "adapter_best.npz")
            print(f"  New best accuracy: {acc:.4f}", flush=True)


# ---------------------------------------------------------------------------
# MLX evaluation wrapper (not Python eval — MLX lazy graph materialization)
# ---------------------------------------------------------------------------

def _mlx_eval(*args):
    """Materialize MLX lazy computation graph. This is mlx.core.eval(), NOT Python eval()."""
    mx.eval(*args)


# ---------------------------------------------------------------------------
# Gradient tree utilities
# ---------------------------------------------------------------------------

def _add_trees(a, b):
    """Element-wise add two nested gradient trees."""
    if isinstance(a, mx.array):
        return a + b
    if isinstance(a, dict):
        return {k: _add_trees(a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [_add_trees(x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple):
        return tuple(_add_trees(x, y) for x, y in zip(a, b))
    return a


def _scale_tree(tree, scale):
    """Scale all mx.array leaves in a nested structure."""
    if isinstance(tree, mx.array):
        return tree * scale
    if isinstance(tree, dict):
        return {k: _scale_tree(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_scale_tree(v, scale) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_scale_tree(v, scale) for v in tree)
    return tree


def _collect_arrays(tree):
    """Recursively collect mx.array leaves from a nested structure."""
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _collect_arrays(v)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            yield from _collect_arrays(v)


def _clip_grad_norm(grads, max_norm: float):
    """Clip gradient norm across all parameters."""
    flat = mx.concatenate([g.reshape(-1) for g in _collect_arrays(grads)])
    total_norm = mx.sqrt(mx.sum(flat * flat))
    clip_coeff = mx.minimum(max_norm / (total_norm + 1e-6), mx.array(1.0))
    clipped = _scale_tree(grads, clip_coeff)
    return clipped, total_norm
