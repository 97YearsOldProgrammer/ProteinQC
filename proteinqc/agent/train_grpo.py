"""GRPO training loop for ORF classification agent.

Phase 3C: Real LLM-driven GRPO with MLX Llama 3.1 8B LoRA.

Adapts both GeneT5 GRPO (GeneT5/lib/util/_grpo.py) and MLX-GRPO
(Doriandarko/MLX-GRPO) for multi-turn tool-calling episodes with
MLX-native LoRA training.

4-phase loop per optimizer step:
1. Generate: run G episodes per sequence (LLM multi-turn tool calls)
2. Score: binary reward (correct classification = 1.0)
3. Advantages: within-group normalization (mean/std from GeneT5)
4. Update: GRPO loss with KL penalty using nn.value_and_grad()
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .episode import Episode, ToolCall
from .logger import AssessmentLogger, GRPOLogger, create_memory_watcher
from .mlx_backend import MLXBackend, compute_grpo_loss
from .prompt import append_tool_result, build_initial_prompt, parse_tool_call
from .tool_schema import ToolExecutor


class GRPOTrainer:
    """GRPO training loop for ORF classification agent.

    Args:
        backend: MLX Llama backend with LoRA.
        tool_executor: Configured ToolExecutor for biological tools.
        train_data: List of {"sequence_id", "sequence", "label"} for training.
        test_data: List of {"sequence_id", "sequence", "label"} for testing.
        output_dir: Directory for logs, checkpoints, episodes.
        group_size: Number of episodes per sequence for GRPO.
        grad_accum: Gradient accumulation steps.
        max_tools_per_episode: Max tool calls before forced classification.
        lr: Learning rate for AdamW.
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon.
        warmup_ratio: Fraction of steps for LR warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        seed: Random seed.
        log_every: Log training metrics every N optimizer steps.
        save_every: Save adapter checkpoint every N optimizer steps.
        assess_every: Run test assessment every N optimizer steps.
    """

    def __init__(
        self,
        backend: MLXBackend,
        tool_executor: ToolExecutor,
        train_data: list[dict],
        test_data: list[dict],
        output_dir: Path,
        group_size: int = 4,
        grad_accum: int = 4,
        max_tools_per_episode: int = 3,
        lr: float = 2e-5,
        beta: float = 0.05,
        clip_eps: float = 0.2,
        warmup_ratio: float = 0.05,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        log_every: int = 10,
        save_every: int = 1000,
        assess_every: int = 500,
    ):
        self.backend = backend
        self.tool_executor = tool_executor
        self.train_data = train_data
        self.test_data = test_data
        self.output_dir = Path(output_dir)
        self.group_size = group_size
        self.grad_accum = grad_accum
        self.max_tools = max_tools_per_episode
        self.lr = lr
        self.beta = beta
        self.clip_eps = clip_eps
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
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

    def run_episode(self, sequence: str, ground_truth: str, seq_id: str = "") -> Episode:
        """Run one multi-turn tool-calling episode with real LLM.

        Loop: prompt -> LLM generates tool call -> execute tool -> append result
        Until: LLM calls classify OR max_tools reached.
        """
        messages = build_initial_prompt(sequence)
        calls: list[ToolCall] = []
        raw_responses: list[str] = []
        prediction = None
        confidence = None

        for _turn in range(self.max_tools + 1):
            response = self.backend.generate(messages, max_tokens=200, temp=0.8)
            raw_responses.append(response)

            parsed = parse_tool_call(response)
            if parsed is None:
                prediction = "noncoding"
                confidence = 0.1
                break

            tool_name = parsed["name"]
            tool_args = parsed.get("arguments", {})

            if tool_name == "classify":
                prediction = tool_args.get("label", "noncoding")
                confidence = tool_args.get("confidence", 0.5)
                break

            # Auto-inject sequence for evidence tools
            if "sequence" not in tool_args:
                tool_args["sequence"] = sequence

            result = self.tool_executor.execute(tool_name, tool_args)
            calls.append(ToolCall(name=tool_name, arguments=tool_args, result=result))

            # Append assistant response + tool result to conversation
            messages.append({"role": "assistant", "content": response})
            messages = append_tool_result(messages, tool_name, result)

        if prediction is None:
            prediction = "noncoding"
            confidence = 0.1

        reward = 1.0 if prediction == ground_truth else 0.0

        return Episode(
            transcript_id=seq_id,
            orf_sequence=sequence,
            ground_truth=ground_truth,
            tool_calls=tuple(calls),
            prediction=prediction,
            confidence=confidence,
            reward=reward,
            raw_responses=tuple(raw_responses),
        )

    def run_group(self, item: dict) -> list[Episode]:
        """Run group_size episodes for a single sequence."""
        return [
            self.run_episode(item["sequence"], item["label"], item.get("sequence_id", ""))
            for _ in range(self.group_size)
        ]

    def compute_group_advantages(self, episodes: list[Episode]) -> list[float]:
        """Within-group advantage normalization (GeneT5 pattern)."""
        if not episodes:
            return []
        rewards = [ep.reward for ep in episodes]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = var_r ** 0.5

        if std_r < 1e-8:
            return [0.0] * len(episodes)
        return [(r - mean_r) / std_r for r in rewards]

    def _get_episode_tokens(self, episode: Episode) -> tuple[list[int], list[int]]:
        """Extract prompt tokens and completion tokens from an episode.

        Uses the raw LLM response texts stored in the episode to ensure
        log-prob computation matches actual generated tokens.
        """
        messages = build_initial_prompt(episode.orf_sequence)
        prompt_tokens = self.backend.tokenize_chat(messages)

        # Use raw response texts for accurate log-prob computation
        completion_text = "\n".join(episode.raw_responses)
        completion_tokens = self.backend.tokenizer.encode(completion_text)

        return prompt_tokens, completion_tokens

    def _lr_schedule(self, step: int, total_steps: int) -> float:
        """Cosine decay with linear warmup."""
        warmup_steps = int(total_steps * self.warmup_ratio)
        if step < warmup_steps:
            return self.lr * step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def train(self, epochs: int = 3) -> dict:
        """Full GRPO training loop."""
        n_train = len(self.train_data)
        steps_per_epoch = n_train // self.grad_accum
        total_steps = steps_per_epoch * epochs

        print(f"\n{'='*60}", flush=True)
        print(f"  GRPO Training: {epochs} epochs, {n_train} sequences", flush=True)
        print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}", flush=True)
        print(f"  Group size: {self.group_size}, Grad accum: {self.grad_accum}", flush=True)
        print(f"  LR: {self.lr}, Beta: {self.beta}, Clip: {self.clip_eps}", flush=True)
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
        """Run one training epoch."""
        data = list(self.train_data)
        self.rng.shuffle(data)

        seq_idx = 0
        epoch_correct = 0
        epoch_total = 0

        for step_in_epoch in range(steps_per_epoch):
            self._global_step += 1
            current_lr = self._lr_schedule(self._global_step, total_steps)
            self.optimizer.learning_rate = current_lr

            # Phase 1: Generate episodes for grad_accum sequences
            gen_start = time.time()
            step_episodes = []
            step_advantages = []

            for _ in range(self.grad_accum):
                if seq_idx >= len(data):
                    break
                item = data[seq_idx]
                seq_idx += 1

                group = self.run_group(item)
                advantages = self.compute_group_advantages(group)
                step_episodes.extend(group)
                step_advantages.extend(advantages)

                for ep in group:
                    epoch_total += 1
                    if ep.is_correct:
                        epoch_correct += 1

            gen_sec = time.time() - gen_start

            if not step_episodes:
                continue

            # Phase 2: Compute old + ref log probs (before update, outside autograd)
            old_lps = []
            ref_lps = []
            token_pairs = []
            for ep in step_episodes:
                prompt_tok, comp_tok = self._get_episode_tokens(ep)
                token_pairs.append((prompt_tok, comp_tok))
                old_lps.append(self.backend.compute_log_probs(prompt_tok, comp_tok))
                ref_lps.append(self.backend.compute_ref_log_probs(prompt_tok, comp_tok))
            mx.eval(*old_lps, *ref_lps)

            # Phase 3+4: GRPO update with gradient computation
            update_start = time.time()
            total_loss = 0.0
            total_policy_loss = 0.0
            total_kl_loss = 0.0
            total_kl_mean = 0.0
            n_groups = 0

            for g_start in range(0, len(step_episodes), self.group_size):
                g_end = min(g_start + self.group_size, len(step_episodes))
                g_advantages = step_advantages[g_start:g_end]
                g_old_lps = old_lps[g_start:g_end]
                g_ref_lps = ref_lps[g_start:g_end]
                g_token_pairs = token_pairs[g_start:g_end]

                if len(g_advantages) < 2:
                    continue

                prompt_tok = g_token_pairs[0][0]
                comp_tokens_list = [tp[1] for tp in g_token_pairs]
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

                grads, _ = _clip_grad_norm(grads, self.max_grad_norm)

                self.optimizer.update(self.backend.model, grads)
                mx.eval(self.backend.model.parameters(), self.optimizer.state)

                # Extract metrics after eval (C1 fix: .item() outside autograd)
                mx.eval(loss, metrics["policy_loss"], metrics["kl_loss"], metrics["kl_mean"])
                total_loss += loss.item()
                total_policy_loss += metrics["policy_loss"].item()
                total_kl_loss += metrics["kl_loss"].item()
                total_kl_mean += metrics["kl_mean"].item()
                n_groups += 1

            update_sec = time.time() - update_start

            # Logging
            if n_groups > 0 and self._global_step % self.log_every == 0:
                rewards = [ep.reward for ep in step_episodes]
                r_mean = sum(rewards) / len(rewards)
                r_std = (sum((r - r_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
                acc = epoch_correct / max(epoch_total, 1)

                self.grpo_logger.log_step(
                    epoch=epoch, step=self._global_step, total_steps=total_steps,
                    reward_mean=r_mean, reward_std=r_std, accuracy=acc,
                    policy_loss=total_policy_loss / n_groups,
                    kl_loss=total_kl_loss / n_groups,
                    kl_mean=total_kl_mean / n_groups,
                    total_loss=total_loss / n_groups,
                    lr=current_lr, gen_sec=gen_sec, update_sec=update_sec,
                )

            # Periodic assessment
            if self._global_step % self.assess_every == 0:
                self._run_assessment(epoch, self._global_step)

            # Checkpoint
            if self._global_step % self.save_every == 0:
                path = self.ckpt_dir / f"adapter_step_{self._global_step}.npz"
                self.backend.save_adapter(path)

            # Save sample episodes (first 50 steps of each epoch)
            if step_in_epoch < 50:
                ep_path = self.episode_dir / f"epoch_{epoch}_sample.jsonl"
                with open(ep_path, "a") as fh:
                    for ep in step_episodes[:self.group_size]:
                        fh.write(json.dumps(ep.to_dict()) + "\n")

        print(
            f"  Epoch {epoch} complete: {epoch_correct}/{epoch_total} "
            f"({100 * epoch_correct / max(epoch_total, 1):.1f}%)",
            flush=True,
        )

    def _run_assessment(self, epoch: int, step: int, max_samples: int = 200) -> None:
        """Run test-set assessment on a random subsample."""
        if len(self.test_data) > max_samples:
            samples = self.rng.sample(self.test_data, max_samples)
        else:
            samples = self.test_data
        correct = 0
        predictions = []
        labels = []

        for item in samples:
            ep = self.run_episode(
                item["sequence"], item["label"], item.get("sequence_id", "")
            )
            if ep.is_correct:
                correct += 1
            pred_val = 1 if ep.prediction == "coding" else 0
            label_val = 1 if item["label"] == "coding" else 0
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
# Gradient clipping (MLX-native)
# ---------------------------------------------------------------------------

def _clip_grad_norm(grads, max_norm: float):
    """Clip gradient norm across all parameters."""
    flat = mx.concatenate([g.reshape(-1) for g in _collect_arrays(grads)])
    total_norm = mx.sqrt(mx.sum(flat * flat))
    clip_coeff = mx.minimum(max_norm / (total_norm + 1e-6), mx.array(1.0))
    clipped = _scale_tree(grads, clip_coeff)
    return clipped, total_norm


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
