"""ProteinQC RL agent: tool-calling agent for ORF classification.

Phase 3B provides the infrastructure:
  - Episode/ToolCall data structures for tracking agent behavior
  - TOOL_SCHEMAS defining the LLM function-calling interface
  - ToolExecutor dispatching tool names to real biological tools
  - DataBuilder for creating training data from GENCODE

Phase 3C adds:
  - MLX Llama 3.1 8B as the policy backbone (mlx_backend)
  - GRPO training with real LLM-driven episodes (train_grpo)
  - Prompt templates for tool-calling reasoning (prompt)
  - RNA Challenge data loader (rnachallenge_loader)
  - GeneT5-style logging and memory monitoring (logger)

Phase 3D (v2) adds:
  - Pre-baked evidence: offline tool execution (baked_data)
  - Single-generation GRPO: constrained structured output (prompt_v2)
  - Simplified episode: no multi-turn token concat (episode_v2)
  - Fixed trainer: proper grad accum + eager eval (train_grpo_v2)
"""

from .episode import Episode, ToolCall
from .tool_schema import TOOL_SCHEMAS, ToolExecutor

# v2 imports (pre-baked evidence pipeline)
from .baked_data import BakedEvidence, load_baked_evidence, split_baked
from .episode_v2 import EpisodeV2

__all__ = [
    "Episode", "ToolCall", "TOOL_SCHEMAS", "ToolExecutor",
    "BakedEvidence", "load_baked_evidence", "split_baked", "EpisodeV2",
]
