"""Rich terminal UI for ProteinQC agent.

Claude Code-style REPL with startup banner, tool spinners,
FASTA loading, slash commands, and Rich-rendered output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

console = Console()

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("proteinqc")
except Exception:
    __version__ = "0.3.0"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Mutable state for the interactive session."""

    sequences: dict[str, str] = field(default_factory=dict)
    query_count: int = 0
    tool_calls: int = 0
    start_time: float = field(default_factory=time.time)
    backend: str = "mlx"
    model_id: str = ""
    device: str = "cpu"

    @property
    def uptime(self) -> str:
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Tool definitions: light vs heavy
# ---------------------------------------------------------------------------

TOOL_INFO: dict[str, tuple[str, str, str, str]] = {
    # name -> (status, weight, description, citation)
    "translate_dna": ("ready", "", "Translate DNA to protein in all 6 reading frames", ""),
    "gc_content": ("ready", "", "Compute GC% and nucleotide composition", ""),
    "kozak_score": ("ready", "", "Score Kozak consensus strength around start codons", ""),
    "scan_orfs": ("ready", "", "Find all open reading frames with start/stop positions", ""),
    "calm_score": (
        "lazy",
        "86M params",
        "Coding probability via codon-level BERT encoder",
        "Outeiral, C. & Deane, C.M. Nat. Mach. Intell. 6, 170\u2013179 (2024)",
    ),
    "pfam_scan": (
        "lazy",
        "4.2GB HMM",
        "Protein domain search against Pfam-A HMM library",
        "Paysan-Lafosse, T. et al. Nucleic Acids Res. 53, D523\u2013D532 (2025)",
    ),
}

LIGHT_TOOLS = [k for k, v in TOOL_INFO.items() if v[0] == "ready"]
HEAVY_TOOLS = {k: v[1] for k, v in TOOL_INFO.items() if v[0] == "lazy"}


def _detect_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu"


def _detect_memory() -> str:
    try:
        import psutil

        mem_gb = psutil.virtual_memory().total / (1024**3)
        return f"{mem_gb:.0f}GB"
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Spinner wrapping for heavy tools
# ---------------------------------------------------------------------------


def wrap_tool_with_spinner(tool: Any, state: SessionState) -> None:
    """Monkey-patch a smolagents Tool.forward() to show a Rich spinner."""
    original_forward = tool.forward

    def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        state.tool_calls += 1
        label = f"[bold cyan]{tool.name}[/] running…"
        with console.status(label, spinner="dots"):
            t0 = time.time()
            result = original_forward(*args, **kwargs)
            elapsed = time.time() - t0
        console.print(
            f"  [dim]{tool.name}[/dim] completed in [green]{elapsed:.1f}s[/green]"
        )
        return result

    tool.forward = _wrapped_forward


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


_MASCOT = (
    "  .~.  \n"
    " (o.O) \n"
    "  |P/  \n"
    "   ~   \n"
    "  d b  "
)


def render_banner(state: SessionState, agent: Any | None = None) -> Panel:
    """Build the startup banner panel with mascot on the right."""
    lines: list[str] = []

    # Title
    lines.append(f"[bold white]ProteinQC v{__version__}[/]")

    # Backend / model
    model_short = state.model_id.split("/")[-1] if state.model_id else "none"
    lines.append(f"Backend: [cyan]{state.backend}[/]  |  Model: [cyan]{model_short}[/]")

    # Device
    device_str = state.device
    mem = _detect_memory()
    lines.append(f"Device: [yellow]{device_str}[/], {mem}")

    # Tools
    ready = "  ".join(f"[green]{t}[/]" for t in LIGHT_TOOLS)
    lazy = "  ".join(
        f"[dim]{t} (lazy, {info})[/dim]" for t, info in HEAVY_TOOLS.items()
    )
    lines.append(f"Tools: {ready}")
    lines.append(f"       {lazy}")

    info_text = Text.from_markup("\n".join(lines))
    mascot_text = Text(_MASCOT, style="dim")

    layout = Table(show_header=False, show_edge=False, box=None, padding=0)
    layout.add_column(ratio=1)
    layout.add_column(width=9, justify="right")
    layout.add_row(info_text, mascot_text)

    return Panel(layout, border_style="blue", padding=(0, 1))


def render_tool_table() -> Table:
    """Render tool status as a Rich Table."""
    table = Table(title="Tools", show_lines=False, padding=(0, 1))
    table.add_column("Tool", style="bold")
    table.add_column("Status", min_width=6)
    table.add_column("Description")

    for name, (status, weight, desc, cite) in TOOL_INFO.items():
        if status == "ready":
            status_str = "[green]ready[/]"
        else:
            status_str = f"[yellow]lazy[/] [dim]({weight})[/dim]"
        desc_text = desc
        if cite:
            desc_text += f"\n[italic]{cite}[/italic]"
        table.add_row(name, status_str, desc_text)
    return table


def render_score(score: float) -> Text:
    """Color-code a 0-1 probability score."""
    if score > 0.7:
        style = "bold green"
    elif score > 0.3:
        style = "bold yellow"
    else:
        style = "bold red"
    return Text(f"{score:.4f}", style=style)


def render_result(result: Any) -> None:
    """Auto-dispatch result rendering."""
    text = str(result)

    # Numeric score
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            console.print("Score: ", end="")
            console.print(render_score(val))
            return
    except ValueError:
        pass

    # ORF table
    if "ORF" in text and "frame=" in text:
        _render_orf_table(text)
        return

    # Default: markdown
    console.print(Markdown(text))


def _render_orf_table(text: str) -> None:
    """Parse ORF scan output into a Rich Table."""
    table = Table(title="ORF Candidates", show_lines=False, padding=(0, 1))
    table.add_column("#", style="dim")
    table.add_column("Frame")
    table.add_column("Start", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("Codons", justify="right", style="bold")

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = {}
        for token in line.split():
            if "=" in token:
                k, v = token.split("=", 1)
                parts[k] = v
        idx = line.split(".")[0].strip() if "." in line else "?"
        table.add_row(
            idx,
            parts.get("frame", "?"),
            parts.get("start", "?"),
            parts.get("stop", "?"),
            parts.get("codons", "?"),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# FASTA loading
# ---------------------------------------------------------------------------


def load_fasta(path: str, state: SessionState) -> None:
    """Load sequences from a FASTA file into session state."""
    fpath = Path(path).expanduser()
    if not fpath.exists():
        # Try relative to project root
        fpath = _PROJECT_ROOT / path
    if not fpath.exists():
        console.print(f"[red]File not found:[/] {path}")
        return

    name, seq_parts = None, []
    count = 0
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name and seq_parts:
                    state.sequences[name] = "".join(seq_parts)
                    count += 1
                name = line[1:].split()[0]
                seq_parts = []
            elif name:
                seq_parts.append(line)
        if name and seq_parts:
            state.sequences[name] = "".join(seq_parts)
            count += 1

    console.print(
        f"[green]Loaded {count} sequence(s)[/] from [cyan]{fpath.name}[/]  "
        f"({len(state.sequences)} total in session)"
    )


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    "/help": "Show this command list",
    "/tools": "Tool status table (loaded / lazy / unavailable)",
    "/load <path>": "Load FASTA file into session",
    "/seq <name>": "Preview loaded sequence (first 200 bp)",
    "/seqs": "List all loaded sequences",
    "/status": "Session stats (uptime, queries, tools, sequences)",
    "/clear": "Clear terminal screen",
    "/model": "Backend and model info",
    "/quit": "Exit",
}


def _handle_slash(cmd: str, state: SessionState) -> bool:
    """Handle a slash command. Returns True if handled."""
    parts = cmd.strip().split(maxsplit=1)
    verb = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if verb == "/help":
        table = Table(show_header=False, padding=(0, 2), show_edge=False)
        table.add_column(style="bold cyan")
        table.add_column()
        for k, v in COMMANDS.items():
            table.add_row(k, v)
        console.print(table)
        return True

    if verb == "/tools":
        console.print(render_tool_table())
        return True

    if verb == "/load":
        if not arg:
            console.print("[red]Usage:[/] /load <path-to-fasta>")
        else:
            load_fasta(arg, state)
        return True

    if verb == "/seq":
        if not arg:
            console.print("[red]Usage:[/] /seq <sequence-name>")
        elif arg not in state.sequences:
            console.print(f"[red]Unknown sequence:[/] {arg}")
            console.print(f"[dim]Available: {', '.join(state.sequences) or 'none'}[/dim]")
        else:
            seq = state.sequences[arg]
            preview = seq[:200] + ("…" if len(seq) > 200 else "")
            console.print(f"[bold]{arg}[/] ({len(seq):,} bp)")
            console.print(f"[dim]{preview}[/dim]")
        return True

    if verb == "/seqs":
        if not state.sequences:
            console.print("[dim]No sequences loaded. Use /load <path>[/dim]")
        else:
            table = Table(
                title=f"Loaded Sequences ({len(state.sequences)})",
                show_lines=False,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", justify="right", min_width=3)
            table.add_column("Name", style="cyan")
            table.add_column("Length", justify="right", style="bold")
            for i, (name, seq) in enumerate(state.sequences.items(), 1):
                table.add_row(str(i), name, f"{len(seq):,} bp")
            console.print(table)
        return True

    if verb == "/status":
        console.print(f"  Uptime:      [cyan]{state.uptime}[/]")
        console.print(f"  Queries:     [cyan]{state.query_count}[/]")
        console.print(f"  Tool calls:  [cyan]{state.tool_calls}[/]")
        console.print(f"  Sequences:   [cyan]{len(state.sequences)}[/]")
        return True

    if verb == "/clear":
        console.clear()
        return True

    if verb == "/model":
        model_short = state.model_id.split("/")[-1] if state.model_id else "none"
        console.print(f"  Backend: [cyan]{state.backend}[/]")
        console.print(f"  Model:   [cyan]{model_short}[/]")
        console.print(f"  Device:  [yellow]{state.device}[/]")
        return True

    if verb in ("/quit", "/exit", "/q"):
        raise SystemExit(0)

    return False


# ---------------------------------------------------------------------------
# Status bar (persistent bottom toolbar)
# ---------------------------------------------------------------------------


def _get_memory_stats() -> tuple[int, int, int]:
    """Return (rss_mb, gpu_allocated_mb, gpu_total_mb)."""
    rss = 0
    try:
        import psutil

        rss = int(psutil.Process().memory_info().rss / (1024**2))
    except Exception:
        pass

    gpu_alloc, gpu_total = 0, 0
    try:
        import torch

        if torch.backends.mps.is_available():
            gpu_alloc = int(torch.mps.current_allocated_memory() / (1024**2))
            gpu_total = int(torch.mps.driver_allocated_memory() / (1024**2))
        elif torch.cuda.is_available():
            gpu_alloc = int(torch.cuda.memory_allocated() / (1024**2))
            gpu_total = int(torch.cuda.memory_reserved() / (1024**2))
    except Exception:
        pass

    return rss, gpu_alloc, gpu_total


def _build_status_bar(state: SessionState):
    """Return a callable for prompt_toolkit's bottom_toolbar."""
    try:
        from prompt_toolkit.formatted_text import HTML
    except ImportError:
        return None

    def _toolbar():
        parts: list[str] = []
        # Device
        parts.append(state.device)
        # Memory
        rss, gpu_alloc, gpu_total = _get_memory_stats()
        if rss:
            parts.append(f"RAM {rss} MB")
        if gpu_total:
            parts.append(f"GPU {gpu_alloc}/{gpu_total} MB")
        elif gpu_alloc:
            parts.append(f"GPU {gpu_alloc} MB")
        return HTML(" | ".join(parts))

    return _toolbar


# ---------------------------------------------------------------------------
# Prompt input (prompt_toolkit with fallback)
# ---------------------------------------------------------------------------


def _build_prompt_session(state: SessionState):
    """Build a prompt_toolkit PromptSession, or None if unavailable."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        history_path = Path.home() / ".cache" / "proteinqc" / "repl_history"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        completer = WordCompleter(
            list(COMMANDS.keys()) + ["/quit", "/exit"],
            sentence=True,
        )
        toolbar_style = Style.from_dict({
            "bottom-toolbar": "noreverse #888888",
        })
        return PromptSession(
            history=FileHistory(str(history_path)),
            completer=completer,
            bottom_toolbar=_build_status_bar(state),
            style=toolbar_style,
        )
    except ImportError:
        return None


def _get_input(session) -> str:
    """Get user input via prompt_toolkit or builtin input()."""
    if session is not None:
        try:
            from prompt_toolkit.formatted_text import HTML

            return session.prompt(HTML("<b>&gt;</b> "))
        except ImportError:
            return session.prompt("> ")
    return input("> ")


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def run_repl(
    agent: Any | None = None,
    backend: str = "mlx",
    model_id: str = "",
    **_kwargs: Any,
) -> None:
    """Run the interactive Rich REPL.

    Args:
        agent: A smolagents CodeAgent (optional). If None, only slash
               commands and direct tool calls work.
        backend: LLM backend name for display.
        model_id: Model identifier for display.
    """
    state = SessionState(
        backend=backend,
        model_id=model_id,
        device=_detect_device(),
    )

    # Wrap heavy tools with spinners
    if agent is not None:
        try:
            # Share our console with smolagents' logger
            if hasattr(agent, "logger") and hasattr(agent.logger, "console"):
                agent.logger.console = console
        except Exception:
            pass

        for tool in getattr(agent, "tools", {}).values():
            if hasattr(tool, "forward") and getattr(tool, "name", "") in HEAVY_TOOLS:
                wrap_tool_with_spinner(tool, state)

    # Banner
    console.print()
    console.print(render_banner(state, agent))
    console.print("[dim]Type /help for commands, or enter a query.[/dim]\n")

    prompt_session = _build_prompt_session(state)

    while True:
        try:
            raw = _get_input(prompt_session).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break
        except SystemExit:
            console.print("[dim]Bye![/dim]")
            break

        if not raw:
            continue

        # Slash commands
        if raw.startswith("/"):
            try:
                if _handle_slash(raw, state):
                    continue
            except SystemExit:
                console.print("[dim]Bye![/dim]")
                break
            console.print(f"[red]Unknown command:[/] {raw.split()[0]}  (try /help)")
            continue

        # Agent query
        state.query_count += 1

        if agent is None:
            console.print(
                "[yellow]No agent loaded.[/] "
                "Start with [bold]run-agent[/] or pass --backend to load a model."
            )
            continue

        try:
            result = agent.run(raw)
            console.print()
            render_result(result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Query interrupted.[/]")
        except Exception as exc:
            console.print(
                Panel(
                    f"[red]{type(exc).__name__}[/]: {exc}",
                    title="Error",
                    border_style="red",
                )
            )

    console.print()
