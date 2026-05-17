from __future__ import annotations

import sys

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeRemainingColumn,
    )

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class ProgressDisplay:
    """Terminal progress display for the Visage pipeline.

    Uses rich for a polished progress bar when available,
    falls back to simple terminal output otherwise.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self._rich_progress: Progress | None = None
        self._rich_task: TaskID | None = None
        self._console: Console | None = None

        if _RICH_AVAILABLE and not quiet:
            self._console = Console(stderr=True)

    def _ensure_rich(self) -> bool:
        """Check if rich is available and not quiet."""
        return _RICH_AVAILABLE and not self.quiet and self._console is not None

    def _print(self, msg: str) -> None:
        if not self.quiet:
            if self._console:
                self._console.print(msg)
            else:
                print(msg, file=sys.stderr)

    def update(self, phase: str, completed: int, total: int, extra: str = "") -> None:
        """Update progress for the current phase."""
        if self.quiet:
            return

        if self._ensure_rich():
            # Finish previous rich task if any
            if self._rich_progress is not None and self._rich_task is not None:
                self._rich_progress.update(self._rich_task, completed=completed, total=total)
                return

            # Start a new rich progress bar
            description = phase
            self._rich_progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=self._console,
                transient=True,
            )
            self._rich_progress.start()
            self._rich_task = self._rich_progress.add_task(
                description, total=total, completed=completed,
            )
            return

        # Fallback: simple terminal output
        if total > 0:
            pct = completed * 100 // total
        else:
            pct = 100

        line = f"\r  [{phase}] {completed}/{total} ({pct}%)"
        if extra:
            line += f" | {extra}"
        line = line.ljust(80)
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish_phase(self, phase: str, message: str) -> None:
        """Print completion message for a phase."""
        if self.quiet:
            return

        # Stop any active rich progress bar
        if self._rich_progress is not None:
            self._rich_progress.stop()
            self._rich_progress = None
            self._rich_task = None

        if self._console:
            self._console.print(f"  [bold green][{phase}][/bold green] {message}")
        else:
            sys.stderr.write("\r" + " " * 80 + "\r")
            print(f"  [{phase}] {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        """Print error message."""
        if self._rich_progress is not None:
            self._rich_progress.stop()
            self._rich_progress = None
            self._rich_task = None

        if self._console:
            self._console.print(f"  [bold red]ERROR:[/bold red] {message}")
        else:
            self._print(f"  ERROR: {message}")

    def finish(self, message: str) -> None:
        """Print final summary."""
        if self._rich_progress is not None:
            self._rich_progress.stop()
            self._rich_progress = None
            self._rich_task = None
        self._print(message)

    def print_plan(self, plan_text: str) -> None:
        """Print the organize plan for dry-run."""
        self._print(plan_text)
