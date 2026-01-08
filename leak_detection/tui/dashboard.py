"""
Dashboard - Main TUI application using Textual.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Callable
from datetime import datetime
from collections import deque

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Label,
    DataTable, ProgressBar, RichLog
)
from textual.reactive import reactive
from textual.timer import Timer
from textual.binding import Binding

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group

from ..config import SamplingMode, SystemStatus, DEFAULT_CONFIG
from ..ai_layer.agent_controller import AgentStatus, MonitoringCycleResult
from ..ai_layer.inference_engine import InferenceResult


logger = logging.getLogger(__name__)


class MetricsPanel(Static):
    """Panel showing live pressure/flow metrics."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics: Dict[str, Dict] = {}

    def update_metrics(self, metrics: Dict[str, Dict]):
        """Update displayed metrics."""
        self._metrics = metrics
        self.refresh()

    def render(self) -> Panel:
        """Render the metrics panel."""
        table = Table(
            title="Live Sensor Metrics",
            show_header=True,
            header_style="bold cyan",
            expand=True,
            box=None  # Remove table border to save space
        )
        table.add_column("Node", style="green", width=10, no_wrap=True)
        table.add_column("Pressure", justify="right", width=10, no_wrap=True)
        table.add_column("P-Z", justify="right", width=8, no_wrap=True)
        table.add_column("Flow", justify="right", width=10, no_wrap=True)
        table.add_column("F-Z", justify="right", width=8, no_wrap=True)
        table.add_column("Status", justify="center", width=8, no_wrap=True)

        # Sort by confidence (highest first) to show anomalies at top
        sorted_metrics = sorted(
            self._metrics.items(),
            key=lambda x: (x[1].get('confidence', 0), abs(x[1].get('pressure_zscore') or 0)),
            reverse=True
        )

        # Show up to 20 nodes, prioritizing anomalies
        for node_id, data in sorted_metrics[:20]:
            pressure = data.get('pressure', 0)
            p_zscore = data.get('pressure_zscore')
            flow = data.get('flow', 0)
            f_zscore = data.get('flow_zscore')
            confidence = data.get('confidence', 0)

            # Color code based on confidence and Z-score
            if confidence >= 0.6:
                status = Text("!! ALERT", style="bold red")
            elif p_zscore is not None and abs(p_zscore) > 2:
                status = Text("! ANOMALY", style="red")
            elif confidence > 0.4:
                status = Text("? WATCH", style="yellow")
            else:
                status = Text("OK", style="green")

            p_zscore_str = f"{p_zscore:.2f}" if p_zscore is not None else "---"
            f_zscore_str = f"{f_zscore:.2f}" if f_zscore is not None else "---"

            # Color Z-scores
            if p_zscore is not None and p_zscore < -2:
                p_zscore_str = f"[red]{p_zscore_str}[/red]"
            elif p_zscore is not None and abs(p_zscore) > 1.5:
                p_zscore_str = f"[yellow]{p_zscore_str}[/yellow]"

            table.add_row(
                node_id,
                f"{pressure:.1f} psi",
                p_zscore_str,
                f"{flow:.1f} L/s",
                f_zscore_str,
                status
            )

        # Show total monitored count
        total = len(self._metrics)
        return Panel(table, title=f"Live Sensor Metrics ({total} nodes)", border_style="blue")


class SystemStatusPanel(Static):
    """Panel showing system status and AI mode."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status: Optional[AgentStatus] = None
        self._sim_time: float = 0.0
        self._ground_truth: List[str] = []
        self._estimated_location: Optional[str] = None
        self._multi_leak_results: List[tuple] = []  # [(node, confidence), ...]
        self._agent_summary: Optional[Dict] = None  # Multi-agent system summary

    def update_status(
        self,
        status: AgentStatus,
        sim_time: float,
        ground_truth: List[str] = None,
        estimated_location: str = None,
        multi_leak_results: List[tuple] = None,
        agent_summary: Dict = None
    ):
        """Update displayed status."""
        self._status = status
        self._sim_time = sim_time
        self._ground_truth = ground_truth or []
        self._estimated_location = estimated_location
        self._multi_leak_results = multi_leak_results or []
        self._agent_summary = agent_summary
        self.refresh()

    def render(self) -> Panel:
        """Render the status panel."""
        if self._status is None:
            return Panel("Initializing...", title="System Status")

        # Format simulation time
        hours = int(self._sim_time // 3600)
        minutes = int((self._sim_time % 3600) // 60)
        seconds = int(self._sim_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # System status color
        status_colors = {
            SystemStatus.NORMAL: ("green", "●"),
            SystemStatus.ALERT: ("yellow", "⚠"),
            SystemStatus.INVESTIGATING: ("yellow", "◉"),
            SystemStatus.LEAK_CONFIRMED: ("red", "⬤"),
        }
        color, icon = status_colors.get(
            self._status.system_status,
            ("white", "○")
        )

        # Sampling mode display
        mode_str = "ECO (Low Power)" if self._status.sampling_mode == SamplingMode.ECO else "HIGH-RES (High Power)"
        mode_color = "green" if self._status.sampling_mode == SamplingMode.ECO else "red"

        # Build content
        lines = [
            f"[bold]Simulation Time:[/bold] {time_str}",
            "",
            f"[bold]System Status:[/bold] [{color}]{icon} {self._status.system_status.name}[/{color}]",
            f"[bold]Sampling Mode:[/bold] [{mode_color}]{mode_str}[/{mode_color}]",
            "",
            f"[bold]Monitored Nodes:[/bold] {self._status.monitored_nodes}",
            f"[bold]Samples Processed:[/bold] {self._status.samples_processed}",
            f"[bold]Alerts Triggered:[/bold] {self._status.alerts_triggered}",
            f"[bold]Leaks Detected:[/bold] {self._status.leaks_detected}",
        ]

        # Show current decision
        if self._status.current_decision:
            lines.append("")
            lines.append(f"[bold]Current Action:[/bold] {self._status.current_decision.action.name}")
            if self._status.current_decision.confidence > 0:
                lines.append(f"[bold]Confidence:[/bold] {self._status.current_decision.confidence:.1%}")

        # === MULTI-AGENT SYSTEM STATUS ===
        if self._agent_summary:
            lines.append("")
            lines.append("[bold magenta]=== MULTI-AGENT SYSTEM ===[/bold magenta]")
            
            coord = self._agent_summary.get("coordinator", {})
            lines.append(f"[bold]Agents:[/bold] {self._agent_summary.get('agent_count', 0)} total")
            lines.append(f"[bold]Coordinator Mode:[/bold] {coord.get('system_mode', 'UNKNOWN')}")
            lines.append(f"[bold]Alerts Received:[/bold] {coord.get('total_alerts_received', 0)}")
            lines.append(f"[bold]Investigations:[/bold] {coord.get('investigations_opened', 0)}")
            
            # Show active investigations
            active_invs = self._agent_summary.get("active_investigations", [])
            if active_invs:
                lines.append(f"[bold yellow]Active Investigations ({len(active_invs)}):[/bold yellow]")
                for inv in active_invs[:3]:
                    status_icon = "🔍" if inv["status"] == "active" else "✓" if inv["status"] == "localized" else "○"
                    loc_str = ""
                    if inv.get("localization"):
                        loc = inv["localization"]
                        loc_str = f" → {loc.get('probable_location', '?')} ({loc.get('confidence', 0):.0%})"
                    lines.append(f"  {status_icon} {inv['id']}: {inv['status']}{loc_str}")

        # Show leak comparison (always show this section)
        lines.append("")
        lines.append("[bold cyan]=== LEAK ANALYSIS ===[/bold cyan]")

        if self._ground_truth:
            lines.append(f"[bold]Active Leaks ({len(self._ground_truth)}):[/bold] [red]{', '.join(self._ground_truth)}[/red]")
        else:
            lines.append("[bold]Active Leaks:[/bold] [green]None[/green]")

        # Show multi-leak detection results
        if self._multi_leak_results:
            lines.append(f"[bold]AI Detected ({len(self._multi_leak_results)}):[/bold]")
            for node, confidence in self._multi_leak_results[:5]:  # Show top 5
                bar_len = int(confidence * 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                color = "red" if confidence >= 0.7 else "yellow" if confidence >= 0.5 else "dim"
                lines.append(f"  [{color}]{node:6s} [{bar}] {confidence:.0%}[/{color}]")
        elif self._estimated_location:
            lines.append(f"[bold]AI Estimate:[/bold] [yellow]{self._estimated_location}[/yellow]")
        else:
            lines.append("[bold]AI Estimate:[/bold] [dim]Searching...[/dim]")

        return Panel(
            "\n".join(lines),
            title="System Status",
            border_style="cyan"
        )


class AnomalyPanel(Static):
    """Panel showing current anomaly information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._anomaly: Optional[InferenceResult] = None

    def update_anomaly(self, anomaly: Optional[InferenceResult]):
        """Update displayed anomaly."""
        self._anomaly = anomaly
        self.refresh()

    def render(self) -> Panel:
        """Render the anomaly panel."""
        if self._anomaly is None:
            return Panel(
                "[green]No anomalies detected[/green]\n\n"
                "System is operating within normal parameters.",
                title="Anomaly Detection",
                border_style="green"
            )

        confidence_pct = self._anomaly.confidence * 100

        # Confidence bar
        bar_width = 20
        filled = int(bar_width * self._anomaly.confidence)
        bar = "█" * filled + "░" * (bar_width - filled)

        if confidence_pct >= 80:
            conf_color = "red"
        elif confidence_pct >= 50:
            conf_color = "yellow"
        else:
            conf_color = "green"

        lines = [
            f"[bold red]⚠ ANOMALY DETECTED[/bold red]",
            "",
            f"[bold]Node:[/bold] {self._anomaly.node_id}",
            f"[bold]Type:[/bold] {self._anomaly.anomaly_type.value}",
            "",
            f"[bold]Confidence:[/bold] [{conf_color}]{confidence_pct:.1f}%[/{conf_color}]",
            f"  [{conf_color}]{bar}[/{conf_color}]",
            "",
            f"[bold]Contributions:[/bold]",
            f"  Pressure: {self._anomaly.pressure_contribution:.1%}",
            f"  Flow:     {self._anomaly.flow_contribution:.1%}",
            f"  Spatial:  {self._anomaly.spatial_contribution:.1%}",
            "",
            f"[bold]Analysis:[/bold]",
            f"  {self._anomaly.explanation}",
        ]

        return Panel(
            "\n".join(lines),
            title="Anomaly Detection",
            border_style="red" if confidence_pct >= 80 else "yellow"
        )


class LeakDetectionDashboard(App):
    """Main TUI Dashboard Application."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #left-area {
        width: 2fr;
        height: 100%;
    }

    #header-area {
        height: 3;
        padding: 0 1;
        background: $surface;
    }

    #metrics-container {
        height: 1fr;
    }

    #metrics-panel {
        height: 100%;
    }

    #log-panel {
        height: 10;
        border: solid green;
    }

    #right-column {
        width: 1fr;
        height: 100%;
    }

    #status-panel {
        height: 1fr;
    }

    #anomaly-panel {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("l", "inject_leak", "Inject Leak"),
        Binding("c", "clear_leaks", "Clear All"),
        Binding("s", "show_summary", "Summary"),
        Binding("r", "reset_system", "Reset"),
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause/Resume"),
    ]

    def __init__(
        self,
        inject_leak_callback: Callable[[], None] = None,
        clear_leaks_callback: Callable[[], None] = None,
        show_summary_callback: Callable[[], str] = None,
        reset_callback: Callable[[], None] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._inject_leak_callback = inject_leak_callback
        self._clear_leaks_callback = clear_leaks_callback
        self._show_summary_callback = show_summary_callback
        self._reset_callback = reset_callback

        self._log_buffer: deque = deque(maxlen=100)
        self._paused = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="left-area"):
                with Container(id="header-area"):
                    yield Static(
                        "[bold cyan]LEAK DETECTION SYSTEM[/bold cyan] [dim]| WNTR Digital Twin[/dim]",
                        id="title"
                    )
                with ScrollableContainer(id="metrics-container"):
                    yield MetricsPanel(id="metrics-panel")
                yield RichLog(id="log-panel", highlight=True, markup=True, max_lines=50)

            with Vertical(id="right-column"):
                yield SystemStatusPanel(id="status-panel")
                yield AnomalyPanel(id="anomaly-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.log_message("[green]System initialized[/green]")
        self.log_message("Press [bold]L[/bold] to inject a leak, [bold]Q[/bold] to quit")

    def action_inject_leak(self) -> None:
        """Inject a leak."""
        if self._inject_leak_callback:
            self._inject_leak_callback()
            self.log_message("[red bold]LEAK INJECTED![/red bold]")

    def action_clear_leaks(self) -> None:
        """Clear all leaks."""
        if self._clear_leaks_callback:
            self._clear_leaks_callback()
            self.log_message("[green]All leaks cleared[/green]")

    def action_show_summary(self) -> None:
        """Show detection summary."""
        if self._show_summary_callback:
            summary = self._show_summary_callback()
            self.log_message("\n" + summary)

    def action_reset_system(self) -> None:
        """Reset the system."""
        if self._reset_callback:
            self._reset_callback()
            self.log_message("[yellow]System reset[/yellow]")

    def action_toggle_pause(self) -> None:
        """Toggle pause state."""
        self._paused = not self._paused
        status = "PAUSED" if self._paused else "RESUMED"
        self.log_message(f"[yellow]Simulation {status}[/yellow]")

    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused

    def log_message(self, message: str) -> None:
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_widget = self.query_one("#log-panel", RichLog)
        log_widget.write(f"[dim]{timestamp}[/dim] {message}")

    def update_metrics(self, metrics: Dict[str, Dict]) -> None:
        """Update the metrics display."""
        panel = self.query_one("#metrics-panel", MetricsPanel)
        panel.update_metrics(metrics)

    def update_status(
        self,
        status: AgentStatus,
        sim_time: float,
        ground_truth: List[str] = None,
        estimated_location: str = None,
        multi_leak_results: List[tuple] = None,
        agent_summary: Dict = None
    ) -> None:
        """Update the status display."""
        panel = self.query_one("#status-panel", SystemStatusPanel)
        panel.update_status(status, sim_time, ground_truth, estimated_location, multi_leak_results, agent_summary)

    def update_anomaly(self, anomaly: Optional[InferenceResult]) -> None:
        """Update the anomaly display."""
        panel = self.query_one("#anomaly-panel", AnomalyPanel)
        panel.update_anomaly(anomaly)
