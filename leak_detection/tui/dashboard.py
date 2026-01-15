
import asyncio
import logging
import threading
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
from collections import deque

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Label,
    DataTable, ProgressBar, RichLog,
    TabbedContent, TabPane
)
from textual.reactive import reactive
from textual.timer import Timer
from textual.binding import Binding

from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group

from ..config import SamplingMode, SystemStatus, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class MetricsPanel(Static):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metrics: Dict[str, Dict] = {}

    def update_metrics(self, metrics: Dict[str, Dict]):
        self._metrics = metrics
        self.refresh()

    def render(self) -> Panel:
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

        sorted_metrics = sorted(
            self._metrics.items(),
            key=lambda x: (
                x[1].get('confidence', 0),
                max(
                    abs(x[1].get('pressure_zscore') or 0),
                    abs(x[1].get('flow_zscore') or 0)
                )
            ),
            reverse=True
        )

        for node_id, data in sorted_metrics[:20]:
            pressure = data.get('pressure', 0)
            p_zscore = data.get('pressure_zscore')
            flow = data.get('flow', 0)
            f_zscore = data.get('flow_zscore')
            confidence = data.get('confidence', 0)

            if confidence >= 0.4:
                status = Text("!! ALERT", style="bold red")
            elif confidence >= 0.2:
                status = Text("? WATCH", style="yellow")
            elif (p_zscore is not None and abs(p_zscore) > 2.0) or (f_zscore is not None and abs(f_zscore) > 2.0):
                status = Text("! ANOMALY", style="magenta")
            else:
                status = Text("OK", style="green")

            p_zscore_str = f"{p_zscore:.2f}" if p_zscore is not None else "---"
            f_zscore_str = f"{f_zscore:.2f}" if f_zscore is not None else "---"

            if p_zscore is not None and p_zscore < -2:
                p_zscore_str = f"[red]{p_zscore_str}[/red]"
            elif p_zscore is not None and abs(p_zscore) > 1.5:
                p_zscore_str = f"[yellow]{p_zscore_str}[/yellow]"
            
            # Highlight flow spikes
            if f_zscore is not None and abs(f_zscore) > 2.0:
                 f_zscore_str = f"[red]{f_zscore_str}[/red]"
            elif f_zscore is not None and abs(f_zscore) > 1.5:
                 f_zscore_str = f"[yellow]{f_zscore_str}[/yellow]"

            table.add_row(
                node_id,
                f"{pressure:.1f} psi",
                p_zscore_str,
                f"{flow:.1f} L/s",
                f_zscore_str,
                status
            )

        total = len(self._metrics)
        return Panel(table, title=f"Live Sensor Metrics ({total} nodes)", border_style="blue")

class SystemStatusPanel(Static):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status: Any = None
        self._sim_time: float = 0.0
        self._ground_truth: List[str] = []
        self._estimated_location: Optional[str] = None
        self._multi_leak_results: List[tuple] = []  # [(node, confidence), ...]
        self._agent_summary: Optional[Dict] = None  # Multi-agent system summary
        self._detection_count: int = 0  # Actual detection count from leak_injector

    def update_status(
        self,
        status: Any,
        sim_time: float,
        ground_truth: List[str] = None,
        estimated_location: str = None,
        multi_leak_results: List[tuple] = None,
        agent_summary: Dict = None,
        detection_count: int = None
    ):
        self._status = status
        self._sim_time = sim_time
        self._ground_truth = ground_truth or []
        self._estimated_location = estimated_location
        self._multi_leak_results = multi_leak_results or []
        self._agent_summary = agent_summary
        
        if detection_count is not None:
             self._detection_count = detection_count
        elif status and hasattr(status, 'leaks_detected'):
             self._detection_count = status.leaks_detected
        
        self.refresh()

    def render(self) -> Panel:
        if self._status is None and self._agent_summary is None:
            return Panel("Initializing...", title="System Status")

        hours = int(self._sim_time // 3600)
        minutes = int((self._sim_time % 3600) // 60)
        seconds = int(self._sim_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        system_status_name = "UNKNOWN"
        sampling_mode_str = "UNKNOWN"
        monitored_nodes = 0
        samples_processed = 0
        alerts_triggered = 0
        
        if self._status:
            system_status_name = self._status.system_status.name
            sampling_mode_str = "ECO" if self._status.sampling_mode == SamplingMode.ECO else "HIGH-RES"
            monitored_nodes = self._status.monitored_nodes
            samples_processed = self._status.samples_processed
            alerts_triggered = self._status.alerts_triggered
        elif self._agent_summary:
            coord = self._agent_summary.get("coordinator", {})
            system_status_name = coord.get("system_mode", "NORMAL")
            
            if system_status_name == "INVESTIGATING":
                 sampling_mode_str = "HIGH-RES"
            else:
                 sampling_mode_str = "ECO"
                 
            monitored_nodes = self._agent_summary.get("sensor_count", 0)
            alerts_triggered = coord.get("total_alerts_received", 0)

        status_colors = {
            "NORMAL": ("green", "●"),
            "ALERT": ("yellow", "!"),
            "INVESTIGATING": ("yellow", "◉"),
            "LEAK_CONFIRMED": ("red", "⬤"),
        }
        color, icon = status_colors.get(
            system_status_name,
            ("white", "○")
        )

        mode_text = "ECO (Low Power)" if sampling_mode_str == "ECO" else "HIGH-RES (High Power)"
        mode_color = "green" if sampling_mode_str == "ECO" else "red"

        lines = [
            f"[bold]Simulation Time:[/bold] {time_str}",
            "",
            f"[bold]System Status:[/bold] [{color}]{icon} {system_status_name}[/{color}]",
            f"[bold]Sampling Mode:[/bold] [{mode_color}]{mode_text}[/{mode_color}]",
            "",
            f"[bold]Monitored Nodes:[/bold] {monitored_nodes}",
            f"[bold]Samples Processed:[/bold] {samples_processed}",
            f"[bold]Alerts Triggered:[/bold] {alerts_triggered}",
            f"[bold]Leaks Detected:[/bold] {self._detection_count}",
        ]

        if self._agent_summary:
            lines.append("")
            lines.append("[bold magenta]=== MULTI-AGENT SYSTEM ===[/bold magenta]")
            
            coord = self._agent_summary.get("coordinator", {})
            lines.append(f"[bold]Agents:[/bold] {self._agent_summary.get('agent_count', 0)} total")
            lines.append(f"[bold]Coordinator Mode:[/bold] {coord.get('system_mode', 'UNKNOWN')}")
            lines.append(f"[bold]Alerts Received:[/bold] {coord.get('total_alerts_received', 0)}")
            lines.append(f"[bold]Investigations:[/bold] {coord.get('investigations_opened', 0)}")
            
            active_invs = self._agent_summary.get("active_investigations", [])
            if active_invs:
                lines.append(f"[bold yellow]Active Investigations ({len(active_invs)}):[/bold yellow]")
                for inv in active_invs[:3]:
                    status_icon = "*" if inv["status"] == "active" else "+" if inv["status"] == "localized" else "○"
                    loc_str = ""
                    if inv.get("localization"):
                        loc = inv["localization"]
                        loc_str = f" → {loc.get('probable_location', '?')} ({loc.get('confidence', 0):.0%})"
                    lines.append(f"  {status_icon} {inv['id']}: {inv['status']}{loc_str}")

        lines.append("")
        lines.append("[bold cyan]=== LEAK ANALYSIS ===[/bold cyan]")

        if self._ground_truth:
            lines.append(f"[bold]Active Leaks ({len(self._ground_truth)}):[/bold] [red]{', '.join(self._ground_truth)}[/red]")
        else:
            lines.append("[bold]Active Leaks:[/bold] [green]None[/green]")

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._anomaly: Any = None

    def update_anomaly(self, anomaly: Any):
        self._anomaly = anomaly
        self.refresh()

    def render(self) -> Panel:
        if self._anomaly is None:
            return Panel(
                "[green]No anomalies detected[/green]\n\n"
                "System is operating within normal parameters.",
                title="Anomaly Detection",
                border_style="green"
            )

        confidence = getattr(self._anomaly, 'confidence', 0) if not isinstance(self._anomaly, dict) else self._anomaly.get('confidence', 0)
        node_id = getattr(self._anomaly, 'node_id', 'Unknown') if not isinstance(self._anomaly, dict) else self._anomaly.get('node_id', 'Unknown')
        
        confidence_pct = confidence * 100

        bar_width = 20
        filled = int(bar_width * confidence)
        bar = "█" * filled + "░" * (bar_width - filled)

        if confidence_pct >= 80:
            conf_color = "red"
        elif confidence_pct >= 50:
            conf_color = "yellow"
        else:
            conf_color = "green"

        lines = [
            f"[bold red]! ANOMALY DETECTED[/bold red]",
            "",
            f"[bold]Node:[/bold] {node_id}",
            f"[bold]Type:[/bold] Network Anomaly",
            "",
            f"[bold]Confidence:[/bold] [{conf_color}]{confidence_pct:.1f}%[/{conf_color}]",
            f"  [{conf_color}]{bar}[/{conf_color}]",
            "",
            f"[bold]Analysis:[/bold]",
            f"  Multi-Agent Network Detection",
        ]

        return Panel(
            "\n".join(lines),
            title="Anomaly Detection",
            border_style="red" if confidence_pct >= 80 else "yellow"
        )

class DetectionHistoryPanel(Static):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history: List[Dict] = []
        self._confirmed_leaks: List[str] = []

    def update_history(self, history: List[Dict], confirmed_leaks: List[str] = None):
        self._history = history
        self._confirmed_leaks = confirmed_leaks or []
        self.refresh()

    def render(self) -> Panel:
        if not self._history:
            return Panel(
                "[dim]No detections yet.[/dim]\n\n"
                "Press [bold]L[/bold] to inject a leak and wait for detection.",
                title="Detection History (0 total)",
                border_style="dim"
            )

        table = Table(
            title=None,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            box=None
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Actual Leak", width=12)
        table.add_column("AI Detected", width=12)
        table.add_column("Distance", width=10)
        table.add_column("Status", width=12)

        for i, det in enumerate(self._history, 1):
            actual = det.get('actual', '?')
            estimated = det.get('estimated', '?')
            distance = det.get('distance', 99)
            
            if distance == 0:
                dist_str = "[green]0 (exact)[/green]"
            elif distance == 1:
                dist_str = "[cyan]1 hop[/cyan]"
            elif distance == 2:
                dist_str = "[yellow]2 hops[/yellow]"
            elif distance <= 4:
                dist_str = f"[yellow]{distance} hops[/yellow]"
            else:
                dist_str = f"[red]{distance}+ hops[/red]"
            
            if actual in self._confirmed_leaks:
                status_str = "[cyan]+ CONFIRMED[/cyan]"
            else:
                status_str = "[yellow]* NEW[/yellow]"

            table.add_row(str(i), actual, estimated, dist_str, status_str)

        total = len(self._history)
        exact = sum(1 for d in self._history if d.get('distance', 99) == 0)
        within_2 = sum(1 for d in self._history if d.get('distance', 99) <= 2)
        confirmed = len(self._confirmed_leaks)
        
        stats = f"\nExact: {exact}/{total} | Within 2 hops: {within_2}/{total} | Confirmed: {confirmed}"

        from rich.console import Group
        content = Group(table, stats)

        return Panel(
            content,
            title=f"Detection History ({total} total)",
            border_style="cyan"
        )

class LeakDetectionDashboard(App):

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
        height: 8;
        border: solid green;
    }

    #right-column {
        width: 1fr;
        height: 100%;
    }

    #status-panel {
        height: auto;
        min-height: 15;
    }

    #anomaly-panel {
        height: auto;
        min-height: 10;
    }
    
    #history-panel {
        height: auto;
        min-height: 12;
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

            with ScrollableContainer(id="right-column"):
                yield SystemStatusPanel(id="status-panel")
                yield AnomalyPanel(id="anomaly-panel")
                yield DetectionHistoryPanel(id="history-panel")

        yield Footer()

    def on_mount(self) -> None:
        self.log_message("[green]System initialized[/green]")
        self.log_message("Press [bold]L[/bold] to inject a leak, [bold]Q[/bold] to quit")

    def action_inject_leak(self) -> None:
        if self._inject_leak_callback:
            self._inject_leak_callback()
            self.log_message("[red bold]LEAK INJECTED![/red bold]")

    def action_clear_leaks(self) -> None:
        if self._clear_leaks_callback:
            self._clear_leaks_callback()
            self.log_message("[green]All leaks cleared[/green]")

    def action_show_summary(self) -> None:
        if self._show_summary_callback:
            summary = self._show_summary_callback()
            self.log_message("\n" + summary)

    def action_reset_system(self) -> None:
        if self._reset_callback:
            self._reset_callback()
            self.log_message("[yellow]System reset[/yellow]")

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = "PAUSED" if self._paused else "RESUMED"
        self.log_message(f"[yellow]Simulation {status}[/yellow]")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def log_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        def _write():
            try:
                log_widget = self.query_one("#log-panel", RichLog)
                log_widget.write(f"[dim]{timestamp}[/dim] {message}")
            except Exception:
                pass
        
        # If on the main thread, run directly. If on a background thread, schedule it.
        if hasattr(self, '_thread_id') and self._thread_id == threading.get_ident():
            _write()
        else:
            self.call_from_thread(_write)

    def update_metrics(self, metrics: Dict[str, Dict]) -> None:
        try:
            panel = self.query_one("#metrics-panel", MetricsPanel)
            panel.update_metrics(metrics)
        except Exception as e:
            with open("dashboard_error.log", "a") as f:
                f.write(f"Update Metrics Error: {e}\n")

    def update_status(
        self,
        status: Any,
        sim_time: float,
        ground_truth: List[str] = None,
        estimated_location: str = None,
        multi_leak_results: List[tuple] = None,
        agent_summary: Dict = None,
        detection_count: int = None
    ) -> None:
        panel = self.query_one("#status-panel", SystemStatusPanel)
        panel.update_status(status, sim_time, ground_truth, estimated_location, multi_leak_results, agent_summary, detection_count)

    def update_anomaly(self, anomaly: Any) -> None:
        panel = self.query_one("#anomaly-panel", AnomalyPanel)
        panel.update_anomaly(anomaly)

    def update_history(self, detections: List[Dict], confirmed_leaks: List[str] = None) -> None:
        panel = self.query_one("#history-panel", DetectionHistoryPanel)
        panel.update_history(detections, confirmed_leaks)