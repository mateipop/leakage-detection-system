#!/usr/bin/env python3
"""
Leak Detection System - Main Entry Point

A Digital Twin for water network monitoring with AI-driven leak detection.

Usage:
    python main.py

Controls:
    L - Inject a random leak
    C - Clear all leaks
    S - Show detection summary
    R - Reset system
    P - Pause/Resume simulation
    Q - Quit

Requirements:
    pip install textual rich numpy

Optional:
    pip install wntr  (for full hydraulic simulation)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from leak_detection.config import SystemConfig
from leak_detection.orchestrator import SystemOrchestrator
from leak_detection.tui.dashboard import LeakDetectionDashboard


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('leak_detection.log'),
    ]
)

# Set our modules to DEBUG
logging.getLogger('leak_detection').setLevel(logging.INFO)


class LeakDetectionApp:
    """Main application controller."""

    def __init__(self):
        self._config = SystemConfig()
        self._orchestrator: SystemOrchestrator = None
        self._dashboard: LeakDetectionDashboard = None
        self._update_task: asyncio.Task = None
        self._log_messages = []

    def _log_callback(self, message: str):
        """Callback for log messages."""
        self._log_messages.append(message)
        if self._dashboard:
            self._dashboard.log_message(message)

    def _inject_leak(self):
        """Inject a random leak."""
        if self._orchestrator:
            self._orchestrator.inject_random_leak()

    def _clear_leaks(self):
        """Clear all leaks."""
        if self._orchestrator:
            self._orchestrator.clear_all_leaks()

    def _show_summary(self) -> str:
        """Get detection summary."""
        if self._orchestrator:
            return self._orchestrator.get_detection_summary()
        return "No data available"

    def _reset_system(self):
        """Reset the system."""
        if self._orchestrator:
            self._orchestrator.reset()

    async def _update_loop(self):
        """Main update loop running in background."""
        # Initial delay to let UI settle
        await asyncio.sleep(0.5)

        while True:
            try:
                # Check if paused
                if self._dashboard and self._dashboard.is_paused:
                    await asyncio.sleep(0.1)
                    continue

                # Run simulation step
                if self._orchestrator:
                    result = self._orchestrator.step()

                    # Get multi-leak detection results (from original AI system)
                    multi_leak_results = []
                    multi = self._orchestrator.agent.get_multi_leak_result(max_leaks=5)
                    if multi:
                        multi_leak_results = [
                            (loc.estimated_node, loc.confidence)
                            for loc in multi.localizations
                        ]

                    # Update dashboard with multi-agent summary
                    if self._dashboard:
                        self._dashboard.update_metrics(result.metrics)
                        self._dashboard.update_status(
                            result.status,
                            result.sim_time,
                            result.ground_truth,
                            result.estimated_location,
                            multi_leak_results,
                            result.agent_summary  # Pass multi-agent system data
                        )
                        self._dashboard.update_anomaly(result.anomaly)

                # Control simulation speed (faster than real-time)
                # 0.2 seconds real time = 1 simulation step (5 min simulated)
                await asyncio.sleep(0.2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception(f"Error in update loop: {e}")
                await asyncio.sleep(1)

    async def run_async(self):
        """Run the application asynchronously."""
        # Initialize orchestrator
        print("Initializing Leak Detection System...")
        self._orchestrator = SystemOrchestrator(
            self._config,
            event_callback=self._log_callback
        )

        # Create dashboard
        self._dashboard = LeakDetectionDashboard(
            inject_leak_callback=self._inject_leak,
            clear_leaks_callback=self._clear_leaks,
            show_summary_callback=self._show_summary,
            reset_callback=self._reset_system
        )

        # Start update loop
        self._update_task = asyncio.create_task(self._update_loop())

        try:
            # Run the TUI
            await self._dashboard.run_async()
        finally:
            # Cleanup
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass

    def run(self):
        """Run the application."""
        asyncio.run(self.run_async())


def main():
    """Main entry point."""
    print("""
+==============================================================+
|       WATER NETWORK LEAK DETECTION SYSTEM                    |
|       Digital Twin with AI-Driven Monitoring                 |
+--------------------------------------------------------------+
|  Controls:                                                   |
|    L - Inject random leak    C - Clear all leaks             |
|    S - Show summary          R - Reset system                |
|    P - Pause/Resume          Q - Quit                        |
+==============================================================+
    """)

    try:
        app = LeakDetectionApp()
        app.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        logging.exception("Fatal error")
        sys.exit(1)


if __name__ == "__main__":
    main()
