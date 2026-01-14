#!/usr/bin/env python3
#!/usr/bin/env python3

import argparse
import asyncio
import logging
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leak_detection.config import SystemConfig
from leak_detection.orchestrator import SystemOrchestrator
from leak_detection.tui.dashboard import LeakDetectionDashboard

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('leak_detection.log'),
    ]
)

logging.getLogger('leak_detection').setLevel(logging.INFO)

class LeakDetectionApp:

    def __init__(self):
        self._config = SystemConfig()
        self._orchestrator: SystemOrchestrator = None
        self._dashboard: LeakDetectionDashboard = None
        self._update_task: asyncio.Task = None
        self._log_messages = []

    def _log_callback(self, message: str):
        self._log_messages.append(message)
        if self._dashboard:
            self._dashboard.log_message(message)

    def _inject_leak(self):
        if self._orchestrator:
            self._orchestrator.inject_random_leak()

    def _clear_leaks(self):
        if self._orchestrator:
            self._orchestrator.clear_all_leaks()

    def _show_summary(self) -> str:
        if self._orchestrator:
            return self._orchestrator.get_detection_summary()
        return "No data available"

    def _reset_system(self):
        if self._orchestrator:
            self._orchestrator.reset()

    async def _update_loop(self):
        await asyncio.sleep(0.5)
        
        last_detection_count = 0

        while True:
            try:
                if self._dashboard and self._dashboard.is_paused:
                    await asyncio.sleep(0.1)
                    continue

                if self._orchestrator:
                    # Offload the heavy simulation step to a thread to keep UI responsive
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, self._orchestrator.step)
                    
                    # LOGGING DEBUG
                    logger = logging.getLogger("leak_detection")
                    logger.info(f"Step completed. Time: {result.sim_time:.1f}. Meters: {len(result.metrics)}")

                    multi_leak_results = []
                    if result.detected_leaks:
                        multi_leak_results = [
                            (d.get('location'), d.get('confidence', 0))
                            for d in result.detected_leaks
                        ]
                    
                    detection_history = self._orchestrator.leak_injector.get_last_detections(5)
                    current_count = len(detection_history)
                    
                    if current_count > last_detection_count and detection_history:
                        det = detection_history[0]  # Most recent
                        dist = det['distance']
                        dist_str = "EXACT MATCH" if dist == 0 else f"{dist} hops away"
                        self._dashboard.log_message(
                            f"[bold cyan]DETECTION #{current_count}:[/bold cyan] "
                            f"Actual={det['actual']} | Detected={det['estimated']} | {dist_str}"
                        )
                        last_detection_count = current_count

                    if self._dashboard:
                        self._dashboard.update_metrics(result.metrics)
                        self._dashboard.update_status(
                            status=result.status,
                            sim_time=result.sim_time,
                            ground_truth=result.ground_truth,
                            estimated_location=result.estimated_location,
                            multi_leak_results=multi_leak_results,
                            agent_summary=result.agent_summary,
                            detection_count=current_count
                        )
                        self._dashboard.update_anomaly(result.anomaly)
                        
                        confirmed_leaks = self._orchestrator.get_confirmed_leaks()
                        self._dashboard.update_history(detection_history, confirmed_leaks)
                        
                        self._orchestrator.auto_confirm_detections(min_cycles=3)

                await asyncio.sleep(0.2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.exception(f"Error in update loop: {e}")
                await asyncio.sleep(1)

    async def run_async(self):
        # logger.info("Initializing Leak Detection System...")
        self._orchestrator = SystemOrchestrator(
            self._config,
            event_callback=self._log_callback
        )
        # logger.info("System Initialized. Loading Dashboard...")

        self._dashboard = LeakDetectionDashboard(
            inject_leak_callback=self._inject_leak,
            clear_leaks_callback=self._clear_leaks,
            show_summary_callback=self._show_summary,
            reset_callback=self._reset_system
        )
        # logger.info("Dashboard Loaded. Starting Event Loop...")

        self._update_task = asyncio.create_task(self._update_loop())

        try:
            # logger.info("Launching TUI...")
            await self._dashboard.run_async()
        finally:
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass

    def run(self):
        asyncio.run(self.run_async())

def main():
    parser = argparse.ArgumentParser(description="Water Network Leak Detection System")
    args = parser.parse_args()

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