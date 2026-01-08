import logging
import time
import json
import numpy as np
from statistics import mean
from typing import List, Dict, Any
from leak_detection.orchestrator import SystemOrchestrator
from leak_detection.config import SystemConfig, DEFAULT_CONFIG

# Configure logging to capture system events
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("TestRunner")
logger.setLevel(logging.INFO)

class PerformanceTester:
    def __init__(self, num_scenarios: int = 5, steps_per_scenario: int = 50):
        self.num_scenarios = num_scenarios
        self.steps_per_scenario = steps_per_scenario
        self.results: List[Dict[str, Any]] = []
        
        # Override config for faster testing if needed
        self.config = DEFAULT_CONFIG
        # Ensure we can detect fast
        self.config.simulation.hydraulic_timestep_seconds = 300 
        
        self.orchestrator = SystemOrchestrator(
            config=self.config,
            use_multi_agent=True,
            event_callback=self._on_event
        )

    def _on_event(self, msg: str):
        # Mute standard events to keep output clean, or log to file
        pass

    def run(self):
        print(json.dumps({"type": "status", "msg": "Starting Performance Test..."}))
        
        # Warmup
        print(json.dumps({"type": "info", "msg": "Warming up system (10 steps)..."}))
        for _ in range(10):
            self.orchestrator.step()

        for i in range(self.num_scenarios):
            result = self.run_scenario(i + 1)
            self.results.append(result)
            self.orchestrator.clear_all_leaks()
            # Stabilize after reset
            for _ in range(5):
                self.orchestrator.step()

        self.print_summary()

    def run_scenario(self, scenario_id: int) -> Dict[str, Any]:
        logger.info(f"--- Scenario {scenario_id} ---")
        
        # 1. Inject Leak
        target_node = self.orchestrator.inject_random_leak(at_sensor=True) # Prioritize sensor nodes for fairness in basic testing
        if not target_node:
            return {"id": scenario_id, "error": "Failed to inject leak"}

        injection_step = 0
        detected = False
        detection_step = -1
        detected_node = None
        localization_error = 0.0
        
        initial_sim_time = self.orchestrator.sim_time

        # 2. Run Wait Loop
        start_time = time.time()
        for step in range(self.steps_per_scenario):
            current_sim_time = self.orchestrator.sim_time
            
            try:
                orch_result = self.orchestrator.step()
            except Exception as e:
                return {"id": scenario_id, "error": str(e), "trace": "Orchestrator step crashed"}

            # Check for detection
            # The orchestrator records detections automatically into the leak injector results
            # But we can also check the result object directly
            
            # Check Multi-Agent detected leaks specifically
            mas_leaks = orch_result.detected_leaks
            
            if mas_leaks:
                detected = True
                detection_step = step
                
                # Get best leak
                best_leak = max(mas_leaks, key=lambda x: x['confidence'])
                detected_node = best_leak['location']
                
                # Calculate distance
                # We need to access the network graph for true distance, simplified here
                # Using orchestrator's helper if available or simple check
                try:
                    dist = self.orchestrator._calculate_detection_distance(target_node, detected_node)
                    localization_error = float(dist)
                except:
                    localization_error = -1.0
                
                break
            
            # Fallback check on orchestrator status (legacy support)
            if orch_result.ground_truth and orch_result.estimated_location:
                 # If the system *thinks* it found something
                 detected = True
                 detection_step = step
                 detected_node = orch_result.estimated_location
                 dist = self.orchestrator._calculate_detection_distance(target_node, detected_node)
                 localization_error = float(dist)
                 break

        duration = time.time() - start_time
        
        return {
            "id": scenario_id,
            "target": target_node,
            "detected": detected,
            "steps_to_detect": detection_step,
            "detected_node": detected_node,
            "localization_error_hops": localization_error,
            "sim_time_elapsed": self.orchestrator.sim_time - initial_sim_time,
            "real_time_sec": duration
        }

    def print_summary(self):
        total = len(self.results)
        detected = sum(1 for r in self.results if r.get("detected"))
        
        valid_results = [r for r in self.results if r.get("detected")]
        
        if valid_results:
            avg_steps = mean(r["steps_to_detect"] for r in valid_results)
            avg_error = mean(r["localization_error_hops"] for r in valid_results)
        else:
            avg_steps = 0
            avg_error = 0

        summary = {
            "total_scenarios": total,
            "detection_rate": detected / total if total > 0 else 0,
            "avg_steps_to_detect": avg_steps,
            "avg_localization_error": avg_error,
            "details": self.results
        }
        
        print("\n=== SYSTEM REPORT ===")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    tester = PerformanceTester(num_scenarios=3, steps_per_scenario=30)
    tester.run()
