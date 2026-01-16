"""
Headless analysis of false positives in the leak detection system.
"""
import logging
import sys

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from leak_detection.orchestrator import SystemOrchestrator
from leak_detection.config import DEFAULT_CONFIG

def run_analysis():
    print("=" * 70)
    print("FALSE POSITIVE ANALYSIS - HEADLESS RUN")
    print("=" * 70)
    
    # Create orchestrator
    events = []
    def capture_event(msg):
        events.append(msg)
        # Strip rich markup for cleaner output
        clean = msg.replace("[", "").replace("]", "")
        for color in ["green", "red", "yellow", "cyan", "blue", "bold", "dim", "/green", "/red", "/yellow", "/cyan", "/blue", "/bold", "/dim"]:
            clean = clean.replace(color, "")
        print(f"  EVENT: {clean.strip()}")
    
    orchestrator = SystemOrchestrator(
        config=DEFAULT_CONFIG,
        event_callback=capture_event
    )
    
    print(f"\nMonitored nodes: {len(orchestrator._fleet.monitored_nodes)}")
    print(f"Sensor nodes: {list(orchestrator._fleet.monitored_nodes)[:10]}...")
    
    # Run simulation WITHOUT any leaks first to see baseline false positive rate
    print("\n" + "=" * 70)
    print("PHASE 1: Running 30 steps WITHOUT any leaks (baseline false positive test)")
    print("=" * 70)
    
    detections_no_leak = []
    for step in range(30):
        result = orchestrator.step()
        
        if result.detected_leaks:
            for d in result.detected_leaks:
                detections_no_leak.append({
                    "step": step,
                    "time": result.sim_time,
                    "location": d.get("location"),
                    "confidence": d.get("confidence"),
                    "evaluation": d.get("evaluation")
                })
                print(f"\n  !!! DETECTION (no leak active): {d.get('location')} @ {d.get('confidence'):.0%}")
        
        if step % 10 == 0:
            print(f"  Step {step}: sim_time={result.sim_time:.0f}s, ground_truth={result.ground_truth}")
    
    print(f"\n  Baseline detections (should be 0): {len(detections_no_leak)}")
    
    # Now inject a leak and observe
    print("\n" + "=" * 70)
    print("PHASE 2: Injecting leak and observing detection")
    print("=" * 70)
    
    # Find a node that HAS a sensor
    sensor_nodes = list(orchestrator._fleet.monitored_nodes)
    leak_node = sensor_nodes[5] if len(sensor_nodes) > 5 else sensor_nodes[0]
    
    print(f"\n  Injecting leak at: {leak_node} (has sensor)")
    orchestrator._leak_injector.inject_leak(leak_node, 5.0, orchestrator._sim_time)
    # Re-run simulation after leak injection
    if not orchestrator._network.is_mock:
        orchestrator._network.run_simulation()
    
    detections_with_leak = []
    seen_evaluations = set()  # Track which evaluations we've seen
    
    for step in range(100):  # Run longer
        result = orchestrator.step()
        
        if result.detected_leaks:
            for d in result.detected_leaks:
                eval_info = d.get("evaluation", {})
                is_tp = eval_info.get("is_true_positive", False)
                is_fp = eval_info.get("is_false_positive", False)
                nearest = eval_info.get("nearest_leak")
                dist = eval_info.get("distance")
                
                # Only add unique detections
                detection_key = f"{d.get('location')}_{d.get('confidence')}"
                if detection_key not in seen_evaluations:
                    seen_evaluations.add(detection_key)
                    detections_with_leak.append({
                        "step": step,
                        "time": result.sim_time,
                        "location": d.get("location"),
                        "confidence": d.get("confidence"),
                        "is_tp": is_tp,
                        "is_fp": is_fp,
                        "nearest": nearest,
                        "distance": dist
                    })
        
        # Show sensor states every 10 steps
        if step % 10 == 0:
            coord = result.agent_summary.get("coordinator", {}) if result.agent_summary else {}
            print(f"  Step {step}: ground_truth={result.ground_truth}, "
                  f"mode={coord.get('system_mode', '?')}, "
                  f"alerts={coord.get('total_alerts_received', 0)}")
            
            # Show top anomalies from sensors
            if result.agent_summary and "sensors" in result.agent_summary:
                sensors = result.agent_summary["sensors"]
                # Sort by confidence
                sorted_sensors = sorted(
                    [(nid, s) for nid, s in sensors.items() if s.get("anomaly_confidence", 0) > 0],
                    key=lambda x: -x[1].get("anomaly_confidence", 0)
                )[:5]
                if sorted_sensors:
                    print(f"    Top alerting sensors:")
                    for nid, s in sorted_sensors:
                        is_leak = " *** ACTUAL LEAK" if nid == leak_node else ""
                        print(f"      {nid}: conf={s.get('anomaly_confidence', 0):.2f}, "
                              f"zscore={s.get('zscore', 0):.2f}{is_leak}")
                
                # Check if leak node sensor is alerting
                if leak_node in sensors:
                    leak_sensor = sensors[leak_node]
                    is_alerting = leak_sensor.get('anomaly_confidence', 0) > 0.25
                    print(f"    LEAK NODE {leak_node} SENSOR: conf={leak_sensor.get('anomaly_confidence', 0):.2f}, "
                          f"zscore={leak_sensor.get('zscore', leak_sensor.get('last_zscore', 0)):.2f}, "
                          f"alerting={is_alerting}, alerts_sent={leak_sensor.get('alerts_sent', 0)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nPhase 1 (no leaks): {len(detections_no_leak)} false positive detections")
    for d in detections_no_leak:
        print(f"  - Step {d['step']}: {d['location']} @ {d['confidence']:.0%}")
    
    print(f"\nPhase 2 (with leak at {leak_node}):")
    tps = [d for d in detections_with_leak if d["is_tp"]]
    fps = [d for d in detections_with_leak if d["is_fp"]]
    
    print(f"  True Positives: {len(tps)}")
    for d in tps[:5]:
        print(f"    - {d['location']} @ {d['confidence']:.0%} (matched {d['nearest']}, {d['distance']} hops)")
    
    print(f"  False Positives: {len(fps)}")
    for d in fps[:5]:
        print(f"    - {d['location']} @ {d['confidence']:.0%} (nearest was {d['nearest']}, {d['distance']} hops away)")
    
    # Look at agent summary to understand why
    print("\n" + "=" * 70)
    print("INVESTIGATING ROOT CAUSE")
    print("=" * 70)
    
    # Get recent anomalies from coordinator
    if orchestrator._multi_agent_system:
        coord = orchestrator._multi_agent_system.coordinator
        print(f"\n  Coordinator state:")
        print(f"    Recent anomalies: {len(coord._recent_anomalies)}")
        print(f"    Active investigations: {len(coord._active_investigations)}")
        
        # Show recent anomalies
        print(f"\n  Recent anomaly nodes:")
        anomaly_nodes = {}
        for a in coord._recent_anomalies:
            if a.node_id not in anomaly_nodes:
                anomaly_nodes[a.node_id] = {"count": 0, "max_conf": 0, "max_z": 0}
            anomaly_nodes[a.node_id]["count"] += 1
            anomaly_nodes[a.node_id]["max_conf"] = max(anomaly_nodes[a.node_id]["max_conf"], a.confidence)
            anomaly_nodes[a.node_id]["max_z"] = max(anomaly_nodes[a.node_id]["max_z"], abs(a.zscore))
        
        for node, stats in sorted(anomaly_nodes.items(), key=lambda x: -x[1]["count"])[:10]:
            is_leak = "*** LEAK" if node == leak_node else ""
            print(f"    {node}: {stats['count']} alerts, max_conf={stats['max_conf']:.2f}, max_z={stats['max_z']:.2f} {is_leak}")
        
        # Show active investigations with more detail
        print(f"\n  Active investigations:")
        for inv_id, inv in coord._active_investigations.items():
            print(f"    {inv_id}: status={inv.status}, sensors={inv.sensor_ids}")
            if inv.localization_result:
                loc = inv.localization_result
                best_loc = loc.get('probable_location')
                print(f"      -> Localized to: {best_loc} @ {loc.get('confidence'):.0%}")
                top_cands = [c['node_id'] for c in loc.get('candidates', [])[:5]]
                print(f"      -> Top candidates: {top_cands}")
                
                # Check hop distances from leak to top candidates
                for cand in top_cands:
                    hops = orchestrator._network.calculate_shortest_path_distance(cand, leak_node)
                    marker = " <-- WITHIN 15 HOPS" if hops <= 15 else ""
                    print(f"         {cand}: {hops} hops from leak{marker}")
    
    # Check if the actual leak node is even in the localizer's view
    print(f"\n  Is {leak_node} a sensor node? {leak_node in orchestrator._multi_agent_system.sensor_agents}")
    
    # Get the localizer's distance info for the leak node
    localizer = orchestrator._multi_agent_system.localizer
    print(f"  Localizer has {len(localizer.candidate_nodes)} candidate nodes")
    print(f"  Is {leak_node} in candidates? {leak_node in localizer.candidate_nodes}")
    
    # Check what sensors are near the leak node
    dists_to_leak = localizer._candidate_distances.get(leak_node, {})
    if dists_to_leak:
        sorted_nearby = sorted(dists_to_leak.items(), key=lambda x: x[1])[:5]
        print(f"  Nearest sensors to {leak_node}: {sorted_nearby}")
    
    # Analyze what score n114 would get if we manually compute it
    print(f"\n  DEBUGGING LOCALIZATION SCORES:")
    
    # Get the anomaly weights from the most recent investigation
    if coord._active_investigations:
        last_inv = list(coord._active_investigations.values())[-1]
        anomalies = last_inv.anomalies
        
        anomaly_weights = {}
        for anom in anomalies:
            node_id = anom.node_id
            weight = abs(anom.zscore) * anom.confidence
            anomaly_weights[node_id] = max(anomaly_weights.get(node_id, 0), weight)
        
        print(f"    Anomaly weights (sorted by z*conf):")
        for node, weight in sorted(anomaly_weights.items(), key=lambda x: -x[1])[:10]:
            is_leak = " *** LEAK" if node == leak_node else ""
            print(f"      {node}: weight={weight:.2f}{is_leak}")
        
        # Manually compute scores for leak_node and compare to winner
        if last_inv.localization_result:
            winner = last_inv.localization_result.get('probable_location')
            
            # Compute score for leak_node
            leak_dists = localizer._candidate_distances.get(leak_node, {})
            winner_dists = localizer._candidate_distances.get(winner, {})
            
            leak_score = 0.0
            winner_score = 0.0
            
            for anom_sensor, weight in anomaly_weights.items():
                leak_dist = leak_dists.get(anom_sensor, 9999.0)
                winner_dist = winner_dists.get(anom_sensor, 9999.0)
                
                if leak_dist < 5000:
                    leak_score += weight / (100.0 + leak_dist)
                if winner_dist < 5000:
                    winner_score += weight / (100.0 + winner_dist)
            
            print(f"\n    Score comparison (before penalty):")
            print(f"      {leak_node} (actual leak): {leak_score:.4f}")
            print(f"      {winner} (chosen):         {winner_score:.4f}")

if __name__ == "__main__":
    run_analysis()