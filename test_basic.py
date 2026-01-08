#!/usr/bin/env python3
"""
Basic diagnostic test - check pressure changes with leak.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from leak_detection.simulation.network_simulator import NetworkSimulator
from leak_detection.config import SimulationConfig


def test_pressure_changes():
    """Test that leaks actually cause pressure drops."""
    print("Testing pressure changes with leak injection...")
    print("=" * 50)

    config = SimulationConfig()
    network = NetworkSimulator(config)

    print(f"Network type: {'MOCK' if network.is_mock else 'WNTR'}")
    print(f"Junctions: {len(network.junction_names)}")

    # Get baseline state
    network.run_simulation()
    baseline = network.get_state_at_time(3600)  # 1 hour

    print(f"\nBaseline pressures at t=3600s:")
    test_nodes = network.junction_names[:5]
    for node in test_nodes:
        print(f"  {node}: {baseline.pressures.get(node, 0):.2f} psi")

    # Inject leak at first node
    leak_node = test_nodes[0]
    print(f"\nInjecting leak at {leak_node}...")
    network.inject_leak(leak_node, leak_rate=10.0)  # Large leak

    # Re-run simulation
    network.run_simulation()
    after_leak = network.get_state_at_time(3600)

    print(f"\nPressures after leak injection:")
    for node in test_nodes:
        baseline_p = baseline.pressures.get(node, 0)
        after_p = after_leak.pressures.get(node, 0)
        diff = after_p - baseline_p
        print(f"  {node}: {after_p:.2f} psi (diff: {diff:+.2f})")

    # Check if pressure dropped
    leak_pressure_drop = baseline.pressures.get(leak_node, 0) - after_leak.pressures.get(leak_node, 0)
    print(f"\nPressure drop at leak node: {leak_pressure_drop:.2f} psi")

    if network.is_mock:
        print("\n[Using mock network - pressure changes are simulated]")
    else:
        print("\n[Using WNTR network - pressure changes from hydraulic simulation]")

    if abs(leak_pressure_drop) > 0.1:
        print("[OK] Leak causes measurable pressure change")
    else:
        print("[WARN] Leak did not cause significant pressure change!")


if __name__ == "__main__":
    test_pressure_changes()
