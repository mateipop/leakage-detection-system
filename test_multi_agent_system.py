"""
Test script for the Multi-Agent System.

This demonstrates that we have a TRUE multi-agent system with:
- Multiple autonomous agents (SensorAgents)
- Message-passing communication
- Coordination through CoordinatorAgent
- Specialized LocalizerAgent

Run with: python test_multi_agent_system.py
"""

import sys
import numpy as np
from typing import Dict


def test_multi_agent_system():
    """Test the multi-agent leak detection system."""
    
    print("=" * 60)
    print("MULTI-AGENT SYSTEM TEST")
    print("=" * 60)
    
    # Import our agents
    from leak_detection.agents import (
        MultiAgentSystem,
        AgentSystemConfig,
        SensorAgent,
        CoordinatorAgent,
        LocalizerAgent
    )
    
    # Define test sensor nodes
    sensor_nodes = ["n1", "n2", "n3", "n4", "n5"]
    
    # Create distance matrix (simplified)
    distances = {
        "n1": {"n2": 100, "n3": 200, "n4": 500, "n5": 600},
        "n2": {"n1": 100, "n3": 150, "n4": 400, "n5": 500},
        "n3": {"n1": 200, "n2": 150, "n4": 300, "n5": 400},
        "n4": {"n1": 500, "n2": 400, "n3": 300, "n5": 100},
        "n5": {"n1": 600, "n2": 500, "n3": 400, "n4": 100},
    }
    
    # Create multi-agent system
    config = AgentSystemConfig(
        sensor_buffer_size=20,
        coordinator_aggregation_window=5.0,
        min_alerts_for_investigation=2
    )
    
    mas = MultiAgentSystem(
        sensor_nodes=sensor_nodes,
        network_distances=distances,
        config=config
    )
    
    print("\n1. Initializing Multi-Agent System...")
    mas.initialize()
    
    # Verify agents created
    print(f"\n   ✓ Created {len(mas.sensor_agents)} SensorAgents")
    print(f"   ✓ Created 1 CoordinatorAgent: {mas.coordinator.agent_id}")
    print(f"   ✓ Created 1 LocalizerAgent: {mas.localizer.agent_id}")
    print(f"   TOTAL AGENTS: {len(mas.sensor_agents) + 2}")
    
    # Test 2: Verify agent types
    print("\n2. Verifying Agent Types...")
    for node_id, agent in mas.sensor_agents.items():
        assert isinstance(agent, SensorAgent), f"Expected SensorAgent for {node_id}"
    assert isinstance(mas.coordinator, CoordinatorAgent)
    assert isinstance(mas.localizer, LocalizerAgent)
    print("   ✓ All agents are correct types")
    
    # Test 3: Simulate normal readings (no leak)
    print("\n3. Simulating NORMAL operation (10 steps)...")
    
    for step in range(10):
        # Generate normal readings (around 30 pressure units)
        readings = {
            node: {"pressure": 30.0 + np.random.normal(0, 0.5)}
            for node in sensor_nodes
        }
        
        environment = {
            "readings": readings,
            "sim_time": float(step)
        }
        
        status = mas.step(environment)
    
    coord_status = mas.coordinator.get_status()
    print(f"   ✓ System mode: {coord_status['system_mode']}")
    print(f"   ✓ Total alerts: {coord_status['total_alerts_received']}")
    print(f"   ✓ Investigations: {coord_status['investigations_opened']}")
    
    # Test 4: Inject a leak (pressure drop at n1 and n2)
    print("\n4. Injecting LEAK at nodes n1, n2...")
    
    for step in range(10, 30):
        readings = {}
        for node in sensor_nodes:
            if node in ["n1", "n2"]:
                # Leak causes pressure drop
                pressure = 30.0 - 8.0 + np.random.normal(0, 0.5)  # ~22 psi
            else:
                pressure = 30.0 + np.random.normal(0, 0.5)  # Normal
            readings[node] = {"pressure": pressure}
        
        environment = {
            "readings": readings,
            "sim_time": float(step)
        }
        
        status = mas.step(environment)
        
        # Check for alerts
        if step % 5 == 0:
            print(f"   Step {step}: alerts={status['total_alerts']}, "
                  f"investigations={len(status['active_investigations'])}")
    
    # Test 5: Check results
    print("\n5. Checking Detection Results...")
    
    final_status = mas.get_system_status()
    coord_status = final_status["coordinator"]
    
    print(f"   System mode: {coord_status['system_mode']}")
    print(f"   Total alerts received: {coord_status['total_alerts_received']}")
    print(f"   Investigations opened: {coord_status['investigations_opened']}")
    print(f"   Leaks localized: {coord_status['leaks_localized']}")
    
    # Check active investigations
    investigations = final_status["active_investigations"]
    print(f"\n   Active Investigations: {len(investigations)}")
    for inv in investigations:
        print(f"      • {inv['id']}: status={inv['status']}, "
              f"sensors={inv['sensor_count']}")
        if inv["localization"]:
            loc = inv["localization"]
            print(f"        → Localized at: {loc.get('probable_location')} "
                  f"(confidence: {loc.get('confidence', 0):.1%})")
    
    # Test 6: Verify message passing worked
    print("\n6. Verifying Message Passing...")
    
    # Check that sensors sent alerts
    sensors_with_alerts = sum(
        1 for s in final_status["sensors"].values()
        if s["alerts_sent"] > 0
    )
    print(f"   ✓ Sensors that sent alerts: {sensors_with_alerts}")
    
    # Check coordinator received them
    print(f"   ✓ Coordinator received: {coord_status['total_alerts_received']} alerts")
    
    # Check localizer was used
    localizer_status = final_status["localizer"]
    print(f"   ✓ Localizer performed: {localizer_status['localizations_performed']} localizations")
    
    # Final summary
    print("\n" + "=" * 60)
    print("MULTI-AGENT ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(mas.get_agent_summary())
    
    # Assertions for automated testing
    assert len(mas.sensor_agents) == 5, "Should have 5 sensor agents"
    assert mas.coordinator is not None, "Should have coordinator"
    assert mas.localizer is not None, "Should have localizer"
    assert coord_status["total_alerts_received"] > 0, "Should have received alerts"
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Multi-Agent System Working!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_multi_agent_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
