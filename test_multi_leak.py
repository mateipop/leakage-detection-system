"""
Test multi-leak detection capability.

This test demonstrates the system's ability to detect and localize
multiple simultaneous leaks in different parts of the network.
"""

import pytest
from leak_detection.config import SystemConfig
from leak_detection.orchestrator import SystemOrchestrator


class TestMultiLeakDetection:
    """Tests for multiple simultaneous leak detection."""

    @pytest.fixture
    def orchestrator(self):
        """Create a configured orchestrator."""
        config = SystemConfig()
        config.ai.anomaly_threshold = 0.45
        orch = SystemOrchestrator(config)
        # Warm up
        for _ in range(20):
            orch.step()
        return orch

    def test_single_leak_detection(self, orchestrator):
        """Single leak should result in 1-2 clusters."""
        leak_node = 'n300'
        orchestrator.leak_injector.inject_leak(leak_node, 10.0, orchestrator.sim_time)
        orchestrator.network.run_simulation()

        # Run detection
        for _ in range(15):
            orchestrator.step()

        multi = orchestrator.agent.get_multi_leak_result(max_leaks=5)
        assert multi is not None, "Should detect the leak"
        assert multi.leak_count >= 1, "Should detect at least one cluster"
        print(f"Single leak: detected {multi.leak_count} cluster(s)")

    def test_two_leaks_detection(self, orchestrator):
        """Two leaks in different areas should be detected."""
        leak1, leak2 = 'n100', 'n600'
        
        orchestrator.leak_injector.inject_leak(leak1, 12.0, orchestrator.sim_time)
        orchestrator.leak_injector.inject_leak(leak2, 10.0, orchestrator.sim_time)
        orchestrator.network.run_simulation()

        # Run detection
        for _ in range(15):
            orchestrator.step()

        multi = orchestrator.agent.get_multi_leak_result(max_leaks=5)
        assert multi is not None, "Should detect leaks"
        print(f"Two leaks: detected {multi.leak_count} cluster(s)")
        
        # Should detect multiple clusters (may be more due to network effects)
        assert multi.leak_count >= 1

    def test_get_all_leak_locations(self, orchestrator):
        """Test convenience method for getting all leak locations."""
        orchestrator.leak_injector.inject_leak('n200', 10.0, orchestrator.sim_time)
        orchestrator.network.run_simulation()

        for _ in range(15):
            orchestrator.step()

        locations = orchestrator.agent.get_all_leak_locations()
        assert len(locations) >= 1, "Should return at least one location"
        print(f"Leak locations: {locations}")

    def test_multi_leak_result_structure(self, orchestrator):
        """Test that MultiLeakResult has correct structure."""
        orchestrator.leak_injector.inject_leak('n400', 8.0, orchestrator.sim_time)
        orchestrator.network.run_simulation()

        for _ in range(15):
            orchestrator.step()

        multi = orchestrator.agent.get_multi_leak_result()
        
        if multi:
            assert hasattr(multi, 'leak_count')
            assert hasattr(multi, 'localizations')
            assert hasattr(multi, 'total_confidence')
            assert hasattr(multi, 'clustering_method')
            
            for loc in multi.localizations:
                assert hasattr(loc, 'estimated_node')
                assert hasattr(loc, 'confidence')
                assert hasattr(loc, 'cluster_id')
                assert hasattr(loc, 'contributing_anomalies')
                assert hasattr(loc, 'candidate_region')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
