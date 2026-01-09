from leak_detect_ai_physics.data_layer.dataset_builder import _window_leak_nodes


def test_window_leak_nodes_collects_unique_ids():
    window = [
        {"active_leak_nodes": ["N1", "N2"]},
        {"active_leak_nodes": ["N1"]},
        {"active_leak_nodes": ["N3"]},
    ]
    assert _window_leak_nodes(window) == {"N1", "N2", "N3"}
