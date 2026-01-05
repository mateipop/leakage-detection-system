from leak_detect_ai_physics.data_layer.dataset_builder import (
    _window_label,
    _window_leak_node,
)


def test_window_label_any_detects_leak():
    window = [
        {"leak_any": False},
        {"leak_any": True},
        {"leak_any": False},
    ]
    assert _window_label(window, mode="any") is True


def test_window_leak_node_picks_most_common():
    window = [
        {"active_leak_nodes": ["N1", "N2"]},
        {"active_leak_nodes": ["N1"]},
        {"active_leak_nodes": ["N3"]},
    ]
    assert _window_leak_node(window) == "N1"
