import wntr

from leak_detect_ai_physics.simulation_layer import wntr_driver


def test_build_leak_plan_limits_and_ranges():
    wn = wntr.network.WaterNetworkModel("Net3")
    duration = 6 * 3600
    plan = wntr_driver.build_leak_plan(wn, duration, seed=7)

    assert (
        1 <= len(plan) <= (wntr_driver.MAX_JUNCTION_LEAKS + wntr_driver.MAX_PIPE_LEAKS)
    )
    junctions = set(wn.junction_name_list)
    pipes = set(wn.pipe_name_list)

    for leak in plan:
        assert leak["type"] in {"junction", "pipe"}
        assert leak["leak_node_id"] in junctions
        assert 0 < leak["start_time"] < duration
        assert leak["start_time"] < leak["end_time"] <= duration
        assert (
            wntr_driver.LEAK_AREA_RANGE[0]
            <= leak["area"]
            <= wntr_driver.LEAK_AREA_RANGE[1]
        )
        if leak["type"] == "junction":
            assert leak["node_id"] in junctions
            assert leak["pipe_id"] is None
        else:
            assert leak["pipe_id"] in pipes
            assert leak["new_pipe_id"] is not None
            assert leak["split_fraction"] is not None


def test_leak_active_sets_matches_plan():
    plan = [
        {"leak_node_id": "A", "pipe_id": None, "start_time": 10, "end_time": 20},
        {"leak_node_id": "B", "pipe_id": "P1", "start_time": 15, "end_time": 25},
    ]

    active, nodes, pipes = wntr_driver.leak_active_sets(plan, 5)
    assert active == []
    assert nodes == set()
    assert pipes == set()

    active, nodes, pipes = wntr_driver.leak_active_sets(plan, 12)
    assert nodes == {"A"}
    assert pipes == set()

    active, nodes, pipes = wntr_driver.leak_active_sets(plan, 20)
    assert nodes == {"A", "B"}
    assert pipes == {"P1"}

    active, nodes, pipes = wntr_driver.leak_active_sets(plan, 30)
    assert active == []


def test_node_and_link_attributes_include_expected_keys():
    wn = wntr.network.WaterNetworkModel("Net3")
    node = wn.get_node(wn.junction_name_list[0])
    link = wn.get_link(wn.pipe_name_list[0])

    node_attrs = wntr_driver.node_attributes(node)
    link_attrs = wntr_driver.link_attributes(link)

    assert set(node_attrs.keys()) == {
        "node_type",
        "elevation",
        "base_demand",
        "emitter_coefficient",
    }
    assert set(link_attrs.keys()) == {
        "link_type",
        "diameter",
        "length",
        "roughness",
        "minor_loss",
        "status",
        "initial_status",
    }
