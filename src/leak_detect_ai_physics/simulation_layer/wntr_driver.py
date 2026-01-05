import json
import time

import redis
import wntr

from leak_detect_ai_physics import config


def run_simulation():
    wn = wntr.network.WaterNetworkModel(config.NETWORK_NAME)

    leak_node = wn.get_node(config.LEAK_NODE_ID)
    leak_start = config.LEAK_START_SECONDS
    leak_end = config.LEAK_END_SECONDS
    leak_node.add_leak(wn, area=0.05, start_time=leak_start, end_time=leak_end)

    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 900

    print("Running WNTR Hydraulic Engine...")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    r = redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        decode_responses=True,
    )

    pressure_results = results.node["pressure"]
    head_results = results.node["head"]
    demand_results = results.node["demand"]

    monitored_sensors = list(pressure_results.columns)

    print("Starting real-time stream to Data Layer...")

    # Iterate through each time step to simulate a live data feed
    for timestamp, row in pressure_results.iterrows():
        leak_active = leak_start <= timestamp <= leak_end
        for sensor_id in monitored_sensors:
            node = wn.get_node(sensor_id)
            # Prepare the telemetry packet
            payload = {
                "sensor_id": f"SENSOR_{sensor_id}",
                "timestamp": int(timestamp),
                "pressure": float(row[sensor_id]),
                "head": float(head_results.loc[timestamp, sensor_id]),
                "demand": float(demand_results.loc[timestamp, sensor_id]),
                "elevation": float(getattr(node, "elevation", 0.0)),
                "network": config.NETWORK_NAME,
                "leak_active": leak_active,
                "leak_node": config.LEAK_NODE_ID,
            }

            r.publish(config.INPUT_CHANNEL, json.dumps(payload))

        print(
            "Time: {hours:>4.1f} hrs | Data sent for sensors: {count}".format(
                hours=timestamp / 3600,
                count=len(monitored_sensors),
            )
        )
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"Simulation error: {e}")
