import wntr
import redis
import json
import time

# --- CONFIGURATION ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
TOPIC = "sensor_telemetry"


def run_simulation():
    # 1. Load the internal Net3 model (a standard benchmark network)
    wn = wntr.network.WaterNetworkModel("Net3")

    # 2. Add a leak to node '123'
    # This will cause a pressure drop that your AI/Data layer should detect
    leak_node = wn.get_node("123")
    # Leak starts at 4 hours (14400 seconds) and ends at 10 hours
    leak_node.add_leak(wn, area=0.05, start_time=4 * 3600, end_time=10 * 3600)

    # 3. Set the simulation duration (e.g., 24 hours)
    wn.options.time.duration = 24 * 3600
    wn.options.time.hydraulic_timestep = 900

    # 4. Run the simulator (EPANET engine)
    print("⚙️ Running WNTR Hydraulic Engine...")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    # 5. Connect to Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

    # 6. Extract Pressure Results
    pressure_results = results.node["pressure"]

    # We will "watch" three specific nodes as sensors
    monitored_sensors = ["123", "161", "101"]

    print(f"📡 Starting Real-Time Stream to Data Layer...")

    # Iterate through each time step to simulate a live data feed
    for timestamp, row in pressure_results.iterrows():
        for sensor_id in monitored_sensors:
            # Prepare the telemetry packet
            payload = {
                "sensor_id": f"SENSOR_{sensor_id}",
                "timestamp": int(timestamp),
                "pressure": float(row[sensor_id]),
            }

            # Publish to the Data Layer
            r.publish(TOPIC, json.dumps(payload))

        # Slow down the loop so you can watch the output in the terminal
        print(
            f"🕒 Time: {timestamp/3600:>4.1f} hrs | Data sent for sensors: {monitored_sensors}"
        )
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        print(f"❌ Simulation Error: {e}")
