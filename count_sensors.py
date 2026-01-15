
import os

filepath = 'data/L-TOWN.inp'

pressure_sensors = set()
amr_nodes = set()

with open(filepath, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('['):
            continue

        if ';PRESSURE SENSOR' in line or ';AMR & PRESSURE SENSOR' in line:
            parts = line.split()
            if parts:
                node_id = parts[0]
                pressure_sensors.add(node_id)

        if ';AMR' in line:
            parts = line.split()
            if parts:
                node_id = parts[0]
                amr_nodes.add(node_id)

monitored_nodes = pressure_sensors.union(amr_nodes)

print(f"Pressure Sensors: {len(pressure_sensors)}")
print(f"AMR Nodes: {len(amr_nodes)}")
print(f"Total Monitored Nodes (Union): {len(monitored_nodes)}")
