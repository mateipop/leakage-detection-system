
import wntr
import os

try:
    wn = wntr.network.WaterNetworkModel('data/L-TOWN.inp')
    print(f"Flow Units: {wn.options.hydraulic.flow_units}")
    print(f"Pressure Units (implied): {wntr.metrics.hydraulic.pressure(wn)}") # No easy way, but SI usually = Meters
except Exception as e:
    print(f"Error: {e}")
