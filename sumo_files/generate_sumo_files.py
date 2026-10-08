"""
Script to generate all required SUMO simulation files for the
2x2 grid traffic network (4 signalized intersections).

Run once before training:
    python sumo_files/generate_sumo_files.py

Uses netconvert (ships with SUMO) to produce a geometrically valid network.
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.vehicle_profiles import VehicleEmissionModel

# ---------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------
CELL_LEN = 300    # metres between intersections
N_ROWS = 2
N_COLS = 2
SPEED = 13.89     # m/s (~50 km/h)
LANES = 2         # lanes per direction
TL_IDS = [f"TL{r}{c}" for r in range(N_ROWS) for c in range(N_COLS)]


def _prettify(elem):
    rough = ET.tostring(elem, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="    ", encoding=None)


# ---------------------------------------------------------------------------
# Build .nod.xml  (nodes / junctions for netconvert)
# ---------------------------------------------------------------------------
def build_nodes():
    root = ET.Element("nodes")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            ET.SubElement(root, "node",
                          id=f"TL{r}{c}",
                          x=str((c + 1) * CELL_LEN),
                          y=str((r + 1) * CELL_LEN),
                          type="traffic_light")

    for c in range(N_COLS):
        ET.SubElement(root, "node",
                      id=f"P_bot{c}",
                      x=str((c + 1) * CELL_LEN), y="0",
                      type="dead_end")
        ET.SubElement(root, "node",
                      id=f"P_top{c}",
                      x=str((c + 1) * CELL_LEN),
                      y=str((N_ROWS + 1) * CELL_LEN),
                      type="dead_end")
    for r in range(N_ROWS):
        ET.SubElement(root, "node",
                      id=f"P_lft{r}",
                      x="0", y=str((r + 1) * CELL_LEN),
                      type="dead_end")
        ET.SubElement(root, "node",
                      id=f"P_rgt{r}",
                      x=str((N_COLS + 1) * CELL_LEN),
                      y=str((r + 1) * CELL_LEN),
                      type="dead_end")

    path = os.path.join(BASE_DIR, "network.nod.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {path}")
    return path


# ---------------------------------------------------------------------------
# Build .edg.xml  (edges for netconvert)
# ---------------------------------------------------------------------------
def build_edges():
    root = ET.Element("edges")
    edge_ids = []

    def add_edge(eid, frm, to):
        edge_ids.append(eid)
        ET.SubElement(root, "edge",
                      id=eid,
                      **{"from": frm, "to": to},
                      numLanes=str(LANES),
                      speed=str(SPEED),
                      priority="7")

    for r in range(N_ROWS):
        for c in range(N_COLS):
            if c + 1 < N_COLS:
                add_edge(f"e_{r}{c}_E", f"TL{r}{c}", f"TL{r}{c+1}")
                add_edge(f"e_{r}{c+1}_W", f"TL{r}{c+1}", f"TL{r}{c}")
            if r + 1 < N_ROWS:
                add_edge(f"e_{r}{c}_N", f"TL{r}{c}", f"TL{r+1}{c}")
                add_edge(f"e_{r+1}{c}_S", f"TL{r+1}{c}", f"TL{r}{c}")

    for c in range(N_COLS):
        add_edge(f"e_bot{c}_in",  f"P_bot{c}", f"TL0{c}")
        add_edge(f"e_bot{c}_out", f"TL0{c}",   f"P_bot{c}")
        add_edge(f"e_top{c}_in",  f"P_top{c}", f"TL{N_ROWS-1}{c}")
        add_edge(f"e_top{c}_out", f"TL{N_ROWS-1}{c}", f"P_top{c}")
    for r in range(N_ROWS):
        add_edge(f"e_lft{r}_in",  f"P_lft{r}", f"TL{r}0")
        add_edge(f"e_lft{r}_out", f"TL{r}0",   f"P_lft{r}")
        add_edge(f"e_rgt{r}_in",  f"P_rgt{r}", f"TL{r}{N_COLS-1}")
        add_edge(f"e_rgt{r}_out", f"TL{r}{N_COLS-1}", f"P_rgt{r}")

    path = os.path.join(BASE_DIR, "network.edg.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {path}")
    return path, edge_ids


# ---------------------------------------------------------------------------
# Build .tll.xml  (traffic light logic for netconvert)
# ---------------------------------------------------------------------------
def build_tll():
    root = ET.Element("additional")

    for tid in TL_IDS:
        tl = ET.SubElement(root, "tlLogic",
                           id=tid, type="static",
                           programID="0", offset="0")
        ET.SubElement(tl, "phase", duration="31", state="GGrrGGrr")
        ET.SubElement(tl, "phase", duration="4",  state="yyrryyrr")
        ET.SubElement(tl, "phase", duration="31", state="rrGGrrGG")
        ET.SubElement(tl, "phase", duration="4",  state="rryyrryy")

    path = os.path.join(BASE_DIR, "traffic_lights_logic.tll.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {path}")
    return path


# ---------------------------------------------------------------------------
# Run netconvert to produce network.net.xml
# ---------------------------------------------------------------------------
def run_netconvert(nod_path, edg_path):
    net_out = os.path.join(BASE_DIR, "network.net.xml")

    sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
    netconvert = os.path.join(sumo_home, "bin", "netconvert.exe")
    if not os.path.exists(netconvert):
        netconvert = "netconvert"

    cmd = [
        netconvert,
        "--node-files", nod_path,
        "--edge-files", edg_path,
        "--output-file", net_out,
        "--no-turnarounds", "true",
        "--tls.default-type", "static",
        "--no-warnings", "true",
    ]

    print(f"[netconvert] running ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("netconvert stdout:\n", result.stdout)
        print("netconvert stderr:\n", result.stderr)
        sys.exit(f"netconvert failed (exit {result.returncode})")
    print(f"[OK] {net_out}")
    return net_out


# ---------------------------------------------------------------------------
# Build routes.rou.xml
# ---------------------------------------------------------------------------
def build_routes(edge_ids):
    """Create vehicle flows entering from perimeter edges with explicit routes."""
    vehicle_model = VehicleEmissionModel(
        os.path.join(PROJECT_DIR, "data", "CO2 and fuel efficiency data", "CO2_and_fuel_efficiency.csv")
    )

    root = ET.Element("routes",
                      **{"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                         "xsi:noNamespaceSchemaLocation":
                         "http://sumo.dlr.de/xsd/routes_file.xsd"})

    mix_id = "vehicle_mix"
    mix = ET.SubElement(root, "vTypeDistribution", id=mix_id)
    for entry in vehicle_model.build_type_distribution():
        ET.SubElement(mix, "vType", id=entry["id"], probability=str(entry["probability"]))

    SIM_END = 3600
    FLOW_PER_ENTRY = 200

    # Explicit straight-through routes: bottom→top, top→bottom, left→right, right→left
    route_defs = []
    for c in range(N_COLS):
        # bottom-in → TL rows going north → top-out
        ns_edges = " ".join(
            [f"e_bot{c}_in"] +
            [f"e_{r}{c}_N" for r in range(N_ROWS - 1)] +
            [f"e_top{c}_out"]
        )
        route_defs.append((f"route_bot{c}", ns_edges, f"e_bot{c}_in"))

        # top-in → TL rows going south → bot-out
        sn_edges = " ".join(
            [f"e_top{c}_in"] +
            [f"e_{r}{c}_S" for r in range(N_ROWS - 1, 0, -1)] +
            [f"e_bot{c}_out"]
        )
        route_defs.append((f"route_top{c}", sn_edges, f"e_top{c}_in"))

    for r in range(N_ROWS):
        # left-in → TL cols going east → right-out
        ew_edges = " ".join(
            [f"e_lft{r}_in"] +
            [f"e_{r}{c}_E" for c in range(N_COLS - 1)] +
            [f"e_rgt{r}_out"]
        )
        route_defs.append((f"route_lft{r}", ew_edges, f"e_lft{r}_in"))

        # right-in → TL cols going west → left-out
        we_edges = " ".join(
            [f"e_rgt{r}_in"] +
            [f"e_{r}{c}_W" for c in range(N_COLS - 1, 0, -1)] +
            [f"e_lft{r}_out"]
        )
        route_defs.append((f"route_rgt{r}", we_edges, f"e_rgt{r}_in"))

    for route_id, edges_str, _ in route_defs:
        ET.SubElement(root, "route", id=route_id, edges=edges_str)

    for i, (route_id, _, _) in enumerate(route_defs):
        ET.SubElement(root, "flow",
                      id=f"flow_{i}",
                      type=mix_id,
                      route=route_id,
                      begin="0",
                      end=str(SIM_END),
                      vehsPerHour=str(FLOW_PER_ENTRY),
                      departLane="best",
                      departSpeed="random")

    out = os.path.join(BASE_DIR, "routes.rou.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# Build traffic_lights.add.xml  (additional detectors)
# ---------------------------------------------------------------------------
def build_additionals():
    root = ET.Element("additional",
                      **{"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                         "xsi:noNamespaceSchemaLocation":
                         "http://sumo.dlr.de/xsd/additional_file.xsd"})

    for c in range(N_COLS):
        ET.SubElement(root, "inductionLoop",
                      id=f"det_bot{c}",
                      lane=f"e_bot{c}_in_0",
                      pos="-10",
                      freq="60",
                      file=f"det_bot{c}.xml")
    for r in range(N_ROWS):
        ET.SubElement(root, "inductionLoop",
                      id=f"det_lft{r}",
                      lane=f"e_lft{r}_in_0",
                      pos="-10",
                      freq="60",
                      file=f"det_lft{r}.xml")

    out = os.path.join(BASE_DIR, "traffic_lights.add.xml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
# Build config.sumocfg
# ---------------------------------------------------------------------------
def build_sumocfg():
    root = ET.Element("configuration",
                      **{"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                         "xsi:noNamespaceSchemaLocation":
                         "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"})

    inp = ET.SubElement(root, "input")
    ET.SubElement(inp, "net-file",        value="network.net.xml")
    ET.SubElement(inp, "route-files",     value="routes.rou.xml")
    ET.SubElement(inp, "additional-files", value="traffic_lights.add.xml")

    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin",  value="0")
    ET.SubElement(time, "end",    value="3600")
    ET.SubElement(time, "step-length", value="1")

    report = ET.SubElement(root, "report")
    ET.SubElement(report, "no-warnings", value="true")
    ET.SubElement(report, "no-step-log", value="true")

    out = os.path.join(BASE_DIR, "config.sumocfg")
    with open(out, "w", encoding="utf-8") as f:
        f.write(_prettify(root))
    print(f"[OK] {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating SUMO files …")
    nod_path = build_nodes()
    edg_path, edge_ids = build_edges()
    tll_path = build_tll()  # kept for reference but not passed to netconvert
    run_netconvert(nod_path, edg_path)
    build_routes(edge_ids)
    build_additionals()
    build_sumocfg()
    print("\nDone!  All files written to:", BASE_DIR)
    print("Run 'python main.py' to start training.")
