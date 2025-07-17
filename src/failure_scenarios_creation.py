import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import defaultdict
from datetime import datetime


def parse_text_file(file_path):
    G = nx.Graph()
    depots = []
    battery_capacity = None
    with open(file_path, 'r') as f:
        lines = f.readlines()
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('NAME'):
            continue
        elif line.startswith('NUMBER OF VERTICES'):
            num_vertices = int(line.split(':')[1].strip())
            G.add_nodes_from(range(1, num_vertices + 1))
        elif line.startswith('VEHICLE CAPACITY'):
            # battery capacity is given in time units
            battery_capacity = float(line.split(':')[1].strip())
        elif line.startswith('LIST_REQUIRED_EDGES:'):
            current_section = 'required_edges'
        elif line.startswith('LIST_NON_REQUIRED_EDGES:'):
            current_section = 'non_required_edges'
        elif line.startswith('FAILURE_SCENARIO:'):
            current_section = 'failure_scenario'
        elif line.startswith('DEPOT:'):
            depots_line = line.split(':', 1)[1].strip()
            depots = [int(x.strip()) for x in depots_line.split(',')]
        else:
            if current_section in ('required_edges', 'non_required_edges'):
                u_v, rest = line.split(') ', 1)
                u_v = u_v.strip('(')
                u, v = map(int, u_v.split(','))
                weight = extract_weight(rest)
                required = (current_section == 'required_edges')
                G.add_edge(u, v, weight=weight, required=required)
    if battery_capacity is None:
        raise ValueError('VEHICLE CAPACITY not found in file')
    return G, depots, battery_capacity

def extract_weight(text):
    """
    Extract numeric weight from a string containing 'edge weight' or 'cost'.
    """
    text = text.strip()
    if 'edge weight' in text:
        value = text.split('edge weight', 1)[1].strip()
    elif 'cost' in text:
        value = text.split('cost', 1)[1].strip()
    else:
        return 1.0
    try:
        return float(value)
    except ValueError:
        return 1.0

def compute_coverage(G, radius):
    """
    For each node, compute the set of nodes reachable within 'radius' using Dijkstra.
    """
    coverage = {}
    for node in G.nodes():
        lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=radius, weight='travel_time')
        coverage[node] = set(lengths.keys())
    return coverage

def select_depots(G, battery_capacity):
    """
    Greedily select depot locations so that every node is within radius = battery_capacity/2.
    """
    # annotate edges with travel_time = weight
    for u, v, data in G.edges(data=True):
        data['travel_time'] = data.get('weight', 1.0)

    radius = battery_capacity / 2.0
    coverage = compute_coverage(G, radius)
    all_nodes = set(G.nodes())
    selected = set()
    covered = set()

    while covered != all_nodes:
        best_node, best_gain = None, -1
        for node in G.nodes():
            if node in selected:
                continue
            gain = len(coverage[node] - covered)
            if gain > best_gain:
                best_gain, best_node = gain, node
        if best_node is None:
            break
        selected.add(best_node)
        covered |= coverage[best_node]

    return selected

def update_all_instances():
    """
    Iterate over all failure scenarios, compute new depot placements,
    set number of vehicles = number of depots, and write updated .txt files.
    """
    instances = {
        'gdb': 37,
        'bccm': 108,
        'eglese': 112
    }
    input_base = "Failure_Scenarios"
    output_base = "Updated_Failure_Scenarios"

    os.makedirs(output_base, exist_ok=True)

    for inst, max_num in instances.items():
        in_folder = os.path.join(input_base, f"{inst}_failure_scenarios")
        out_folder = os.path.join(output_base, f"{inst}_failure_scenarios")
        os.makedirs(out_folder, exist_ok=True)

        for scenario in range(1, max_num + 1):
            infile = os.path.join(in_folder, f"{inst}.{scenario}.txt")
            if not os.path.exists(infile):
                print(f"[!] Missing file: {infile}, skipping.")
                continue

            # parse and compute new depots
            G, _ , battery_capacity = parse_text_file(infile)
            new_depots = select_depots(G, battery_capacity)
            num_vehicles = len(new_depots)

            # read original lines and strip old DEPOT and VEHICLE COUNT lines
            with open(infile, 'r') as f:
                lines = f.readlines()
            filtered = [
                l for l in lines
                if not l.strip().startswith('DEPOT:')
                and not l.strip().startswith('NUMBER OF VEHICLES')
            ]

            # prepare new header lines
            vehicle_line = f"NUMBER OF VEHICLES: {num_vehicles}\n"
            depot_line   = f"DEPOT: {','.join(str(d) for d in sorted(new_depots))}\n"

            # find where to insert vehicle count (after VEHICLE CAPACITY)
            vc_idx = next(
                (i for i, l in enumerate(filtered) if l.strip().startswith('VEHICLE CAPACITY')),
                None
            )
            # if vc_idx is not None:
            #     filtered.insert(vc_idx + 1, vehicle_line)
            # else:
            #     # fallback to top
            #     filtered.insert(0, vehicle_line)

            # find where to insert DEPOT line (before first edge line)
            edge_idx = next(
                (i for i, l in enumerate(filtered) if l.strip().startswith('(')),
                len(filtered)
            )
            # filtered.insert(edge_idx, depot_line)
            filtered.append(vehicle_line)
            filtered.append(depot_line)
            # write out updated file
            outfile = os.path.join(out_folder, f"{inst}.{scenario}.txt")
            with open(outfile, 'w') as f:
                f.writelines(filtered)

            print(f"Updated {inst}.{scenario} → {outfile}")

def visualize_routes(G, routes, title_prefix, depots=None):
    """
    Plot each vehicle's set of trips side by side to show continuity.
    Each subplot shows one trip with direction arrows on the graph.
    """
    # Compute a fixed layout for consistent node positions across subplots
    pos = nx.spring_layout(G, seed=42)

    # Determine node colors if depots are known
    if depots is not None:
        node_colors = ['orange' if n in depots else 'lightgreen' for n in G.nodes()]
    else:
        node_colors = ['lightgreen' for _ in G.nodes()]

    # Plot per vehicle
    for vidx, vehicle_trips in enumerate(routes, start=1):
        n_trips = len(vehicle_trips)
        fig, axes = plt.subplots(1, n_trips, figsize=(4 * n_trips, 4), squeeze=False)
        axes = axes[0]  # get the list of axes

        for tidx, trip in enumerate(vehicle_trips):
            ax = axes[tidx]
            # Draw base graph lightly
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=75, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='lightgray', ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=7, font_color='black', ax=ax)

            # Draw this trip with arrows
            trip_edges = [(trip[i], trip[i+1]) for i in range(len(trip) - 1)]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=trip_edges,
                edge_color='blue',
                arrows=True,
                arrowstyle='-|>',
                arrowsize=12,
                width=2,
                ax=ax
            )

            ax.set_title(f"{title_prefix} — Vehicle {vidx} Trip {tidx+1}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

def visualize_graph(G, depots, title):
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['orange' if node in depots else 'lightgreen' for node in G.nodes()]
    required_edges = [(u, v) for u, v in G.edges() if G[u][v].get('required', False)]
    non_required_edges = [(u, v) for u, v in G.edges() if not G[u][v].get('required', False)]
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=75)
    nx.draw_networkx_edges(G, pos, edgelist=required_edges, edge_color='red', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=non_required_edges, edge_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
    nx.draw_networkx_labels(G, pos, font_size=7, font_color='black')

    depot_patch = mpatches.Patch(color='orange', label='Depot Nodes')
    node_patch = mpatches.Patch(color='lightgreen', label='Other Nodes')
    required_edge_line = mlines.Line2D([], [], color='red', label='Required Edges', linewidth=2)
    non_required_edge_line = mlines.Line2D([], [], color='black', label='Non-required Edges')
    plt.legend(handles=[depot_patch, node_patch, required_edge_line, non_required_edge_line], loc='best')

    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # choose instance
    while True:
        instance_name = input("Enter instance name (gdb, bccm, eglese): ").strip().lower()
        if instance_name in ['gdb', 'bccm', 'eglese']:
            break
        print("Invalid instance name.")

    max_scenario = {'gdb': 37, 'bccm': 108, 'eglese': 112}[instance_name]
    while True:
        try:
            n = int(input(f"Enter failure scenario number [1-{max_scenario}]: ").strip())
            if 1 <= n <= max_scenario:
                scenario_number = n
                break
        except ValueError:
            pass
        print("Invalid scenario number.")


    folder = f"../New_Failure_Scenarios/{instance_name}_failure_scenarios"
    file_path = os.path.join(folder, f"{instance_name}.{scenario_number}.txt")
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    # parse original scenario (graph + original depots + capacity)
    G, new_depots, battery_capacity = parse_text_file(file_path)
    recharge_time = 2*battery_capacity
    print(f'battery capacity - {battery_capacity}')
    print(f'recharge time - {recharge_time}')
    print(G, type(G))
    # visualize before placement
    visualize_graph(
        G,
        new_depots,
        f"Scenario {instance_name}.{scenario_number} — Heuristic Depot Placement - ({len(new_depots)} depots)"
    )

    # load and visualize solution routes
    sol_folder = f"../results/Instance_results_without_failure/{instance_name.upper()}_Failure_Scenarios_Results/results/{instance_name}_failure_scenarios_solutions"
    sol_file = os.path.join(sol_folder, f"{scenario_number}.npy")
    if os.path.exists(sol_file):
        routes = np.load(sol_file, allow_pickle=True).tolist()
        print(routes)
        vehicle_route_times = []
        for vehicle_routes in routes:
            total_time = 0
            for i, trip in enumerate(vehicle_routes):
                trip_time = 0
                for u, v in zip(trip[:-1], trip[1:]):
                    if G.has_edge(u, v):
                        trip_time += G[u][v].get('weight', 1)
                    else:
                        print(f"[!] Warning: Edge ({u}, {v}) not found in graph.")
                total_time += trip_time
                print(f"Trip {i+1} time: {trip_time}")
                if i < len(vehicle_routes) - 1:
                    total_time += recharge_time  # Recharge time between trips
            vehicle_route_times.append(total_time)

        print("Vehicle route times:", vehicle_route_times)

        visualize_routes(
            G, routes,
            f"Scenario {instance_name}.{scenario_number} Routes",
            depots=new_depots
        )
    else:
        print(f"[!] Solution file not found: {sol_file}")

def plot_failure_histograms(failure_counts_by_instance, output_dir="../New_Failure_Scenarios/Failure_Distributions"):
        os.makedirs(output_dir, exist_ok=True)
        for instance_name, failure_counts in failure_counts_by_instance.items():
            histogram = defaultdict(int)
            for count in failure_counts:
                histogram[count] += 1

            x = sorted(histogram.keys())
            y = [histogram[k] for k in x]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(x, y, color='skyblue', edgecolor='black')
            plt.xlabel("Number of Failed Vehicles")
            plt.ylabel("Number of Instances")
            plt.title(f"Failure Distribution for {instance_name.upper()}")
            plt.xticks(range(min(x), max(x)+1))
            # plt.xlim(min(x) - 0.5, max(x) + 0.5)
            plt.ylim(0, max(y) + 5)
            # plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            # Add numbers on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=9)


            out_path = os.path.join(output_dir, f"{instance_name}_failure_histogram.png")
            plt.savefig(out_path)
            plt.close()

def augment_all_failure_scenarios():

    random.seed(datetime.now())

    instances = {'gdb': 37, 'bccm': 108, 'eglese': 112}
    base_folder = "../New_Failure_Scenarios"
    sol_base = "../results/Instance_results_without_failure"

    failure_counts_by_instance = {inst: [] for inst in instances}

    def compute_vehicle_route_times(G, routes, recharge_time):
        vehicle_route_times = []
        for vehicle_routes in routes:
            total_time = 0
            for i, trip in enumerate(vehicle_routes):
                trip_time = sum(G[u][v]['weight'] for u, v in zip(trip[:-1], trip[1:]) if G.has_edge(u, v))
                total_time += trip_time
                if i < len(vehicle_routes) - 1:
                    total_time += recharge_time
            vehicle_route_times.append(total_time)
        return vehicle_route_times

    def generate_updated_failure_scenario(file_path, vehicle_route_times, output_base="../New_Failure_Scenarios"):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        failure_start_idx = next((i for i, l in enumerate(lines) if l.strip().startswith("FAILURE_SCENARIO:")), None)
        if failure_start_idx is None:
            print("FAILURE_SCENARIO not found.")
            return None, None

        failed_lines = lines[failure_start_idx + 1:]
        num_failed_vehicles_original = len([l for l in failed_lines if l.strip()])

        num_vehicles = next(
            int(l.strip().split(":")[1])
            for l in lines if l.strip().startswith("NUMBER OF VEHICLES:")
        )

        # if num_failed_vehicles_original > num_vehicles - 1:
        num_failed = random.randint(1, min(num_vehicles - 1, 6))
        # else:
            # num_failed = num_failed_vehicles_original

        failed_vehicle_ids = random.sample(range(1, num_vehicles + 1), num_failed)
        failure_lines = []
        for vid in failed_vehicle_ids:
            max_fail_time = max(3, int(vehicle_route_times[vid - 1] - 3))
            failure_time = random.randint(3, max_fail_time)
            failure_lines.append(f"Vehicle {vid} will fail in {failure_time} time units.\n")

        rel_path = os.path.relpath(file_path, "../Balanced_Failure_Scenarios")
        new_path = os.path.join(output_base, rel_path)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        updated_lines = lines[:failure_start_idx + 1] + failure_lines
        with open(new_path, 'w') as f:
            f.writelines(updated_lines)

        return new_path, failed_vehicle_ids


    for inst, max_n in instances.items():
        print(f"\n[===] Processing instance: {inst.upper()}")

        txt_folder = os.path.join(base_folder, f"{inst}_failure_scenarios")
        sol_folder = os.path.join(sol_base, f"{inst.upper()}_Failure_Scenarios_Results/results/{inst}_failure_scenarios_solutions")

        for scenario in range(1, max_n + 1):
            txt_path = os.path.join(txt_folder, f"{inst}.{scenario}.txt")
            sol_path = os.path.join(sol_folder, f"{scenario}.npy")

            if not os.path.exists(txt_path):
                print(f"[!] Missing .txt: {txt_path}")
                continue
            if not os.path.exists(sol_path):
                print(f"[!] Missing .npy: {sol_path}")
                continue

            try:
                G, _, battery_capacity = parse_text_file(txt_path)
                recharge_time = 2 * battery_capacity
                routes = np.load(sol_path, allow_pickle=True).tolist()
                vehicle_route_times = compute_vehicle_route_times(G, routes, recharge_time)
                new_path, failed_vehicle_ids = generate_updated_failure_scenario(txt_path, vehicle_route_times)
                if failed_vehicle_ids is not None:
                    failure_counts_by_instance[inst].append(len(failed_vehicle_ids))
            except Exception as e:
                print(f"[!] Error with {inst}.{scenario}: {e}")
                continue

    plot_failure_histograms(failure_counts_by_instance)

def regenerate_histograms_from_augmented_files(base_dir="../New_Failure_Scenarios"):
    import re
    from collections import defaultdict

    instance_map = {
        "gdb": 37,
        "bccm": 108,
        "eglese": 112
    }
    failure_counts_by_instance = {inst: [] for inst in instance_map}

    pattern = re.compile(r"Vehicle\s+(\d+)\s+will\s+fail\s+in\s+(\d+)\s+time units")

    for inst in instance_map:
        folder = os.path.join(base_dir, f"{inst}_failure_scenarios")
        if not os.path.isdir(folder):
            print(f"[!] Folder not found: {folder}")
            continue

        for scenario in range(1, instance_map[inst] + 1):
            file_path = os.path.join(folder, f"{inst}.{scenario}.txt")
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'r') as f:
                lines = f.readlines()

            in_failure_block = False
            fail_count = 0
            for line in lines:
                if "FAILURE_SCENARIO:" in line:
                    in_failure_block = True
                    continue
                if in_failure_block and pattern.search(line):
                    fail_count += 1

            failure_counts_by_instance[inst].append(fail_count)

    # Plot histograms
    plot_failure_histograms(failure_counts_by_instance)


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Visualize a specific scenario")
    print("2. Generate new failure scenarios for all instances")
    print("3. Regenerate histograms from existing scenarios")
    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        main()  # Visualize a scenario and generate failure scenario for it
    elif choice == "2":
        augment_all_failure_scenarios()  # Generate new failure scenarios across all instances
    elif choice == "3":
        regenerate_histograms_from_augmented_files()  # Regenerate histograms only
    else:
        print("Invalid choice.")