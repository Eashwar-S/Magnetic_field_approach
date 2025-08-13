import os
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def read_dat_file(file_path):
    graph = nx.Graph()
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Find the start of the required edges section
    try:
        start_index = next(i for i, line in enumerate(lines) if 'LISTA_ARISTAS_REQ' in line) + 1
    except StopIteration:
        raise ValueError("Could not find 'LISTA_ARISTAS_REQ' in the file")

    # Find the start of the non-required edges section
    try:
        end_index = next(i for i, line in enumerate(lines[start_index:], start=start_index) if 'LISTA_ARISTAS_NOREQ' in line)
    except StopIteration:
        raise ValueError("Could not find 'LISTA_ARISTAS_NOREQ' in the file")
    
    # Process required edges
    for line in lines[start_index:end_index]:
        parts = line.strip().split()
        if len(parts) >= 7:  # Ensure we have enough parts
            try:
                node1 = int(parts[1].strip('( ,)'))
                node2 = int(parts[2].strip('( ,)'))
                weight = int(parts[4])
                demand = int(parts[6])
                graph.add_edge(node1, node2, weight=weight, demand=demand, required=True)
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()}. Error: {e}")
    
    # Process non-required edges
    for line in lines[end_index+1:]:
        if 'DEPOSITO' in line:
            break
        parts = line.strip().split()
        if len(parts) >= 5:  # Ensure we have enough parts
            try:
                node1 = int(parts[1].strip('( ,)'))
                node2 = int(parts[2].strip('( ,)'))
                weight = int(parts[4])
                graph.add_edge(node1, node2, weight=weight, required=False)
            except ValueError as e:
                print(f"Skipping invalid line: {line.strip()}. Error: {e}")
    
    return graph


def plot_graph(graph, instance):
    pos = nx.spring_layout(graph)
    
    
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    # print(edge_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    # required_edges = [(u, v) for (u, v, d) in graph.edges(data=True) if d['required']]
    nx.draw_networkx_edges(graph, pos, edgelist=instance['required_edges'], edge_color='r', width=4)


    # print(f'depot nodes - {instance["depots"]}')

    node_color = ["lightblue"] * int(graph.number_of_nodes())
    depot_node_color = node_color
    # print(f'len of node_color - {node_color}')
    # print(f'graph nodes - {int(graph.number_of_nodes())}')
    for i in range(1, len(node_color) + 1):
        if i in instance['depots']:
            depot_node_color[i - 1] = "g"
    # nx.draw_networkx(graph, pos, node_color=depot_node_color)
    nx.draw(graph, pos, with_labels=True, node_color=depot_node_color, node_size=500, font_size=8)
    
    plt.title("MD-RPP-RRV Graph Instance")
    plt.axis('off')
    plt.show()

def create_instance(graph):
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    
    # print(f'nodes list - {nodes}')
    # print(f'edges list - {edges}')
    # Randomly choose one-fifth of the nodes as depots
    num_depots = max(1, len(nodes) // 5)
    depots = list(np.array(np.linspace(1, len(nodes), num_depots), dtype=np.int64))#random.sample(nodes, num_depots)
    
    # Randomly choose one-third of the edges as required edges
    num_required_edges = max(1, len(edges) // 3)
    required_edges = random.sample(edges, num_required_edges)
    
    # Set number of vehicles
    num_vehicles = max(1, len(required_edges) // 2)
    
    # Set vehicle capacity
    max_edge_weight = max(data['weight'] for _, _, data in graph.edges(data=True))
    vehicle_capacity = 2 * max_edge_weight
    
    # Create instance dictionary
    instance = {
        'required_edges': required_edges,
        'depots': depots,
        'num_vehicles': num_vehicles,
        'vehicle_capacity': vehicle_capacity
    }
    
    return instance

def process_dat_files(input_folder, output_folder, output_info_folder):
    
    vehCap = [0] * 6
    reqNod = [0] * 6
    deNo = [0] * 6

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each .dat file in the input folder
    ind = 0
    for _ , filename in enumerate(os.listdir(input_folder)):
        print(filename, ind)
        if filename.endswith('.dat'):# and filename == "egl-s4-A.dat":
            
            dat_file_path = os.path.join(input_folder, filename)
            net_file_path = os.path.join(output_folder, str(ind) + '.net')

            try:
                # Read the .dat file and create the graph
                graph = read_dat_file(dat_file_path)

                # # Save the graph as a .net file
                nx.write_pajek(graph, net_file_path)
            
            
                # Plot the graph

                verify_graph = nx.Graph(nx.read_pajek(net_file_path))
                mapping = {str(i): i for i in range(1, verify_graph.number_of_nodes()+1)}
                verify_graph = nx.relabel_nodes(verify_graph, mapping=mapping)
                instance = create_instance(verify_graph)

                reqNod[ind] = instance['required_edges']
                vehCap[ind] = instance['vehicle_capacity']
                deNo[ind] = instance['depots']
                plot_graph(verify_graph, instance)


                print(f"Processed {filename} successfully.")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
            ind += 1
    
    print(deNo)
    print(reqNod)
    print(vehCap)
    np.save(output_info_folder + '/requiredNodes.npy', np.array(reqNod, dtype=object))
    np.save(output_info_folder + '/vehicleCapacity.npy', np.array(vehCap, dtype=object))
    np.save(output_info_folder + '/depotNodes.npy', np.array(deNo, dtype=object))   

# Specify the input and output folders
input_folder = "C:/Users/eashw/Desktop/Multi-Trip-Algorithms/dataset/kshs/"  
output_folder = 'C:/Users/eashw/Desktop/Multi-Trip-Algorithms/dataset/kshs/graph_files'  # Replace with the path where you want to save .net files
output_info_folder = 'C:/Users/eashw/Desktop/Multi-Trip-Algorithms/dataset/kshs/graph info'
# Process all .dat files
process_dat_files(input_folder, output_folder, output_info_folder)





