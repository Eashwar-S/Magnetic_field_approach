import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
from math import exp, sqrt
import time
import os
import pandas as pd
import copy
import re
from centralized_auction_new import *
from magnetic_field import MagneticFieldRouter, IntelligentCapacityTuner
plt.rcParams["figure.dpi"] = 300


class ModifiedMagneticFieldRouter:
    def __init__(self, graph, start_depot, depot_nodes, capacity, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.depot_nodes = depot_nodes
        self.capacity = capacity
        self.alpha = alpha
        self.gamma = gamma
        self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
    def _create_layout(self):
        return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
    def calculate_distances(self):
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
    def calculate_required_edge_influence(self, required_edges_to_cover):
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            influences[edge] = {}
            influences[edge[::-1]] = {}
            
            for i, req_edge in enumerate(required_edges_to_cover):
                u_req, v_req = req_edge
                u_edge, v_edge = edge
                
                d1 = min(distances[u_edge].get(u_req, float('inf')), 
                        distances[u_edge].get(v_req, float('inf')))
                d2 = min(distances[v_edge].get(u_req, float('inf')), 
                        distances[v_edge].get(v_req, float('inf')))
                
                if d1 != float('inf') and d2 != float('inf'):
                    influence = 0.5 * (exp(-self.alpha * d1) + exp(-self.alpha * d2))
                else:
                    influence = 0.0
                
                influences[edge][f'req_{i}'] = influence
                influences[edge[::-1]][f'req_{i}'] = influence
        
        return influences
    
    def calculate_depot_node_influence_factor(self, depot_node, non_traversed_edges):
        """
        Calculate influence factor for a specific depot node based on non-traversed required edges
        """
        if not non_traversed_edges:
            return 0.0
            
        distances = self.calculate_distances()
        total_influence = 0.0
        
        for req_edge in non_traversed_edges:
            u_req, v_req = req_edge
            
            # Calculate distance from depot node to the required edge endpoints
            d1 = distances[depot_node].get(u_req, float('inf'))
            d2 = distances[depot_node].get(v_req, float('inf'))
            
            if d1 != float('inf') and d2 != float('inf'):
                # Use minimum distance to the required edge
                min_distance = min(d1, d2)
                influence = exp(-self.alpha * min_distance)
                total_influence += influence
        
        # Normalize by number of non-traversed edges to keep values reasonable
        return total_influence / max(1, len(non_traversed_edges))
    
    def calculate_required_edge_influence_enhanced(self, required_edges_to_cover):
        """
        Enhanced influence calculation using depot-specific factors
        """
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            influences[edge] = {}
            influences[edge[::-1]] = {}
            
            for i, req_edge in enumerate(required_edges_to_cover):
                u_req, v_req = req_edge
                u_edge, v_edge = edge
                
                d1 = min(distances[u_edge].get(u_req, float('inf')), 
                        distances[u_edge].get(v_req, float('inf')))
                d2 = min(distances[v_edge].get(u_req, float('inf')), 
                        distances[v_edge].get(v_req, float('inf')))
                
                if d1 != float('inf') and d2 != float('inf'):
                    # Calculate depot influence factors a and b
                    a = self.calculate_depot_node_influence_factor(u_edge, required_edges_to_cover)
                    b = self.calculate_depot_node_influence_factor(v_edge, required_edges_to_cover)
                    
                    # Apply your proposed formula
                    influence = (1 + a) * exp(-self.alpha * d1) + (1 + b) * exp(-self.alpha * d2)
                else:
                    influence = 0.0
                
                influences[edge][f'req_{i}'] = influence
                influences[edge[::-1]][f'req_{i}'] = influence
        
        return influences
    
    def select_optimal_end_depot(self, current_node, non_traversed_edges):
        """
        Select end depot based on enhanced influence calculation - no arbitrary thresholds
        """
        # best_depot = None
        # best_influence = -1.0
        
        # for depot in self.depot_nodes:
        #     # Check if we can reach this depot
        #     try:
        #         path_length = nx.shortest_path_length(self.graph, current_node, depot, weight='weight')
        #         if path_length <= self.capacity:  # Feasible to reach
        #             influence = self.calculate_depot_influence_score(depot, non_traversed_edges)
        #             if influence > best_influence:
        #                 best_influence = influence
        #                 best_depot = depot
        #     except nx.NetworkXNoPath:
        #         continue
        influences = self.calculate_depot_influence(non_traversed_edges)
        # find depot with highest influence which is reachable from current_node within capacity
        reachable_depots = []
        for depot in self.depot_nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, current_node, depot, weight='weight')
                if path_length <= self.capacity:
                    reachable_depots.append((depot, influences.get(depot, 0.0)))
            except nx.NetworkXNoPath:
                continue
        
        # return depot with highest influence in reachable depots    
        return max(reachable_depots, key=lambda x: x[1])[0]
        # return best_depot if best_depot is not None else self.depot_nodes[0]
    
    def calculate_depot_influence(self, non_traversed_edges):
        """
        Calculate depot influence using original formula but with enhanced distance calculation
        """
        distances = self.calculate_distances()
        influences = {}
        
        for depot in self.depot_nodes:
            total_influence = 0.0
            for neighbor in self.graph.neighbors(depot):
                depot_edge = (depot, neighbor)
                
                for req_edge in non_traversed_edges:
                    u_req, v_req = req_edge
                    
                    d1 = min(distances[depot].get(u_req, float('inf')), 
                            distances[depot].get(v_req, float('inf')))
                    d2 = min(distances[neighbor].get(u_req, float('inf')), 
                            distances[neighbor].get(v_req, float('inf')))
                    
                    if d1 != float('inf') and d2 != float('inf'):
                        # Calculate depot influence factors a and b
                        a = self.calculate_depot_node_influence_factor(depot, non_traversed_edges)
                        b = self.calculate_depot_node_influence_factor(neighbor, non_traversed_edges)
                        
                        # Apply enhanced formula
                        influence = (1 + a) * exp(-self.alpha * d1) + (1 + b) * exp(-self.alpha * d2)
                        total_influence += influence
            
            influences[depot] = total_influence
        
        return influences
    
    def calculate_edge_score(self, edge, required_to_cover, current_length, end_depot, is_new_required=False, total_required_edges=0):
        # Use enhanced influence calculation for P
        req_influences = self.calculate_required_edge_influence_enhanced(required_to_cover)
        
        # Calculate depot influence D using original formula
        distances = self.calculate_distances()
        u_edge, v_edge = edge
        d_u = min(distances[u_edge].get(self.start_depot, float('inf')),
                 distances[u_edge].get(end_depot, float('inf')))
        d_v = min(distances[v_edge].get(self.start_depot, float('inf')),
                 distances[v_edge].get(end_depot, float('inf')))
        
        if d_u != float('inf') and d_v != float('inf'):
            D = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
                      exp(-self.gamma * d_v / self.capacity))
        else:
            D = 0.1
        
        P = sum(req_influences[edge].values()) if req_influences[edge] else 0.0
        w = current_length / self.capacity if self.capacity > 0 else 0
        
        # Keep original S formula
        S = (1 - w) * P + w * D
        
        if is_new_required:
            final_score = 1000/total_required_edges + S
        else:
            final_score = S + (2*self.capacity)/(self.capacity - current_length + 1)
            
        return {
            'P': P,
            'D': D,
            'w': w,
            'S': S,
            'final_score': final_score,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
        }
    
    def find_trip_with_fixed_end_depot(self, required_edges, verbose=False):
        """
        Find trip with end depot determined upfront and fixed for the entire trip
        """
        # Determine optimal end depot at the beginning based on all non-traversed edges
        non_traversed_edges = required_edges.copy()
        end_depot = self.select_optimal_end_depot(self.start_depot, non_traversed_edges)
        
        if verbose:
            print(f"Fixed end depot for trip: {end_depot}")
        
        current_route = [self.start_depot]
        current_length = 0
        required_covered = set()
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0
        total_required_edges = len(required_edges)

        while len(required_covered) < total_required_edges and iteration_count < max_iterations:
            current_node = current_route[-1]
            candidates = []
            iteration_count += 1
            
            # Update required edges to cover
            required_to_cover = [req for req in required_edges if tuple(sorted(req)) not in required_covered]
            
            # Get all possible next edges
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                edge_weight = self.graph[current_node][neighbor]['weight']
                
                # Capacity check with fixed end depot
                try:
                    if neighbor == end_depot:
                        path_to_end_length = 0
                    else:
                        path_to_end_length = nx.shortest_path_length(
                            self.graph, neighbor, end_depot, weight='weight'
                        )
                    
                    if current_length + edge_weight + path_to_end_length > self.capacity:
                        continue
                except nx.NetworkXNoPath:
                    continue
                
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
                                and edge_sorted not in required_covered)
                
                # Pass fixed end_depot to scoring function
                score_data = self.calculate_edge_score(edge, required_to_cover, current_length, 
                                                     end_depot, is_new_required, total_required_edges)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
            
            # Handle no candidates
            if not candidates:
                if verbose:
                    print(f"No direct candidates at node {current_node} - seeking uncovered required edges...")
                
                if required_to_cover and current_node != end_depot:
                    best_path = None
                    best_length = float('inf')
                    target_edge = None
                    
                    for req_edge in required_to_cover:
                        u, v = req_edge
                        for target_node in [u, v]:
                            try:
                                path = nx.shortest_path(self.graph, current_node, target_node, weight='weight')
                                path_length = nx.shortest_path_length(self.graph, current_node, target_node, weight='weight')
                                
                                try:
                                    final_path_length = nx.shortest_path_length(self.graph, target_node, end_depot, weight='weight')
                                    total_length = current_length + path_length + final_path_length
                                    
                                    if total_length <= self.capacity and path_length < best_length:
                                        best_path = path[1:]
                                        best_length = path_length
                                        target_edge = req_edge
                                except nx.NetworkXNoPath:
                                    continue
                            except nx.NetworkXNoPath:
                                continue
                    
                    if best_path:
                        if verbose:
                            print(f"Forcing path {best_path} to reach required edge {target_edge}")
                        for next_node in best_path:
                            edge_weight = self.graph[current_route[-1]][next_node]['weight']
                            current_route.append(next_node)
                            current_length += edge_weight
                        continue
                
                # Go to fixed end depot if no further progress possible
                if current_node != end_depot:
                    try:
                        path_to_end = nx.shortest_path(self.graph, current_node, end_depot, weight='weight')
                        additional_length = nx.shortest_path_length(self.graph, current_node, end_depot, weight='weight')
                        if current_length + additional_length <= self.capacity:
                            current_route.extend(path_to_end[1:])
                            current_length += additional_length
                            if verbose:
                                print(f"Final path to fixed end depot: {path_to_end[1:]}")
                    except nx.NetworkXNoPath:
                        pass
                break
            
            # Sort candidates: prioritize new required edges
            candidates.sort(key=lambda x: (
                not x['is_new_required'],
                -x['score_data']['final_score']
            ))
            
            best = candidates[0]
            current_route.append(best['neighbor'])
            current_length += best['score_data']['edge_weight']
            
            if best['is_new_required']:
                required_covered.add(tuple(sorted(best['edge'])))
                if verbose:
                    print(f"Covered required edge: {best['edge']} ({len(required_covered)}/{len(required_edges)})")
            
            if verbose:
                print(f"Step: {best['edge']} -> Node {best['neighbor']}, "
                    f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
        # Ensure end at fixed depot
        if current_route[-1] != end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], end_depot, weight='weight')
                current_route.extend(path_to_end[1:])
                current_length += additional_length
            except nx.NetworkXNoPath:
                return None, float('inf'), len(required_covered)
        
        if verbose:
            print(f"Final route: {current_route}, Length: {current_length:.2f}")
            print(f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
        return current_route, current_length, len(required_covered)


def extract_vehicle_future_required_edges(vehicle_routes, vehicle_index, current_trip_index, required_edges):
    """
    Extract all required edges that will be traversed by vehicle in future trips
    """
    future_required_edges = []
    
    # Get all future trips for this vehicle (starting from current_trip_index)
    future_trips = vehicle_routes[vehicle_index][current_trip_index:]
    
    for trip in future_trips:
        for i in range(len(trip) - 1):
            edge = [trip[i], trip[i + 1]]
            if edge in required_edges or edge[::-1] in required_edges:
                # Normalize edge representation
                normalized_edge = tuple(sorted(edge))
                if normalized_edge not in [tuple(sorted(e)) for e in future_required_edges]:
                    future_required_edges.append(edge)
    
    return future_required_edges



def design_multiple_trips_with_magnetic_field(G, start_depot, depot_nodes, required_edges, 
                                             vehicle_capacity, recharge_time, verbose=False):
    """
    Design multiple trips to cover all required edges using magnetic field approach
    """
    trips = []
    trip_times = []
    remaining_required_edges = required_edges.copy()
    current_depot = start_depot
    
    while remaining_required_edges:
        if verbose:
            print(f"Designing trip starting from depot {current_depot}")
            print(f"Remaining required edges: {remaining_required_edges}")
        
        # Use modified magnetic field router for this trip
        router = ModifiedMagneticFieldRouter(
            G, current_depot, depot_nodes, vehicle_capacity, alpha=1.0, gamma=1.0
        )
        
        # Find trip covering as many remaining required edges as possible
        trip_route, trip_cost, edges_covered_count = router.find_trip_with_fixed_end_depot(
            remaining_required_edges, verbose=verbose
        )
        
        if trip_route is None or edges_covered_count == 0:
            if verbose:
                print("Cannot design feasible trip for remaining required edges")
            break
        
        # Track which edges were actually covered in this trip
        covered_edges_in_trip = []
        for i in range(len(trip_route) - 1):
            edge = [trip_route[i], trip_route[i + 1]]
            normalized_edge = tuple(sorted(edge))
            
            # Check if this edge is in remaining required edges
            for req_edge in remaining_required_edges:
                if tuple(sorted(req_edge)) == normalized_edge:
                    covered_edges_in_trip.append(req_edge)
        
        # Remove covered edges from remaining list
        for covered_edge in covered_edges_in_trip:
            if covered_edge in remaining_required_edges:
                remaining_required_edges.remove(covered_edge)
            # Also check reverse
            reverse_edge = [covered_edge[1], covered_edge[0]]
            if reverse_edge in remaining_required_edges:
                remaining_required_edges.remove(reverse_edge)
        
        trips.append(trip_route)
        trip_times.append(trip_cost)
        current_depot = trip_route[-1]  # Next trip starts from where this one ended
        
        if verbose:
            print(f"Trip designed: {trip_route}")
            print(f"Trip cost: {trip_cost:.2f}")
            print(f"Covered edges: {covered_edges_in_trip}")
            print(f"Remaining edges: {len(remaining_required_edges)}")
        
        # Safety check to prevent infinite loops
        if len(trips) > 20:  # Maximum reasonable number of trips
            if verbose:
                print("Maximum trip limit reached")
            break
    
    # Calculate total cost including recharge times
    total_cost = sum(trip_times) + (len(trip_times) - 1) * recharge_time
    
    success = len(remaining_required_edges) == 0
    
    if verbose:
        print(f"Trip design {'successful' if success else 'failed'}")
        print(f"Total trips: {len(trips)}")
        print(f"Total cost: {total_cost:.2f}")
        print(f"Uncovered edges: {remaining_required_edges}")
    
    return trips, trip_times, total_cost, success


def optimize_vehicle_routes_post_auction(G, vehicle_routes, vehicle_trip_times, depot_nodes, 
                                       required_edges, vehicle_capacity, recharge_time, 
                                       vehicle_trip_index, failure_history, verbose=False):
    """
    Optimize vehicle routes after centralized auction using magnetic field approach
    Takes into account current vehicle positions and cannot modify completed trips
    """
    improved_routes = copy.deepcopy(vehicle_routes)
    improved_trip_times = copy.deepcopy(vehicle_trip_times)
    optimization_applied = False
    total_improvement = 0.0
    
    for vehicle_idx in range(len(vehicle_routes)):
        # Skip failed vehicles and empty vehicles
        if vehicle_idx in failure_history or not vehicle_routes[vehicle_idx]:
            continue
        
        # Get current trip index for this vehicle
        current_trip_idx = vehicle_trip_index[vehicle_idx] + 1  # Next trip to be executed
        
        if current_trip_idx >= len(vehicle_routes[vehicle_idx]):
            if verbose:
                print(f"Vehicle {vehicle_idx + 1} has completed all trips")
            continue
            
        if verbose:
            print(f"\nOptimizing Vehicle {vehicle_idx + 1} from trip index {current_trip_idx}")
        
        # Get future required edges for this vehicle
        future_required_edges = extract_vehicle_future_required_edges(
            vehicle_routes, vehicle_idx, current_trip_idx, required_edges
        )
        
        if not future_required_edges:
            if verbose:
                print("No future required edges to optimize")
            continue
        
        if verbose:
            print(f"Future required edges: {future_required_edges}")
        
        # Determine starting depot for optimization
        if current_trip_idx == 0:
            start_depot = vehicle_routes[vehicle_idx][0][0]  # Vehicle's original starting depot
        else:
            # Start from the end of the last completed trip
            last_completed_trip = vehicle_routes[vehicle_idx][current_trip_idx - 1]
            start_depot = last_completed_trip[-1]
        
        if verbose:
            print(f"Starting depot for optimization: {start_depot}")
        
        # Design multiple trips using magnetic field approach
        optimized_trips, optimized_times, optimized_total_cost, success = design_multiple_trips_with_magnetic_field(
            G, start_depot, depot_nodes, future_required_edges, vehicle_capacity, recharge_time, verbose=verbose
        )
        
        if not success:
            if verbose:
                print("Magnetic field optimization failed to cover all required edges")
            continue
        
        # Calculate original cost for future trips
        original_future_trips = vehicle_routes[vehicle_idx][current_trip_idx:]
        original_future_times = vehicle_trip_times[vehicle_idx][current_trip_idx:]
        original_cost = sum(original_future_times) + (len(original_future_times) - 1) * recharge_time
        
        if optimized_total_cost < original_cost:
            # Apply optimization - replace future trips with optimized ones
            improved_routes[vehicle_idx] = (vehicle_routes[vehicle_idx][:current_trip_idx] + 
                                          optimized_trips)
            improved_trip_times[vehicle_idx] = (vehicle_trip_times[vehicle_idx][:current_trip_idx] + 
                                              optimized_times)
            
            improvement = original_cost - optimized_total_cost
            total_improvement += improvement
            optimization_applied = True
            
            if verbose:
                print(f"Optimization applied!")
                print(f"Original future cost: {original_cost:.2f}")
                print(f"Optimized cost: {optimized_total_cost:.2f}")
                print(f"Improvement: {improvement:.2f}")
                print(f"Original future trips: {len(original_future_trips)}")
                print(f"Optimized trips: {len(optimized_trips)}")
        else:
            if verbose:
                print(f"No improvement found (Original: {original_cost:.2f}, Optimized: {optimized_total_cost:.2f})")
    
    if verbose and optimization_applied:
        print(f"\nTotal improvement across all vehicles: {total_improvement:.2f}")
    
    return improved_routes, improved_trip_times, optimization_applied


def simulate_1_with_post_optimization(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": [],
        "Post-optimization applied": [], "Final Maximum Trip Time": [], "Total improvement": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/gdb_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/GDB_Failure_Scenarios_Results/results/gdb_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]
        
        if int(scenario_num) == 1:  # Process only scenario 1 for testing
            print(f'Running GDB scenario {scenario_num}')
            
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_route_times(G, vehicle_routes)
            
            # Calculate initial mission time
            mission_time = max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time 
                               for k in range(len(vehicle_routes)) if vehicle_trip_times[k]])
            
            uavLocation = []
            for k in range(len(vehicle_routes)):
                if vehicle_routes[k]:
                    uavLocation.append(vehicle_routes[k][0][0])
                else:
                    uavLocation.append(depot_nodes[k % len(depot_nodes)])
            
            previous_mission_time = mission_time
            instanceData['Instance Name'].append(f"gdb.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
            # Run centralized auction (original algorithm)
            t = 0
            vehicle_status = [True] * len(vehicle_routes)
            failure_history = []
            detected_failed_vehicles = {}
            vehicle_failure_detection_time = -np.inf
            idle_time = [{} for _ in range(num_vehicles)]
            for k in range(len(vehicle_routes)):
                for trip in range(len(vehicle_routes[k])):
                    idle_time[k][trip] = 0
            
            recent_failure_vehicle = -1
            num_function_calls = 0
            
            start = time.time()
            print("\nStarting Centralized Auction Phase")
            print(f"Initial mission time: {mission_time} time units")
            
            while t <= mission_time:
                for i, k in enumerate(failure_vehicles):
                    if t == vehicle_failure_times[i]:
                        vehicle_status[k] = False
                        recent_failure_vehicle = k
                        vehicle_failure_detection_time = t
                        
                for k, status in enumerate(vehicle_status):
                    if status == False:
                        detected_failed_vehicles[k] = vehicle_failure_detection_time
                        
                if detected_failed_vehicles and t == vehicle_failure_detection_time:
                    for k in list(detected_failed_vehicles.keys()):
                        vehicle_status[k] = None
                    failure_history += list(detected_failed_vehicles.keys())
                    
                    vehicle_trip_index, idle_time = vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times,
                                                                     recharge_time, failure_history, idle_time,
                                                                     recent_failure_vehicle)
                    
                    failure_trips = identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, 
                                                             recent_failure_vehicle, required_edges)
                    
                    vehicle_routes[recent_failure_vehicle] = vehicle_routes[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    vehicle_trip_times[recent_failure_vehicle] = vehicle_trip_times[recent_failure_vehicle][:vehicle_trip_index[recent_failure_vehicle]]
                    
                    # Run centralized auction
                    vehicle_routes, vehicle_trip_times = centralized_auction_optimized(G, t, failure_history, vehicle_routes, 
                                                                                         vehicle_trip_times, vehicle_trip_index, 
                                                                                         depot_nodes, failure_trips, vehicle_capacity, 
                                                                                         recharge_time, uavLocation)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    print(f"\n--- Post-Failure Optimization at t={t} ---")
                    pre_opt_mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                                     sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                    # Apply post-auction optimization immediately after each failure
                    optimized_routes, optimized_trip_times, optimization_applied = optimize_vehicle_routes_post_auction(
                        G, vehicle_routes, vehicle_trip_times, depot_nodes, required_edges, 
                        vehicle_capacity, recharge_time, vehicle_trip_index, failure_history, verbose=True
                    )
                    
                    if optimization_applied:
                        vehicle_routes = optimized_routes
                        vehicle_trip_times = optimized_trip_times
                        post_opt_mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                                          sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                        improvement = pre_opt_mission_time - post_opt_mission_time
                        print(f"Optimization applied: Mission time reduced from {pre_opt_mission_time} to {post_opt_mission_time} (improvement: {improvement:.2f})")
                    else:
                        print("No optimization improvement found")
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            print("Centralized Auction with Real-time Optimization Complete")
            print(f"Final mission time: {mission_time}")
            
            # Final mission time is already optimized through real-time optimization
            final_mission_time = mission_time
            
            total_improvement = previous_mission_time - final_mission_time
            
            print(f"\nReal-time Optimization Complete")
            print(f"Total improvement: {total_improvement:.2f}")
            
            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)
            instanceData['Post-optimization applied'].append(True)  # Always true since we optimize after each failure
            instanceData['Final Maximum Trip Time'].append(final_mission_time)
            instanceData['Total improvement'].append(total_improvement)
            
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(f"Original mission time: {previous_mission_time:.2f}")
            print(f"After auction: {mission_time:.2f}")
            print(f"After optimization: {final_mission_time:.2f}")
            print(f"Total improvement: {total_improvement:.2f}")
            print("="*60)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"{p}/results/instances_results_with_failure/centralized_auction_with_post_optimization_gdb.csv", index=True)


if __name__ == "__main__":
    simulate_1_with_post_optimization(save_results_to_csv=False)