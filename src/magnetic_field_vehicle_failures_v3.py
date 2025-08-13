from tracemalloc import start
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import time
import os
import pandas as pd
import copy
import tqdm
import re
from magnetic_field import MagneticFieldRouter, IntelligentCapacityTuner
plt.rcParams["figure.dpi"] = 300

def calculate_depot_distances(G, depots, C, recharge_time):
    depot_to_depot_distances = {}
    for i, depot1 in enumerate(depots):
        for depot2 in depots[i:]:
            try:
                if depot1 == depot2:
                    depot_to_depot_distances[(depot1, depot2)] = [0, []]
                    depot_to_depot_distances[(depot2, depot1)] = [0, []]
                else:
                    distance, path = nx.single_source_dijkstra(G, depot1, depot2, weight='weight', cutoff=C)
                    distance = round(distance, 1)
                    depot_to_depot_distances[(depot1, depot2)] = [distance, [path]]
                    depot_to_depot_distances[(depot2, depot1)] = [distance, [path[::-1]]]
            except nx.NetworkXNoPath:
                pass
    for k in depots:
        for i in depots:
            for j in depots:
                if (i, k) in depot_to_depot_distances and (k, j) in depot_to_depot_distances:
                    dist_ik, path_ik = depot_to_depot_distances[(i, k)]
                    dist_kj, path_kj = depot_to_depot_distances[(k, j)]
                    total_dist = round(dist_ik + dist_kj + recharge_time, 1)
                    if (i, j) not in depot_to_depot_distances or total_dist < depot_to_depot_distances[(i, j)][0]:
                        depot_to_depot_distances[(i, j)] = [total_dist, path_ik + path_kj]
    return depot_to_depot_distances


def parse_txt_file(file_path):
    """Parse the .txt failure scenario file to extract graph and parameters"""
    G = nx.Graph()
    required_edges = []
    depot_nodes = []
    vehicle_capacity = None
    recharge_time = None
    num_vehicles = None
    failure_vehicles = []
    vehicle_failure_times = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('NUMBER OF VERTICES'):
            num_vertices = int(line.split(':')[1].strip())
            G.add_nodes_from(range(1, num_vertices + 1))
        elif line.startswith('VEHICLE CAPACITY'):
            vehicle_capacity = float(line.split(':')[1].strip())
        elif line.startswith('RECHARGE TIME'):
            recharge_time = float(line.split(':')[1].strip())
        elif line.startswith('NUMBER OF VEHICLES'):
            num_vehicles = int(line.split(':')[1].strip())
        elif line.startswith('DEPOT:'):
            depot_line = line.split(':', 1)[1].strip()
            depot_nodes = [int(x.strip()) for x in depot_line.split(',')]
        elif line.startswith('LIST_REQUIRED_EDGES:'):
            current_section = 'required_edges'
        elif line.startswith('LIST_NON_REQUIRED_EDGES:'):
            current_section = 'non_required_edges'
        elif line.startswith('FAILURE_SCENARIO:'):
            current_section = 'failure_scenario'
        elif line.startswith('(') and current_section in ['required_edges', 'non_required_edges']:
            # Parse edge: (u,v) edge weight w
            parts = line.split(') ')
            if len(parts) >= 2:
                edge_part = parts[0].strip('(')
                u, v = map(int, edge_part.split(','))
                weight_part = parts[1].strip()
                weight = float(weight_part.split()[-1])
                G.add_edge(u, v, weight=weight)
                if current_section == 'required_edges':
                    required_edges.append([u, v])
        elif current_section == 'failure_scenario' and line.startswith('Vehicle'):
            # Parse: Vehicle X will fail in Y time units.
            match = re.search(r'Vehicle (\d+) will fail in (\d+) time units', line)
            if match:
                vehicle_id = int(match.group(1)) - 1  # Convert to 0-indexed
                failure_time = float(match.group(2))
                failure_vehicles.append(vehicle_id)
                vehicle_failure_times.append(failure_time)
    
    return G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times

def get_vehicle_end_locations(vehicle_routes, uav_location):
    """Get the end location of each vehicle's last trip"""
    end_locations = []
    for i, vehicle_route in enumerate(vehicle_routes):
        if vehicle_route and len(vehicle_route) > 0:
            # Get the last node of the last trip
            last_trip = vehicle_route[-1]
            end_locations.append(last_trip[-1])
        else:
            end_locations.append(uav_location[i])  # If no trips, use UAV location
    return end_locations


def calculate_vehicle_route_times(vehicle_routes, vehicle_trip_times, recharge_time):
    """Calculate total route time for each vehicle"""
    route_times = []
    for k, trips in enumerate(vehicle_routes):
        if vehicle_trip_times[k] and len(vehicle_trip_times[k]) > 0:
            total_time = sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time
            route_times.append(total_time)
        else:
            route_times.append(0.0)
    return route_times

def find_reachable_depots(G, start_depot, depot_nodes, vehicle_capacity, depot_distances):
    """Find all depots reachable from start_depot within vehicle capacity"""
    reachable_depots = []
    
    for end_depot in depot_nodes:
        if start_depot != end_depot:    
            # Check if depot is reachable within capacity
            if (start_depot, end_depot) in depot_distances:
                distance, _ = depot_distances[(start_depot, end_depot)]
                if distance <= vehicle_capacity:
                    reachable_depots.append(end_depot)

    return reachable_depots

def count_required_edges_coverage(trip, required_edges):
    """Count how many required edges are covered by a trip"""
    if not trip or len(trip) < 2:
        return 0
    
    covered_count = 0
    for i in range(len(trip) - 1):
        edge = [trip[i], trip[i + 1]]
        edge_reversed = [trip[i + 1], trip[i]]
        
        for req_edge in required_edges:
            if edge == list(req_edge) or edge_reversed == list(req_edge):
                covered_count += 1
                break
    return covered_count


def calculate_trip_times_in_route(G, routes):
    """Calculate trip times for each vehicle's routes"""
    vehicle_trip_times = []
    for vehicle_routes in routes:
        trip_times = []
        for trip in vehicle_routes:
            trip_time = 0
            for i in range(len(trip) - 1):
                if G.has_edge(trip[i], trip[i + 1]):
                    trip_time += G[trip[i]][trip[i + 1]]['weight']
                else:
                    print(f"Warning: Edge ({trip[i]}, {trip[i + 1]}) not found in graph")
            trip_times.append(trip_time)
        vehicle_trip_times.append(trip_times)
    return vehicle_trip_times

def vehicle_at_time_t(t, vehicle_routes, vehicle_trip_times, recharge_time, failure_history,
                      idle_time, recent_failure_vehicle):
    vehicle_trip_index = [-1] * len(vehicle_routes)

    for k, routes in enumerate(vehicle_routes):
        sum_distance = 0
        if k not in failure_history or k == recent_failure_vehicle:
            if routes:
                for trip_index, trip in enumerate(routes):
                    if trip_index == len(routes) - 1:
                        sum_distance += vehicle_trip_times[k][trip_index]
                        vehicle_trip_index[k] = trip_index
                        ready_time = sum_distance + recharge_time  # Vehicle is ready after recharging
                        if ready_time < t:  # Only idle if current time exceeds ready time
                            # if trip_index in idle_time[k]:
                            idle_time[k][trip_index] = round(t - ready_time, 1)
                            # else:
                            #     idle_time[k][trip_index] = round(t - ready_time, 1)
                        # if sum_distance < t:
                        #     if trip_index in idle_time[k]:
                        #         idle_time[k][trip_index] += round(t - sum_distance, 1)
                        #     else:
                        #         idle_time[k][trip_index] = round(t - sum_distance, 1)
                    else:
                        if trip_index not in idle_time[k]:
                            idle_time[k][trip_index] = 0
                        sum_distance += vehicle_trip_times[k][trip_index] + idle_time[k][trip_index] + recharge_time
                        if sum_distance >= t:
                            vehicle_trip_index[k] = trip_index
                            break
            else:
                idle_time[k][0] = t
                vehicle_trip_index[k] = 0
        sum_distance = round(sum_distance, 1)
        # print(f'vehicle {k+1} - vehicle trip index - {vehicle_trip_index[k]}, sum_distance - {sum_distance}')
    return vehicle_trip_index, idle_time


def identifying_failed_trips(vehicle_routes, vehicle_trip_times, vehicle_trip_index, recent_failure_vehicle, required_edges):
    failed_trips = {}
    seen = set()  # canonicalized trips we've already kept

    for trip_index, trip in enumerate(vehicle_routes[recent_failure_vehicle]):
        if trip_index >= vehicle_trip_index[recent_failure_vehicle]:
            # Does this trip cover at least one required edge?
            touches_required = any(
                ([trip[i], trip[i + 1]] in required_edges) or ([trip[i + 1], trip[i]] in required_edges)
                for i in range(len(trip) - 1)
            )
            if not touches_required:
                continue

            # Canonicalize trip to catch duplicates and reverse-duplicates
            t = tuple(trip)
            canon = t if t <= t[::-1] else t[::-1]

            if canon in seen:
                continue  # skip duplicates
            seen.add(canon)

            key = f"{vehicle_trip_times[recent_failure_vehicle][trip_index]}_{trip_index}_{recent_failure_vehicle}"
            failed_trips[key] = trip

    return failed_trips


def handle_vehicle_failure_with_magnetic_field(G, t, failure_history, vehicle_routes, vehicle_trip_times, 
                                              vehicle_trip_index, depot_nodes, failure_trips, vehicle_capacity, 
                                              recharge_time, uav_location, recent_failure_vehicle):
    """
    Handle vehicle failure using magnetic field approach to reassign failed trips to non-failed vehicles.
    
    Args:
        G: Graph
        t: Current time
        failure_history: List of failed vehicle indices
        vehicle_routes: Current vehicle routes
        vehicle_trip_times: Current vehicle trip times
        vehicle_trip_index: Current trip index for each vehicle
        depot_nodes: List of depot nodes
        failure_trips: Dictionary of failed trips
        vehicle_capacity: Vehicle capacity
        recharge_time: Recharge time
        uav_location: Current UAV locations
        recent_failure_vehicle: Index of recently failed vehicle
    
    Returns:
        Updated vehicle_routes, vehicle_trip_times, and metrics
    """
    start_time = time.time()
    
    print(f"Starting magnetic field failure handling at time {t}")
    print(f"Failed vehicle: {recent_failure_vehicle}")
    print(f"Failure trips: {list(failure_trips.keys())}")
    
    # Extract required edges from failed trips
    required_edges = []
    for trip in failure_trips.values():
        for i in range(len(trip) - 1):
            edge = [trip[i], trip[i + 1]]
            if edge not in required_edges and edge[::-1] not in required_edges:
                required_edges.append(edge)

    # Add required edges from future trips of non-failed vehicles
    non_failed_vehicles = [i for i in range(len(vehicle_routes)) if i not in failure_history]
    
    for vehicle_idx in non_failed_vehicles:
        if vehicle_trip_index[vehicle_idx] < len(vehicle_routes[vehicle_idx]):
            # Add required edges from future trips (after current trip)
            future_trips = vehicle_routes[vehicle_idx][vehicle_trip_index[vehicle_idx] + 1:]
            for trip in future_trips:
                for i in range(len(trip) - 1):
                    edge = [trip[i], trip[i + 1]]
                    if edge not in required_edges and edge[::-1] not in required_edges:
                        required_edges.append(edge)
            
            # Remove future trips from the vehicle's route since they'll be reassigned
            vehicle_routes[vehicle_idx] = vehicle_routes[vehicle_idx][:vehicle_trip_index[vehicle_idx] + 1]
            vehicle_trip_times[vehicle_idx] = vehicle_trip_times[vehicle_idx][:vehicle_trip_index[vehicle_idx] + 1]

    print(f"Total required edges to reassign: {len(required_edges)}")
    
    if not required_edges:
        return vehicle_routes, vehicle_trip_times, {'execution_time': time.time() - start_time}

    if not non_failed_vehicles:
        return vehicle_routes, vehicle_trip_times, {'execution_time': time.time() - start_time}

    # Calculate depot-to-depot distances
    depot_distances = calculate_depot_distances(G, depot_nodes, vehicle_capacity, recharge_time)
    
    # Get vehicle end locations (where their last trip ended)
    vehicle_end_locations = get_vehicle_end_locations(vehicle_routes, uav_location.copy())
    
    # Calculate current route times for vehicle ordering
    vehicle_route_times = calculate_vehicle_route_times(vehicle_routes, vehicle_trip_times, recharge_time)
    
    # Create list of (vehicle_idx, route_time) for non-failed vehicles and sort by route time (ascending)
    vehicle_usage = [(idx, vehicle_route_times[idx]) for idx in non_failed_vehicles]
    vehicle_usage.sort(key=lambda x: x[1])  # Sort by route time (least used first)
    
    print(f"Vehicle usage order: {vehicle_usage}")
    
    remaining_required_edges = required_edges.copy()
    new_trips_assigned = {vehicle_idx: [] for vehicle_idx in non_failed_vehicles}
    
    while remaining_required_edges:
        # Process vehicles in order of least used first
        for vehicle_idx, current_route_time in vehicle_usage:
            if not remaining_required_edges:
                break
                
            print(f"\nProcessing vehicle {vehicle_idx} (current route time: {current_route_time})")
            
            # Determine start depot (where vehicle's last trip ended)
            start_depot = vehicle_end_locations[vehicle_idx]
            if start_depot is None:
                # Vehicle has no trips yet, use its initial location
                start_depot = uav_location[vehicle_idx] if vehicle_idx < len(uav_location) else depot_nodes[0]
            
            print(f"  Start depot: {start_depot}")
            
            # Find all reachable end depots within capacity
            reachable_depots = find_reachable_depots(G, start_depot, depot_nodes, vehicle_capacity, depot_distances)
            print(f"  Reachable depots: {reachable_depots}")
            
            best_trip = None
            best_end_depot = None
            best_coverage = 0
            best_cost = float('inf')
            
            # Try each reachable end depot
            for end_depot in reachable_depots:
                try:
                    # Create magnetic field router for this start-end depot pair
                    tuner = IntelligentCapacityTuner(G, start_depot, end_depot, vehicle_capacity, required_edges.copy())
                    
                    res = tuner.random_search(50)
                    # router = MagneticFieldRouter(G, start_depot, end_depot, vehicle_capacity, alpha=1.0, gamma=1.0)
                    
                    trip, cost, covered_count = tuner.best_route, tuner.best_cost, tuner.num_required_edges_covered
                    # # Find route using magnetic field approach
                    # trip, cost, covered_count = router.find_trip_with_magnetic_scoring(
                    #     remaining_required_edges.copy(), verbose=False
                    # )
                    
                    if trip is not None:
                        # Count actual required edges covered by this trip
                        actual_coverage = count_required_edges_coverage(trip, remaining_required_edges.copy())
                        
                        # print(f"    End depot {end_depot}: cost={cost:.2f}, coverage={actual_coverage}/{len(remaining_required_edges.copy())}")
                        
                        # Select best trip based on coverage first, then cost
                        if (actual_coverage > best_coverage or 
                            (actual_coverage == best_coverage and cost < best_cost)):
                            best_trip = trip
                            best_end_depot = end_depot
                            best_coverage = actual_coverage
                            best_cost = cost
                            
                except Exception as e:
                    print(f"    Error with end depot {end_depot}: {e}")
                    continue
            
            # Assign best trip to vehicle if found
            if best_trip is not None and best_coverage > 0:
                print(f"  Assigned trip to vehicle {vehicle_idx}: {best_trip}")
                print(f"  Cost: {best_cost:.2f}, Coverage: {best_coverage}")
                
                # Add the new trip to vehicle's trips
                vehicle_routes[vehicle_idx].append(best_trip)
                
                # Calculate trip time
                trip_time = 0
                for i in range(len(best_trip) - 1):
                    try:
                        trip_time += G.get_edge_data(best_trip[i], best_trip[i + 1])[0]["weight"]
                    except:
                        trip_time += G.get_edge_data(best_trip[i], best_trip[i + 1])["weight"]
                
                vehicle_trip_times[vehicle_idx].append(trip_time)
                new_trips_assigned[vehicle_idx].append(best_trip)
                
                # Remove covered required edges from remaining list
                edges_to_remove = []
                for i in range(len(best_trip) - 1):
                    trip_edge = [best_trip[i], best_trip[i + 1]]
                    trip_edge_reversed = [best_trip[i + 1], best_trip[i]]
                    
                    for req_edge in remaining_required_edges:
                        if (trip_edge == req_edge or trip_edge_reversed == req_edge):
                            if req_edge not in edges_to_remove:
                                edges_to_remove.append(req_edge)
                
                for edge in edges_to_remove:
                    if edge in remaining_required_edges:
                        remaining_required_edges.remove(edge)
                    if edge[::-1] in remaining_required_edges:
                        remaining_required_edges.remove(edge[::-1])
                
                # Update vehicle end location
                vehicle_end_locations[vehicle_idx] = best_trip[-1]
                
                print(f"  Remaining required edges: {len(remaining_required_edges)}")
            else:
                print(f"  No suitable trip found for vehicle {vehicle_idx}")
    
    execution_time = time.time() - start_time
    
    print(f"\nMagnetic field failure handling completed in {execution_time:.3f} seconds")
    print(f"Remaining uncovered edges: {len(remaining_required_edges)}")
    
    # Create metrics similar to SA results
    mf_results = {
        'execution_time': round(execution_time, 3),
        'uncovered_edges': len(remaining_required_edges),
        'total_edges': len(required_edges),
        'coverage_percentage': round((len(required_edges) - len(remaining_required_edges)) / len(required_edges) * 100, 1) if required_edges else 100,
        'trips_assigned': sum(len(trips) for trips in new_trips_assigned.values())
    }
    
    return vehicle_routes, vehicle_trip_times, mf_results

def evaluate_all_vehicles_for_best_trip(G, non_failed_vehicles, vehicle_end_locations, 
                                       depot_nodes, depot_distances, vehicle_capacity, 
                                       remaining_required_edges, vehicle_route_times, recharge_time):
    """
    Simplified approach: Get the best trip for each vehicle, then compare all trips
    to select the globally best one.
    
    Returns the best vehicle assignment with trip details.
    """
    vehicle_best_trips = []
    
    # Step 1: Find the best trip for each vehicle independently
    for vehicle_idx in non_failed_vehicles:
        start_depot = vehicle_end_locations[vehicle_idx]
        current_route_time = vehicle_route_times[vehicle_idx]
        
        # Find reachable end depots
        reachable_depots = find_reachable_depots(G, start_depot, depot_nodes, vehicle_capacity, depot_distances)
        
        best_trip_for_vehicle = None
        best_coverage_for_vehicle = 0
        best_cost_for_vehicle = float('inf')
        best_end_depot_for_vehicle = None
        
        # Find best trip among all reachable depots for this vehicle
        for end_depot in reachable_depots:
            try:
                # Evaluate trip using magnetic field approach
                tuner = IntelligentCapacityTuner(G, start_depot, end_depot, vehicle_capacity, 
                                               remaining_required_edges.copy())
                
                tuner.random_search(50)
                
                if tuner.best_capacity is not None:
                    trip, cost, covered_count = tuner.best_route, tuner.best_cost, tuner.num_required_edges_covered
                    
                    if trip is not None:
                        # Calculate actual coverage
                        actual_coverage = count_required_edges_coverage(trip, remaining_required_edges)
                        
                        # Select best trip for this vehicle based on:
                        # 1. Maximum coverage first
                        # 2. Minimum cost second (tie-breaker)
                        if (actual_coverage > best_coverage_for_vehicle or 
                            (actual_coverage == best_coverage_for_vehicle and cost < best_cost_for_vehicle)):
                            
                            best_trip_for_vehicle = trip
                            best_coverage_for_vehicle = actual_coverage
                            best_cost_for_vehicle = cost
                            best_end_depot_for_vehicle = end_depot
                            
            except Exception as e:
                continue
        
        # Store the best trip found for this vehicle
        if best_trip_for_vehicle is not None and best_coverage_for_vehicle > 0:
            new_route_time = current_route_time + best_cost_for_vehicle + recharge_time
            
            vehicle_best_trips.append({
                'vehicle_idx': vehicle_idx,
                'start_depot': start_depot,
                'end_depot': best_end_depot_for_vehicle,
                'trip': best_trip_for_vehicle,
                'cost': best_cost_for_vehicle,
                'coverage': best_coverage_for_vehicle,
                'current_route_time': current_route_time,
                'new_route_time': new_route_time,
                'route_time_increase': best_cost_for_vehicle + recharge_time
            })
    
    # Step 2: Compare all vehicles' best trips and select the globally best one
    if not vehicle_best_trips:
        return None
    
    # Sort by coverage (descending), then by route time increase (ascending)
    vehicle_best_trips.sort(key=lambda x: (-x['coverage'], x['route_time_increase']))
    
    return vehicle_best_trips[0]  # Return the best overall assignment


def evaluate_vehicle_depot_combinations(G, non_failed_vehicles, vehicle_end_locations, 
                                      depot_nodes, depot_distances, vehicle_capacity, 
                                      remaining_required_edges, vehicle_route_times, recharge_time):
    """
    Evaluate all vehicle-depot combinations to find the best coverage potential
    with balanced mission time impact.
    
    Returns list of (vehicle_idx, start_depot, end_depot, coverage_score, mission_time_penalty)
    sorted by a combined heuristic score.
    """
    combinations = []
    
    for vehicle_idx in non_failed_vehicles:
        start_depot = vehicle_end_locations[vehicle_idx]
        current_route_time = vehicle_route_times[vehicle_idx]
        
        # Find reachable end depots
        reachable_depots = find_reachable_depots(G, start_depot, depot_nodes, vehicle_capacity, depot_distances)
        
        for end_depot in reachable_depots:
            try:
                # Quick evaluation using magnetic field approach
                tuner = IntelligentCapacityTuner(G, start_depot, end_depot, vehicle_capacity, 
                                               remaining_required_edges.copy())
                
                # Use a small number of iterations for quick evaluation
                tuner.random_search(50)  # Reduced from 50 for speed
                
                if tuner.best_capacity is not None:
                    trip, cost, covered_count = tuner.best_route, tuner.best_cost, tuner.num_required_edges_covered
                    
                    if trip is not None:
                        # Calculate actual coverage
                        actual_coverage = count_required_edges_coverage(trip, remaining_required_edges)
                        
                        # Calculate mission time penalty (how much this adds to current mission time)
                        new_route_time = current_route_time + cost + recharge_time
                        mission_time_penalty = new_route_time
                        
                        # Combined heuristic score
                        # Prioritize: high coverage, low mission time penalty
                        if actual_coverage > 0:
                            # Coverage efficiency: edges covered per unit time
                            coverage_efficiency = actual_coverage #/ cost if cost > 0 else 0
                            
                            # Normalized mission time penalty (0-1 scale)
                            max_route_time = max(vehicle_route_times) if vehicle_route_times else 1
                            normalized_penalty = mission_time_penalty -  max_route_time if max_route_time > 0 else 0
                            
                            # Combined score: prioritize coverage efficiency, penalize mission time
                            # Higher score is better
                            # combined_score = coverage_efficiency - normalized_penalty
                            combined_score = normalized_penalty - coverage_efficiency *(vehicle_capacity + recharge_time)
                            
                            combinations.append({
                                'vehicle_idx': vehicle_idx,
                                'start_depot': start_depot,
                                'end_depot': end_depot,
                                'trip': trip,
                                'cost': cost,
                                'coverage': actual_coverage,
                                'coverage_efficiency': coverage_efficiency,
                                'mission_time_penalty': mission_time_penalty,
                                'normalized_penalty': normalized_penalty,
                                'combined_score': combined_score
                            })
                            
            except Exception as e:
                continue
    
    # Sort by combined score (descending - higher is better)
    combinations.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return combinations

def group_vehicles_by_depot(non_failed_vehicles, vehicle_end_locations):
    """Group vehicles by their current depot location"""
    depot_groups = defaultdict(list)
    
    for vehicle_idx in non_failed_vehicles:
        depot = vehicle_end_locations[vehicle_idx]
        depot_groups[depot].append(vehicle_idx)
    
    return depot_groups


def handle_vehicle_failure_with_improved_magnetic_field(G, t, failure_history, vehicle_routes, vehicle_trip_times, 
                                                       vehicle_trip_index, depot_nodes, failure_trips, vehicle_capacity, 
                                                       recharge_time, uav_location, recent_failure_vehicle):
    """
    Improved magnetic field approach that balances edge coverage potential with mission time impact.
    
    Key improvements:
    1. Evaluates all vehicle-depot combinations for coverage potential
    2. Uses combined heuristic score (coverage efficiency vs mission time penalty)
    3. Groups vehicles by depot to avoid redundant evaluations
    4. Prioritizes combinations that maximize coverage while minimizing mission time impact
    """
    start_time = time.time()
    
    # print(f"[IMPROVED MF] Starting magnetic field failure handling at time {t}")
    # print(f"[IMPROVED MF] Failed vehicle: {recent_failure_vehicle}")
    # print(f"[IMPROVED MF] Failure trips: {len(failure_trips)}")
    
    # Extract required edges from failed trips
    required_edges = []
    for trip in failure_trips.values():
        for i in range(len(trip) - 1):
            edge = [trip[i], trip[i + 1]]
            if edge not in required_edges and edge[::-1] not in required_edges:
                required_edges.append(edge)

    # Add required edges from future trips of non-failed vehicles
    non_failed_vehicles = [i for i in range(len(vehicle_routes)) if i not in failure_history]
    
    for vehicle_idx in non_failed_vehicles:
        if vehicle_trip_index[vehicle_idx] < len(vehicle_routes[vehicle_idx]):
            future_trips = vehicle_routes[vehicle_idx][vehicle_trip_index[vehicle_idx] + 1:]
            for trip in future_trips:
                for i in range(len(trip) - 1):
                    edge = [trip[i], trip[i + 1]]
                    if edge not in required_edges and edge[::-1] not in required_edges:
                        required_edges.append(edge)
            
            # Remove future trips since they'll be reassigned
            vehicle_routes[vehicle_idx] = vehicle_routes[vehicle_idx][:vehicle_trip_index[vehicle_idx] + 1]
            vehicle_trip_times[vehicle_idx] = vehicle_trip_times[vehicle_idx][:vehicle_trip_index[vehicle_idx] + 1]

    # print(f"[IMPROVED MF] Total required edges to reassign: {len(required_edges)}")
    
    if not required_edges:
        return vehicle_routes, vehicle_trip_times, {'execution_time': time.time() - start_time}

    if not non_failed_vehicles:
        return vehicle_routes, vehicle_trip_times, {'execution_time': time.time() - start_time}

    # Calculate depot distances and vehicle states
    depot_distances = calculate_depot_distances(G, depot_nodes, vehicle_capacity, recharge_time)
    vehicle_end_locations = get_vehicle_end_locations(vehicle_routes, uav_location.copy())
    vehicle_route_times = calculate_vehicle_route_times(vehicle_routes, vehicle_trip_times, recharge_time)
    
    remaining_required_edges = required_edges.copy()
    new_trips_assigned = {vehicle_idx: [] for vehicle_idx in non_failed_vehicles}
    
    # Main assignment loop
    iteration = 0
    while remaining_required_edges:  # Prevent infinite loops
        iteration += 1
        # print(f"\n[IMPROVED MF] Iteration {iteration}: {len(remaining_required_edges)} edges remaining")
        
        # Group vehicles by depot to avoid redundant evaluations
        # depot_groups = group_vehicles_by_depot(non_failed_vehicles, vehicle_end_locations)
        # print(f"[IMPROVED MF] Depot groups: {dict(depot_groups)}")
        
        # Evaluate all vehicle-depot combinations
        # Find the best assignment among all vehicles
        best_assignment = evaluate_all_vehicles_for_best_trip(
            G, non_failed_vehicles, vehicle_end_locations, depot_nodes, 
            depot_distances, vehicle_capacity, remaining_required_edges, 
            vehicle_route_times, recharge_time
        )
        
        if best_assignment is None:
            print("[SIMPLIFIED MF] No valid assignment found")
            break
        
        vehicle_idx = best_assignment['vehicle_idx']
        trip = best_assignment['trip']
        cost = best_assignment['cost']
        coverage = best_assignment['coverage']
        
        # print(f"[SIMPLIFIED MF] Selected vehicle {vehicle_idx}")
        # print(f"[SIMPLIFIED MF]   Trip: {trip}")
        # print(f"[SIMPLIFIED MF]   Coverage: {coverage}, Cost: {cost:.2f}")
        # print(f"[SIMPLIFIED MF]   Route time increase: {best_assignment['route_time_increase']:.2f}")
        
        # Assign trip to selected vehicle
        vehicle_routes[vehicle_idx].append(trip)
        vehicle_trip_times[vehicle_idx].append(cost)
        new_trips_assigned[vehicle_idx].append(trip)
        
        # Remove covered edges
        edges_to_remove = []
        for i in range(len(trip) - 1):
            trip_edge = [trip[i], trip[i + 1]]
            trip_edge_reversed = [trip[i + 1], trip[i]]
            
            for req_edge in remaining_required_edges:
                if (trip_edge == req_edge or trip_edge_reversed == req_edge):
                    if req_edge not in edges_to_remove:
                        edges_to_remove.append(req_edge)
        
        for edge in edges_to_remove:
            if edge in remaining_required_edges:
                remaining_required_edges.remove(edge)
            if edge[::-1] in remaining_required_edges:
                remaining_required_edges.remove(edge[::-1])
        
        # Update vehicle state
        vehicle_end_locations[vehicle_idx] = trip[-1]
        vehicle_route_times[vehicle_idx] += cost + recharge_time
        
        print(f"[SIMPLIFIED MF] Removed {len(edges_to_remove)} edges. Remaining: {len(remaining_required_edges)}")
    
    execution_time = time.time() - start_time
    
    # print(f"\n[IMPROVED MF] Magnetic field failure handling completed in {execution_time:.3f} seconds")
    # print(f"[IMPROVED MF] Remaining uncovered edges: {len(remaining_required_edges)}")
    # print(f"[IMPROVED MF] Total iterations: {iteration}")
    
    # Create metrics
    mf_results = {
        'execution_time': round(execution_time, 3),
        'uncovered_edges': len(remaining_required_edges),
        'total_edges': len(required_edges),
        'coverage_percentage': round((len(required_edges) - len(remaining_required_edges)) / len(required_edges) * 100, 1) if required_edges else 100,
        'trips_assigned': sum(len(trips) for trips in new_trips_assigned.values()),
        'iterations': iteration
    }
    
    return vehicle_routes, vehicle_trip_times, mf_results


# GDB
def simulate_1(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": [],
        # "Mean Execution Time SA": [], "STDEV Execution Time SA": [], "Best Execution Time SA": [],
        "Mean Mission Time SA": [], "STDEV Mission Time SA": [], "Best Mission Time SA": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/gdb_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/GDB_Failure_Scenarios_Results/results/gdb_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from gdb.X.txt
        

        print()
        print("-"*50)
        print(f'Running GDB scenario {scenario_num}')
        if True:#int(scenario_num) == 1:  # Process only scenario 1 for testing
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
            
            # print(f"Failure Vehicles: {failure_vehicles}")
            # print(f"Vehicle Failure Times: {vehicle_failure_times}")
            # print(f"recharge_time: {recharge_time}")
            # print(f"Depot Nodes: {depot_nodes}")

            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_trip_times_in_route(G, vehicle_routes)
            
            print(f"Loaded vehicle routes for scenario {scenario_num}: {vehicle_routes}")
            print(f"Vehicle trip times for scenario {scenario_num}: {vehicle_trip_times}")

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
            
            all_mf_results = []
            start = time.time()
            
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

                    # print(f"Detected failed vehicles at time {t}: {detected_failed_vehicles}")
                    # print(f"Recent failure vehicle: {recent_failure_vehicle}")
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
                    
                    # Simulated annealing function call
                    vehicle_routes, vehicle_trip_times, mf_results = handle_vehicle_failure_with_improved_magnetic_field(
                                                                G, t, failure_history, vehicle_routes, vehicle_trip_times,
                                                                vehicle_trip_index, depot_nodes, failure_trips, vehicle_capacity,
                                                                recharge_time, uavLocation, recent_failure_vehicle
                                                            )
                                                                                
                    print(f"Vehicle routes after handling failure: {vehicle_routes}")
                    print(f"Vehicle trip times after handling failure: {vehicle_trip_times}")
                    

                    # Store SA results if available
                    if mf_results:
                        all_mf_results.append(mf_results)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            print(f"Final Vehicle routes: {vehicle_routes}")
            print(f"Final Vehicle trip times: {vehicle_trip_times}")
            print(f"Total mission time after SA algorithm: {mission_time}")

            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)

            # Aggregate SA metrics across all failure events
            if all_mf_results:
                # all_mean_times = [result['mean_time'] for result in all_mf_results]
                # all_std_times = [result['std_time'] for result in all_mf_results]
                # all_best_times = [result['best_time'] for result in all_mf_results]
                # all_mean_mission_times = [result['mean_mission_time'] for result in all_mf_results]
                # all_std_mission_times = [result['std_mission_time'] for result in all_mf_results]
                # all_best_mission_times = [result['best_mission_time'] for result in all_mf_results]
                
                # instanceData['Mean Execution Time SA'].append(round(np.mean(all_mean_times), 3))
                # instanceData['STDEV Execution Time SA'].append(round(np.mean(all_std_times), 3))
                # instanceData['Best Execution Time SA'].append(round(np.min(all_best_times), 3))
                instanceData['Mean Mission Time SA'].append(round(mission_time, 1))
                instanceData['STDEV Mission Time SA'].append(0)
                instanceData['Best Mission Time SA'].append(round(mission_time, 1))
            else:
                # No SA runs occurred
                instanceData['Mean Execution Time SA'].append(0)
                instanceData['STDEV Execution Time SA'].append(0)
                instanceData['Best Execution Time SA'].append(0)
                instanceData['Mean Mission Time SA'].append(0)
                instanceData['STDEV Mission Time SA'].append(0)
                instanceData['Best Mission Time SA'].append(0)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"magnetic_field_gdb_new.csv", index=True)

# BCCM
def simulate_2(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": [],
        "Mean Execution Time SA": [], "STDEV Execution Time SA": [], "Best Execution Time SA": [],
        "Mean Mission Time SA": [], "STDEV Mission Time SA": [], "Best Mission Time SA": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/bccm_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/BCCM_Failure_Scenarios_Results/results/bccm_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from gdb.X.txt
        
        print()
        print("-"*50)
        print(f'Running BCCM scenario {scenario_num}')
        if True:#int(scenario_num) == 1:  # Process only scenario 1 for testing
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
       
            # # print(f"Failure Vehicles: {failure_vehicles}")
            # # print(f"Vehicle Failure Times: {vehicle_failure_times}")
            # # print(f"recharge_time: {recharge_time}")
            # # print(f"Depot Nodes: {depot_nodes}")

            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_trip_times_in_route(G, vehicle_routes)
            
            print(f"Loaded vehicle routes for scenario {scenario_num}: {vehicle_routes}")
            print(f"Vehicle trip times for scenario {scenario_num}: {vehicle_trip_times}")

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
            instanceData['Instance Name'].append(f"bccm.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
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
            
            all_mf_results = []
            start = time.time()
            
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

                    # # print(f"Detected failed vehicles at time {t}: {detected_failed_vehicles}")
                    # # print(f"Recent failure vehicle: {recent_failure_vehicle}")
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
                    
                    # Simulated annealing function call
                    vehicle_routes, vehicle_trip_times, mf_results = handle_vehicle_failure_with_magnetic_field(
                        G, t, failure_history, vehicle_routes, vehicle_trip_times,
                        vehicle_trip_index, depot_nodes, failure_trips, vehicle_capacity,
                        recharge_time, uavLocation, recent_failure_vehicle
                    )
                    
                    # print(f"Vehicle routes after handling failure: {vehicle_routes}")
                    # print(f"Vehicle trip times after handling failure: {vehicle_trip_times}")
                    # print(f"Idle time after handling failure: {idle_time}")

                    # Store SA results if available
                    if mf_results:
                        all_mf_results.append(mf_results)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            print(f"Final Vehicle routes: {vehicle_routes}")
            print(f"Final Vehicle trip times: {vehicle_trip_times}")
            print(f"Total mission time after SA algorithm: {mission_time}")

            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)

            # Aggregate SA metrics across all failure events
            if all_mf_results:
                all_mean_times = [result['mean_time'] for result in all_mf_results]
                all_std_times = [result['std_time'] for result in all_mf_results]
                all_best_times = [result['best_time'] for result in all_mf_results]
                all_mean_mission_times = [result['mean_mission_time'] for result in all_mf_results]
                all_std_mission_times = [result['std_mission_time'] for result in all_mf_results]
                all_best_mission_times = [result['best_mission_time'] for result in all_mf_results]
                
                instanceData['Mean Execution Time SA'].append(round(np.mean(all_mean_times), 3))
                instanceData['STDEV Execution Time SA'].append(round(np.mean(all_std_times), 3))
                instanceData['Best Execution Time SA'].append(round(np.min(all_best_times), 3))
                instanceData['Mean Mission Time SA'].append(round(np.mean(all_mean_mission_times), 1))
                instanceData['STDEV Mission Time SA'].append(round(np.mean(all_std_mission_times), 1))
                instanceData['Best Mission Time SA'].append(round(np.min(all_best_mission_times), 1))
            else:
                # No SA runs occurred
                instanceData['Mean Execution Time SA'].append(0)
                instanceData['STDEV Execution Time SA'].append(0)
                instanceData['Best Execution Time SA'].append(0)
                instanceData['Mean Mission Time SA'].append(0)
                instanceData['STDEV Mission Time SA'].append(0)
                instanceData['Best Mission Time SA'].append(0)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"magnetic_field_bccm_new.csv", index=True)

# EGLESE
def simulate_3(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": []
    }
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": [],
        "Mean Execution Time SA": [], "STDEV Execution Time SA": [], "Best Execution Time SA": [],
        "Mean Mission Time SA": [], "STDEV Mission Time SA": [], "Best Mission Time SA": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/eglese_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/EGLESE_Failure_Scenarios_Results/results/eglese_failure_scenarios_solutions"
    
    # Get all scenario files
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]  # Extract scenario number from gdb.X.txt
        
        print()
        print("-"*50)
        print(f'Running GDB scenario {scenario_num}')
        if True:#int(scenario_num) == 1:  # Process only scenario 1 for testing
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
          
            # print(f"Failure Vehicles: {failure_vehicles}")
            # print(f"Vehicle Failure Times: {vehicle_failure_times}")
            # print(f"recharge_time: {recharge_time}")
            # print(f"Depot Nodes: {depot_nodes}")

            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_trip_times_in_route(G, vehicle_routes)
            
            print(f"Loaded vehicle routes for scenario {scenario_num}: {vehicle_routes}")
            print(f"Vehicle trip times for scenario {scenario_num}: {vehicle_trip_times}")

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
            instanceData['Instance Name'].append(f"eglese.{scenario_num}")
            instanceData['Number of Nodes'].append(G.number_of_nodes())
            instanceData['Number of Edges'].append(G.number_of_edges())
            instanceData["Number of Required Edges"].append(len(required_edges))
            instanceData['Capacity'].append(vehicle_capacity)
            instanceData['Recharge Time'].append(recharge_time)
            instanceData['Number of Depot Nodes'].append(len(depot_nodes))
            instanceData['Maximum Trip Time'].append(mission_time)
            instanceData['Number of vehicles failed'].append(len(failure_vehicles))
            instanceData['Total Number of Vehicles'].append(len(vehicle_routes))
            
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
            
            all_mf_results = []
            start = time.time()
            
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

                    # print(f"Detected failed vehicles at time {t}: {detected_failed_vehicles}")
                    # print(f"Recent failure vehicle: {recent_failure_vehicle}")
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
                    
                    # Simulated annealing function call
                    vehicle_routes, vehicle_trip_times, mf_results = handle_vehicle_failure_with_magnetic_field(
                        G, t, failure_history, vehicle_routes, vehicle_trip_times,
                        vehicle_trip_index, depot_nodes, failure_trips, vehicle_capacity,
                        recharge_time, uavLocation, recent_failure_vehicle
                    )
                    
                    # print(f"Vehicle routes after handling failure: {vehicle_routes}")
                    # print(f"Vehicle trip times after handling failure: {vehicle_trip_times}")
                    

                    # Store SA results if available
                    if mf_results:
                        all_mf_results.append(mf_results)
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            print(f"Final Vehicle routes: {vehicle_routes}")
            print(f"Final Vehicle trip times: {vehicle_trip_times}")
            print(f"Total mission time after SA algorithm: {mission_time}")

            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)

            # Aggregate SA metrics across all failure events
            if all_mf_results:
                all_mean_times = [result['mean_time'] for result in all_mf_results]
                all_std_times = [result['std_time'] for result in all_mf_results]
                all_best_times = [result['best_time'] for result in all_mf_results]
                all_mean_mission_times = [result['mean_mission_time'] for result in all_mf_results]
                all_std_mission_times = [result['std_mission_time'] for result in all_mf_results]
                all_best_mission_times = [result['best_mission_time'] for result in all_mf_results]
                
                instanceData['Mean Execution Time SA'].append(round(np.mean(all_mean_times), 3))
                instanceData['STDEV Execution Time SA'].append(round(np.mean(all_std_times), 3))
                instanceData['Best Execution Time SA'].append(round(np.min(all_best_times), 3))
                instanceData['Mean Mission Time SA'].append(round(np.mean(all_mean_mission_times), 1))
                instanceData['STDEV Mission Time SA'].append(round(np.mean(all_std_mission_times), 1))
                instanceData['Best Mission Time SA'].append(round(np.min(all_best_mission_times), 1))
            else:
                # No SA runs occurred
                instanceData['Mean Execution Time SA'].append(0)
                instanceData['STDEV Execution Time SA'].append(0)
                instanceData['Best Execution Time SA'].append(0)
                instanceData['Mean Mission Time SA'].append(0)
                instanceData['STDEV Mission Time SA'].append(0)
                instanceData['Best Mission Time SA'].append(0)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"magnetic_field_eglese_new.csv", index=True)



if __name__ == "__main__":
    simulate_1(save_results_to_csv=True)
    # simulate_2(save_results_to_csv=False)
    # simulate_3(save_results_to_csv=True)