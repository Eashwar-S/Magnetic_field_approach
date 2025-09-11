import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import time
import os
import pandas as pd
import copy
import itertools
import re
from centralized_auction_new import *
from magnetic_field import MagneticFieldRouter, IntelligentCapacityTuner
from itertools import combinations
import heapq
plt.rcParams["figure.dpi"] = 300


class Node:
    def __init__(self, route, route_time, trip_times, required_edges_to_be_traversed):
        self.route = route
        self.trip_times = trip_times
        self.route_time = route_time
        self.required_edges_to_be_traversed = required_edges_to_be_traversed


class BranchBoundMagneticOptimizer:
    def __init__(self, graph, depot_nodes, vehicle_capacity, recharge_time, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.depot_nodes = depot_nodes
        self.vehicle_capacity = vehicle_capacity
        self.recharge_time = recharge_time
        self.alpha = alpha
        self.gamma = gamma
        self.best_solution = None
        self.best_cost = float('inf')
        self.nodes_explored = 0
        self.nodes_pruned = 0
        
    def trip_using_magnetic_field(self, start_depot, end_depot, required_edges):
        """
        Designs trip using magnetic field with intelligent tuning.
        """
        # Use intelligent capacity tuner
        tuner = IntelligentCapacityTuner(self.graph, start_depot, end_depot, 
                                        self.vehicle_capacity, required_edges)
        tuner.random_search(n_iterations=100)  # Quick evaluation for bounding
        
        # if tuner.num_required_edges_covered >= 1:
        return tuner
        # return None
        
    # def refine_trip(self, trip, req_edges):
        
    #     traversal_trip_time = 0
    #     t_index = 0
    #     edges_traversed = False
    #     for i in range(len(trip) - 1):
    #         e_node = trip[i+1]
    #         traversal_trip_time += self.graph[trip[i]][trip[i+1]]['weight']
    #         edge = [trip[i], trip[i+1]]
    #         if edge in req_edges:
    #             req_edges.remove(edge)
    #         if edge[::-1] in req_edges:
    #             req_edges.remove(edge[::-1])

    #         if len(req_edges) == 0:
    #             edges_traversed = True
            
    #         if edges_traversed:
    #             if e_node in self.depot_nodes:
    #                 t_index = i+1
    #                 break
                
    #     return trip[:t_index], traversal_trip_time

    def refine_trip(self, trip, req_edges):
        """
        FIXED VERSION - stops at first depot after covering all required edges
        """
        req_edges_copy = req_edges.copy()  # Don't modify original
        traversal_trip_time = 0
        t_index = len(trip)  # Default to full trip
        edges_traversed = False
        
        for i in range(len(trip) - 1):
            e_node = trip[i+1]
            edge_weight = self.graph[trip[i]][trip[i+1]]['weight']
            traversal_trip_time += edge_weight
            
            edge = [trip[i], trip[i+1]]
            
            # Check if this edge is required (handle both directions)
            if edge in req_edges_copy:
                req_edges_copy.remove(edge)
            elif edge[::-1] in req_edges_copy:
                req_edges_copy.remove(edge[::-1])

            # Check if all required edges are covered
            if len(req_edges_copy) == 0:
                edges_traversed = True
            
            # If all edges covered and we reach a depot, stop here
            if edges_traversed and e_node in self.depot_nodes:
                t_index = i + 1
                break
        
        # Recalculate length for the actual returned trip
        if t_index < len(trip):
            actual_length = 0
            for i in range(t_index):
                if i < len(trip) - 1:
                    actual_length += self.graph[trip[i]][trip[i+1]]['weight']
            return trip[:t_index+1], actual_length  # +1 to include the depot node
        
        return trip, traversal_trip_time
    
    def branch_and_bound_optimization(self, vehicles_to_optimize, required_edges_per_vehicle, 
                                    original_costs, traversed_costs, upper_bound, start_positions,  verbose=False):
        """
        Branch and bound optimization exploring depot combinations
        """
        if verbose:
            print(f"Starting branch and bound with {len(vehicles_to_optimize)} vehicles")
            print(f"Upper bound (centralized auction total): {upper_bound}")
        
        
        
        if verbose:
            print(f"Branch and bound approach ")
        
        counter = itertools.count()
        routes = {vehicle_idx: None for vehicle_idx in vehicles_to_optimize}

        for vehicle_idx in vehicles_to_optimize:
            required_edges = required_edges_per_vehicle[vehicle_idx]
            original_cost = original_costs[vehicle_idx]
            traversed_cost = traversed_costs[vehicle_idx]

            best_vehicle_cost = original_cost
            route_time_so_far = 0#traversed_cost
            print(f'Starting routes time - {original_cost}')
            best_vehicle_solution = None
            best_trip_times = None
        
            
            root_node = Node(route=[], trip_times=[], route_time=0, required_edges_to_be_traversed=copy.deepcopy(required_edges))
            explored_nodes = []
            priority = len(root_node.required_edges_to_be_traversed)
            heapq.heappush(explored_nodes, (priority, next(counter), root_node))
            iteration = 0
            while explored_nodes:
                
                print(f'len of explored nodes - {len(explored_nodes)}')
                priority, _, node = heapq.heappop(explored_nodes)
                if len(node.route) == 0:
                    start_depot = start_positions[vehicle_idx]
                else:
                    start_depot = node.route[-1][-1]
                for end_depot in self.depot_nodes:
                    
                    iteration += 1
                    

                    print(f'------- Iteration - {iteration} ----------')

                    print(f'start, end depots - {start_depot, end_depot}')
                    print(f'node route so far - {node.route}')
                    print(f'Node required edges to be traversed - {node.required_edges_to_be_traversed}')

                    tuner = self.trip_using_magnetic_field(start_depot, end_depot, copy.deepcopy(node.required_edges_to_be_traversed))

                    print(f'tuner - {tuner}')
                    if tuner is not None:   
                        
                        print(f'before refining - {tuner.best_route, tuner.best_cost}')
                        tuner.best_route, tuner.best_cost = self.refine_trip(tuner.best_route, copy.deepcopy(tuner.required_edges_convered))
                        print(f'after refining - {tuner.best_route, tuner.best_cost}')
                        # print(f'required edges to be traversed - {node.required_edges_to_be_traversed}')
                        # print(f'turner required edges - {tuner.required_edges_convered}')
                        left_over_required_edges = [edge for edge in node.required_edges_to_be_traversed if edge not in tuner.required_edges_convered]
                        # print(f'left over required edges - {left_over_required_edges}')
                        route_time_so_far = node.route_time + tuner.best_cost + self.recharge_time
                        
                        
                        print(f'nodes required edges - {node.required_edges_to_be_traversed}')
                        print(f'required edges covered in current trip addition - {tuner.required_edges_convered}')
                        print(f'trip time - {tuner.best_route, tuner.best_cost}')
                        print(f'left over required edges - {left_over_required_edges}')
                        print(f'Node possible route - {node.route + [tuner.best_route]}')
                        print(f'Node routes time so far - {route_time_so_far}')
                        # print(f'')
                        if route_time_so_far <= original_cost:
                            if left_over_required_edges:
                                if node.route_time == 0: # root node
                                    # print(f'root node details')
                                    # print(f'route details - {tuner.best_route, tuner.best_cost}')
                                    child_node = Node(route=[tuner.best_route], trip_times=[tuner.best_cost], route_time=route_time_so_far, required_edges_to_be_traversed=left_over_required_edges)
                                else:
                                    new_route = node.route + [tuner.best_route]  # Use + instead of append
                                    new_trip_times = node.trip_times + [tuner.best_cost]
                                    child_node = Node(route=new_route, trip_times=new_trip_times, route_time=route_time_so_far, required_edges_to_be_traversed=left_over_required_edges)
                                priority = len(left_over_required_edges) + tuner.best_cost/tuner.max_capacity
                                heapq.heappush(explored_nodes, (len(left_over_required_edges), next(counter), child_node))
                            else:
                                if route_time_so_far <= best_vehicle_cost:
                                    print()
                                    print("------------- Entered here ------------------")
                                    
                                    best_vehicle_cost = route_time_so_far
                                    
                                    best_vehicle_solution = node.route + [tuner.best_route]
                                    best_trip_times = node.trip_times + [tuner.best_cost]
                                    print()
            
            if best_vehicle_solution is not None:
                routes[vehicle_idx] = (best_vehicle_solution, best_trip_times)
        # print(routes)
        return routes


def optimize_with_branch_bound_post_auction(G, vehicle_routes, vehicle_trip_times, depot_nodes, 
                                          required_edges, vehicle_capacity, recharge_time, 
                                          vehicle_trip_index, failure_history, verbose=False):
    """
    Optimize using branch and bound with magnetic field and intelligent tuning
    """    
    improved_routes = copy.deepcopy(vehicle_routes)
    improved_trip_times = copy.deepcopy(vehicle_trip_times)
    optimization_applied = False
    total_improvement = 0.0
    
    # Identify vehicles that can be optimized and their required edges
    vehicles_to_optimize = []
    required_edges_per_vehicle = {}
    original_costs = {}
    traversed_costs = {}
    start_positions = [-1] * len(vehicle_routes)
    
    for vehicle_idx in range(len(vehicle_routes)):

        if vehicle_idx not in failure_history:
            if vehicle_routes[vehicle_idx]:
                start_positions[vehicle_idx] = vehicle_routes[vehicle_idx][vehicle_trip_index[vehicle_idx]][-1]
            else:
                start_positions[vehicle_idx] = depot_nodes[vehicle_idx % len(depot_nodes)]

        # Skip failed vehicles and empty vehicles
        if vehicle_idx in failure_history or not vehicle_routes[vehicle_idx]:
            continue
        
        
        # Get current trip index for this vehicle
        current_trip_idx = vehicle_trip_index[vehicle_idx] + 1
        
        if current_trip_idx >= len(vehicle_routes[vehicle_idx]):
            continue
            
        # Extract future required edges
        future_required_edges = []
        future_trips = vehicle_routes[vehicle_idx][current_trip_idx:]
        
        for trip in future_trips:
            for i in range(len(trip) - 1):
                edge = [trip[i], trip[i + 1]]
                if edge in required_edges or edge[::-1] in required_edges:
                    normalized_edge = tuple(sorted(edge))
                    if normalized_edge not in [tuple(sorted(e)) for e in future_required_edges]:
                        future_required_edges.append(edge)
        
        if future_required_edges:
            vehicles_to_optimize.append(vehicle_idx)
            required_edges_per_vehicle[vehicle_idx] = future_required_edges
            
            # Calculate original cost for this vehicle's future trips
            original_future_times = vehicle_trip_times[vehicle_idx][current_trip_idx:]
            original_cost = sum(original_future_times) + (len(original_future_times)) * recharge_time
            original_costs[vehicle_idx] = original_cost

            traversed_times = vehicle_trip_times[vehicle_idx][:current_trip_idx]
            traversed_cost = sum(traversed_times) + (len(traversed_times) - 1) * recharge_time
            traversed_costs[vehicle_idx] = traversed_cost
    
    # print(f"Vehicles to optimize: {vehicles_to_optimize}")
    # print(f"Required edges per vehicle: {required_edges_per_vehicle}")
    # print(f"Original costs per vehicle: {original_costs}")
    # print(f'traversed costs per vehicle: {traversed_costs} ')
    # print(f"Start positions: {start_positions}")
    if not vehicles_to_optimize:
        if verbose:
            print("No vehicles available for branch and bound optimization")
        return improved_routes, improved_trip_times, optimization_applied
    
    if verbose:
        print(f"Branch and bound optimization for {len(vehicles_to_optimize)} vehicles")
        for v_idx in vehicles_to_optimize:
            print(f"Vehicle {v_idx + 1}: {len(required_edges_per_vehicle[v_idx])} required edges, "
                  f"original cost: {original_costs[v_idx]:.2f}")
    
    # Calculate upper bound (total original cost)
    upper_bound = sum(original_costs.values())
    
    # Initialize branch and bound optimizer
    optimizer = BranchBoundMagneticOptimizer(G, depot_nodes, vehicle_capacity, recharge_time)
    
    # Run branch and bound optimization
    best_solutions = optimizer.branch_and_bound_optimization(
        vehicles_to_optimize, required_edges_per_vehicle, original_costs, traversed_costs, upper_bound, start_positions, verbose=verbose
    )
    
    # Apply improvements if found
    
    for vehicle_idx, vehicle_route in best_solutions.items():
        
        if vehicle_route is not None:
            route, trip_times = vehicle_route
            current_trip_idx = vehicle_trip_index[vehicle_idx] + 1
            
            # print(f'improved route and route time - {route, trip_times}')
            # print(f'current trip index - {current_trip_idx}')

            # Replace future trips with optimized route
            improved_routes[vehicle_idx] = (vehicle_routes[vehicle_idx][:current_trip_idx] + route)
            improved_trip_times[vehicle_idx] = (vehicle_trip_times[vehicle_idx][:current_trip_idx] + trip_times)
            
            route_cost = (sum(trip_times) + len(trip_times)*recharge_time)
            improvement = original_costs[vehicle_idx] - route_cost
            total_improvement += improvement
            optimization_applied = True
            
            if verbose:
                print(f"Vehicle {vehicle_idx + 1} optimized: {original_costs[vehicle_idx]:.2f} -> {route_cost:.2f} "
                        f"(improvement: {improvement:.2f})")
    
    if verbose:
        if optimization_applied:
            print(f"Total improvement: {total_improvement:.2f}")
        else:
            print("No improvements found through branch and bound")
    
    return improved_routes, improved_trip_times, optimization_applied


def simulate_1_with_branch_bound_optimization(save_results_to_csv=False):
    p = '..'
    
    instanceData = {
        "Instance Name": [], "Number of Nodes": [], "Number of Edges": [], "Number of Required Edges": [],
        "Capacity": [], "Recharge Time": [], 'Total Number of Vehicles': [], "Number of Depot Nodes": [],
        "Number of vehicles failed": [], "Maximum Trip Time": [], "Number of Function Calls": [],
        "Execution time for auction algorithm in sec": [], "Maximum Trip Time after auction algorithm": [],
        "% increase in maximum trip time": [], 'idle time': [], "Average time per function call": [],
        "Branch-bound optimization applied": [], "Final Maximum Trip Time": [], "Total improvement": [],
        "Nodes explored": [], "Nodes pruned": []
    }

    txt_folder = f"{p}/dataset/new_failure_scenarios/gdb_failure_scenarios"
    sol_folder = f"{p}/results/instances_results_without_failure/GDB_Failure_Scenarios_Results/results/gdb_failure_scenarios_solutions"
    
    scenario_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for file in scenario_files:
        scenario_num = file.split('.')[1]
        
        if int(scenario_num) == 1:  # Process scenario 1 for testing
            print(f'Running GDB scenario {scenario_num} with Branch & Bound')
            
            # Parse txt file
            txt_path = os.path.join(txt_folder, file)
            G, required_edges, depot_nodes, vehicle_capacity, recharge_time, num_vehicles, failure_vehicles, vehicle_failure_times = parse_txt_file(txt_path)
            
            # Load solution routes
            sol_path = os.path.join(sol_folder, f"{scenario_num}.npy")
            if not os.path.exists(sol_path):
                continue
                
            vehicle_routes = np.load(sol_path, allow_pickle=True).tolist()
            vehicle_trip_times = calculate_route_times(G, vehicle_routes)
            
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
            
            # Run centralized auction with branch & bound optimization
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
            total_nodes_explored = 0
            total_nodes_pruned = 0
            
            start = time.time()
            print(f"\nStarting Centralized Auction with Branch & Bound")
            print(f"Initial mission time: {mission_time} time units")
            print(f"Required edges - {required_edges}")
            print(f'vehicle capacity - {vehicle_capacity}')
            print(f'depot nodes - {depot_nodes}')
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
                    
                    print(f"vehicle trip indexes before - {vehicle_trip_index}")
                    # Run centralized auction
                    vehicle_routes, vehicle_trip_times = centralized_auction_optimized(G, t, failure_history, vehicle_routes, 
                                                                                         vehicle_trip_times, vehicle_trip_index, 
                                                                                         depot_nodes, failure_trips, vehicle_capacity, 
                                                                                         recharge_time, uavLocation)
                    
                    print(f"\n--- Centralized auction results after failure at t={t} ---")
                    print(f"vehicle routes after auction: {vehicle_routes}")
                    print(f"vehicle trip times after auction: {vehicle_trip_times}")
                    # print()
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    print(f"\n--- Branch & Bound Optimization at t={t} ---")
                    pre_opt_mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                                     sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                    
                    
                    # Apply branch & bound optimization
                    optimized_routes, optimized_trip_times, optimization_applied = optimize_with_branch_bound_post_auction(
                        G, vehicle_routes, vehicle_trip_times, depot_nodes, required_edges, 
                        vehicle_capacity, recharge_time, vehicle_trip_index, failure_history, verbose=False
                    )

                    print(f"vehicle routes after branch & bound: {optimized_routes}")
                    print(f"vehicle trip times after branch & bound: {optimized_trip_times}")
                    
                    if optimization_applied:
                        vehicle_routes = optimized_routes
                        vehicle_trip_times = optimized_trip_times
                        post_opt_mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                                          sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                        improvement = pre_opt_mission_time - post_opt_mission_time
                        print(f"Branch & bound optimization applied: {pre_opt_mission_time} -> {post_opt_mission_time} (improvement: {improvement:.2f})")
                    else:
                        print("No branch & bound improvement found")
                    
                    mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                             sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                t += 0.1
                t = round(t, 1)
                
            end = time.time()
            execution_time = round(end - start, 3)
            avg_time_per_call = round(execution_time / num_function_calls, 3) if num_function_calls > 0 else 0
            
            print("Centralized Auction with Branch & Bound Complete")
            final_mission_time = mission_time
            total_improvement = previous_mission_time - final_mission_time
            
            total_idle_time = sum(sum(idle_time[k].values()) for k in range(len(vehicle_routes)))
            
            instanceData['idle time'].append(total_idle_time)
            instanceData['Maximum Trip Time after auction algorithm'].append(mission_time)
            instanceData['% increase in maximum trip time'].append(round(((mission_time - previous_mission_time) / previous_mission_time) * 100, 1))
            instanceData['Execution time for auction algorithm in sec'].append(execution_time)
            instanceData['Number of Function Calls'].append(num_function_calls)
            instanceData['Average time per function call'].append(avg_time_per_call)
            instanceData['Branch-bound optimization applied'].append(True)
            instanceData['Final Maximum Trip Time'].append(final_mission_time)
            instanceData['Total improvement'].append(total_improvement)
            instanceData['Nodes explored'].append(total_nodes_explored)
            instanceData['Nodes pruned'].append(total_nodes_pruned)
            
            print("\n" + "="*70)
            print("FINAL RESULTS - BRANCH & BOUND OPTIMIZATION")
            print("="*70)
            print(f"Original mission time: {previous_mission_time:.2f}")
            print(f"Final mission time: {final_mission_time:.2f}")
            print(f"Total improvement: {total_improvement:.2f}")
            print("="*70)
            
            if save_results_to_csv:
                df = pd.DataFrame(instanceData)
                df.to_csv(f"centralized_auction_branch_bound_gdb.csv", index=True)


if __name__ == "__main__":
    simulate_1_with_branch_bound_optimization(save_results_to_csv=False)