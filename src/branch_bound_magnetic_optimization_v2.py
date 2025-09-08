import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import time
import os
import pandas as pd
import copy
import re
from centralized_auction_new import *
from magnetic_field import MagneticFieldRouter, IntelligentCapacityTuner
from itertools import combinations
import heapq
plt.rcParams["figure.dpi"] = 300


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
        
    def calculate_actual_solution_cost(self, start_depot, end_depot, required_edges):
        """
        Calculate actual cost using magnetic field with intelligent tuning
        This is used as both lower bound and actual solution
        """
        # Use intelligent capacity tuner
        tuner = IntelligentCapacityTuner(self.graph, start_depot, end_depot, 
                                        self.vehicle_capacity, required_edges)
        tuner.random_search(n_iterations=50)  # Quick evaluation for bounding
        
        if tuner.best_capacity and tuner.best_route:
            print(f'Intelligent tuning successful: capacity {tuner.best_capacity}, cost {tuner.best_cost}, covered {tuner.num_required_edges_covered}/{len(required_edges)}')
            return tuner.best_cost, True  # cost, feasible
        else:
            return float('inf'), False
        
    
    def branch_and_bound_optimization(self, vehicles_to_optimize, required_edges_per_vehicle, 
                                    original_costs, upper_bound, start_positions,  verbose=False):
        """
        Branch and bound optimization exploring depot combinations
        """
        if verbose:
            print(f"Starting branch and bound with {len(vehicles_to_optimize)} vehicles")
            print(f"Upper bound (centralized auction total): {upper_bound}")
        
        # For single vehicle case, just try all depot combinations
        if len(vehicles_to_optimize) == 1:
            vehicle_idx = vehicles_to_optimize[0]
            required_edges = required_edges_per_vehicle[vehicle_idx]
            original_cost = original_costs[vehicle_idx]
            
            # Determine start depot (where vehicle currently is)
            start_depot = start_positions[vehicle_idx]#required_edges[0][0] if required_edges else self.depot_nodes[0]
            
            best_cost = original_cost
            best_solution = None
            
            if verbose:
                print(f"Single vehicle optimization - trying {len(self.depot_nodes)} end depots")
            
            for end_depot in self.depot_nodes:
                print(f'Inputs for intelligent tuning start depot {start_depot}, end depot {end_depot}, required edges {required_edges} ')
                cost, feasible = self.calculate_actual_solution_cost(start_depot, end_depot, required_edges)
                self.nodes_explored += 1
                
                if feasible and cost < best_cost:
                    # Get the actual detailed solution
                    route, detailed_cost, covered = self.design_trip_with_intelligent_tuning(
                        start_depot, end_depot, required_edges
                    )
                    if route and covered == len(required_edges):
                        best_cost = detailed_cost
                        best_solution = {vehicle_idx: (route, detailed_cost, covered)}
                        if verbose:
                            print(f"Better solution found: depot {end_depot}, cost {detailed_cost:.2f} vs original {original_cost:.2f}")
                elif cost >= best_cost:
                    self.nodes_pruned += 1
            
            self.best_cost = best_cost
            self.best_solution = best_solution if best_solution else {}
            
            if verbose:
                print(f"Single vehicle optimization complete: best cost {self.best_cost:.2f}")
            
            return self.best_solution, self.best_cost
        
        # For multiple vehicles, use simplified approach due to complexity
        # Try to optimize each vehicle independently and combine results
        combined_solutions = {}
        combined_cost = 0.0
        any_improvement = False
        
        if verbose:
            print(f"Multiple vehicle optimization - independent optimization")
        
        for vehicle_idx in vehicles_to_optimize:
            required_edges = required_edges_per_vehicle[vehicle_idx]
            original_cost = original_costs[vehicle_idx]
            
            # start_depot = required_edges[0][0] if required_edges else self.depot_nodes[0]
            start_depot = start_positions[vehicle_idx]

            best_vehicle_cost = original_cost
            best_vehicle_solution = None
            
            for end_depot in self.depot_nodes:
                cost, feasible = self.calculate_actual_solution_cost(start_depot, end_depot, required_edges)
                print(f'cost = P{cost}, feasible = {feasible}')
                self.nodes_explored += 1
                
                if feasible and cost < best_vehicle_cost:
                    route, detailed_cost, covered = self.design_trip_with_intelligent_tuning(
                        start_depot, end_depot, required_edges
                    )
                    if route and covered == len(required_edges):
                        best_vehicle_cost = detailed_cost
                        best_vehicle_solution = (route, detailed_cost, covered)
                        any_improvement = True
                        if verbose:
                            print(f"Vehicle {vehicle_idx + 1}: improved {original_cost:.2f} -> {detailed_cost:.2f}")
                elif cost >= best_vehicle_cost:
                    self.nodes_pruned += 1
            
            if best_vehicle_solution:
                combined_solutions[vehicle_idx] = best_vehicle_solution
                combined_cost += best_vehicle_cost
            else:
                combined_cost += original_cost  # No improvement, use original
        
        if any_improvement and combined_cost < upper_bound:
            self.best_cost = combined_cost
            self.best_solution = combined_solutions
        else:
            self.best_cost = upper_bound
            self.best_solution = {}
        
        if verbose:
            print(f"Multiple vehicle optimization complete: best combined cost {self.best_cost:.2f}")
        
        return self.best_solution, self.best_cost


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
            original_cost = sum(original_future_times) + (len(original_future_times) - 1) * recharge_time
            original_costs[vehicle_idx] = original_cost
    
    print(f"Vehicles to optimize: {vehicles_to_optimize}")
    print(f"Required edges per vehicle: {required_edges_per_vehicle}")
    print(f"Original costs per vehicle: {original_costs}")
    print(f"Start positions: {start_positions}")
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
    best_solutions, best_total_cost = optimizer.branch_and_bound_optimization(
        vehicles_to_optimize, required_edges_per_vehicle, original_costs, upper_bound, start_positions, verbose=verbose
    )
    
    # Apply improvements if found
    if best_solutions and best_total_cost < upper_bound:
        for vehicle_idx, (route, cost, covered) in best_solutions.items():
            current_trip_idx = vehicle_trip_index[vehicle_idx] + 1
            
            # Replace future trips with optimized route
            improved_routes[vehicle_idx] = (vehicle_routes[vehicle_idx][:current_trip_idx] + [route])
            improved_trip_times[vehicle_idx] = (vehicle_trip_times[vehicle_idx][:current_trip_idx] + [cost])
            
            improvement = original_costs[vehicle_idx] - cost
            total_improvement += improvement
            optimization_applied = True
            
            if verbose:
                print(f"Vehicle {vehicle_idx + 1} optimized: {original_costs[vehicle_idx]:.2f} -> {cost:.2f} "
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
                    
                    print(f"vehicle routes after auction: {vehicle_routes}")
                    print(f"vehicle trip times after auction: {vehicle_trip_times}")
                    num_function_calls += 1
                    detected_failed_vehicles = {}
                    
                    print(f"\n--- Branch & Bound Optimization at t={t} ---")
                    pre_opt_mission_time = round(max([sum(vehicle_trip_times[k]) + (len(vehicle_trip_times[k]) - 1) * recharge_time +
                                                     sum(idle_time[k].values()) for k in range(len(vehicle_routes))]), 1)
                    
                    
                    
                    # Apply branch & bound optimization
                    optimized_routes, optimized_trip_times, optimization_applied = optimize_with_branch_bound_post_auction(
                        G, vehicle_routes, vehicle_trip_times, depot_nodes, required_edges, 
                        vehicle_capacity, recharge_time, vehicle_trip_index, failure_history, verbose=True
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
                df.to_csv(f"{p}/results/instances_results_with_failure/centralized_auction_branch_bound_gdb.csv", index=True)


if __name__ == "__main__":
    simulate_1_with_branch_bound_optimization(save_results_to_csv=False)