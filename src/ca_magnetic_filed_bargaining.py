import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import math
import time
import os
import pandas as pd
import copy
import re
from centralized_auction_new import *
plt.rcParams["figure.dpi"] = 300

# class MagneticFieldRouter:
#     def __init__(self, graph, start_depot, end_depot, capacity, alpha=1.0, gamma=1.0):
#         self.graph = graph
#         self.start_depot = start_depot
#         self.end_depot = end_depot
#         self.capacity = capacity
#         self.alpha = alpha
#         self.gamma = gamma
#         self.pos = self._create_layout()
#         self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
#     def _create_layout(self):
#         return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
#     def calculate_distances(self):
#         return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
#     def calculate_required_edge_influence(self, required_edges_to_cover):
#         distances = self.calculate_distances()
#         influences = {}
        
#         for edge in self.graph.edges():
#             influences[edge] = {}
#             influences[edge[::-1]] = {}
            
#             for i, req_edge in enumerate(required_edges_to_cover):
#                 u_req, v_req = req_edge
#                 u_edge, v_edge = edge
                
#                 d1 = min(distances[u_edge].get(u_req, float('inf')), 
#                         distances[u_edge].get(v_req, float('inf')))
#                 d2 = min(distances[v_edge].get(u_req, float('inf')), 
#                         distances[v_edge].get(v_req, float('inf')))
                
#                 if d1 != float('inf') and d2 != float('inf'):
#                     influence = 0.5 * (exp(-self.alpha * d1) + exp(-self.alpha * d2))
#                 else:
#                     influence = 0.0
                
#                 influences[edge][f'req_{i}'] = influence
#                 influences[edge[::-1]][f'req_{i}'] = influence
        
#         return influences
    
#     def calculate_depot_influence(self):
#         distances = self.calculate_distances()
#         influences = {}
        
#         for edge in self.graph.edges():
#             u, v = edge
#             d_u = min(distances[u].get(self.start_depot, float('inf')),
#                      distances[u].get(self.end_depot, float('inf')))
#             d_v = min(distances[v].get(self.start_depot, float('inf')),
#                      distances[v].get(self.end_depot, float('inf')))
            
#             if d_u != float('inf') and d_v != float('inf'):
#                 influence = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
#                                  exp(-self.gamma * d_v / self.capacity))
#             else:
#                 influence = 0.1
            
#             influences[edge] = influence
#             influences[edge[::-1]] = influence
        
#         return influences

#     def calculate_edge_score(self, edge, required_to_cover, current_length, is_new_required=False, total_required_edges=0):
#         req_influences = self.calculate_required_edge_influence(required_to_cover)
#         depot_influences = self.calculate_depot_influence()
        
#         # print(f'required edge influences - {req_influences}')
#         P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
#         # print(f'P = {P}')
#         D = depot_influences[edge]
#         w = current_length / self.capacity if self.capacity > 0 else 0
#         if is_new_required:
#             S = 2*(1 - w) * P + w * D
#         else:
#             S = (1 - w) * P + w * D
#         final_score = S
#         # if is_new_required:
#         #     final_score = S + 1000
#         #     final_score = 1000/total_required_edges + S  # Large bonus for uncovered required edges
#         # else:
#         #     final_score = S + (2*self.capacity)/(self.capacity - current_length + 1)
            
#         return {
#             'P': P,
#             'D': D,
#             'w': w,
#             'S': S,
#             'final_score': final_score,
#             'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
#             'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
#         }
    
#     def find_trip_with_magnetic_scoring(self, required_edges, verbose=False):
#         current_route = [self.start_depot]
#         current_length = 0
#         required_covered = set()
#         max_iterations = len(self.graph.edges()) * 10
#         iteration_count = 0
#         total_required_edges = len(required_edges)

#         while len(required_covered) < total_required_edges and iteration_count < max_iterations:
#             current_node = current_route[-1]
#             candidates = []
#             iteration_count += 1
            
#             # Update required edges to cover
#             required_to_cover = [req for req in required_edges if tuple(sorted(req)) not in required_covered]
            
#             # print(f'required edges to cover - {required_to_cover, required_covered}')
#             # Get all possible next edges
#             for neighbor in self.graph.neighbors(current_node):
                
                
                
#                 edge = (current_node, neighbor)
#                 edge_sorted = tuple(sorted(edge))
#                 edge_weight = self.graph[current_node][neighbor]['weight']
                
#                 # Capacity check
#                 try:
#                     if neighbor == self.end_depot:
#                         path_to_end_length = 0
#                     else:
#                         path_to_end_length = nx.shortest_path_length(
#                             self.graph, neighbor, self.end_depot, weight='weight'
#                         )
                    
#                     # print(f"Evaluating edge ({current_node}, {neighbor})")
#                     # print(f"  - Edge weight: {edge_weight}")
#                     # print(f"  - Path to end length: {path_to_end_length}")
#                     # print(f"  - Total would be: {current_length + edge_weight + path_to_end_length}")
#                     # print(f"  - Capacity: {self.capacity}")
#                     # print(f"  - Passes capacity check: {current_length + edge_weight + path_to_end_length <= self.capacity}")
#                     # print(f"  - Is required: {is_new_required}")
#                     if current_length + edge_weight + path_to_end_length > self.capacity:
#                         continue
#                 except nx.NetworkXNoPath:
#                     continue
                
#                 is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
#                                 and edge_sorted not in required_covered)
                
#                 # print(f'new required edge - {is_new_required}')
                
#                 score_data = self.calculate_edge_score(edge, required_to_cover, current_length, is_new_required, total_required_edges)
                
#                 # print(f'score data - {score_data}')
#                 candidates.append({
#                     'edge': edge,
#                     'neighbor': neighbor,
#                     'is_new_required': is_new_required,
#                     'score_data': score_data
#                 })

                
                
            
#             # Handle no candidates
#             if not candidates:
#                 if verbose:
#                     print(f"No direct candidates at node {current_node} - seeking uncovered required edges...")
                
#                 uncovered_required = required_to_cover
                
#                 if uncovered_required and current_node != self.end_depot:
#                     best_path = None
#                     best_length = float('inf')
#                     target_edge = None
                    
#                     for req_edge in uncovered_required:
#                         u, v = req_edge
#                         for target_node in [u, v]:
#                             try:
#                                 path = nx.shortest_path(self.graph, current_node, target_node, weight='weight')
#                                 path_length = nx.shortest_path_length(self.graph, current_node, target_node, weight='weight')
                                
#                                 try:
#                                     final_path_length = nx.shortest_path_length(self.graph, target_node, self.end_depot, weight='weight')
#                                     total_length = current_length + path_length + final_path_length
                                    
#                                     if total_length <= self.capacity and path_length < best_length:
#                                         best_path = path[1:]
#                                         best_length = path_length
#                                         target_edge = req_edge
#                                 except nx.NetworkXNoPath:
#                                     continue
#                             except nx.NetworkXNoPath:
#                                 continue
                    
#                     if best_path:
#                         if verbose:
#                             print(f"Forcing path {best_path} to reach required edge {target_edge}")
#                         for next_node in best_path:
#                             edge_weight = self.graph[current_route[-1]][next_node]['weight']
#                             current_route.append(next_node)
#                             current_length += edge_weight
#                             if verbose:
#                                 print(f"Forced step: ({current_route[-2]}, {next_node}) -> Node {next_node}, Length: {current_length:.2f}")
#                         continue
                
#                 # Go to end depot if no further progress possible
#                 if current_node != self.end_depot:
#                     try:
#                         path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
#                         additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
#                         if current_length + additional_length <= self.capacity:
#                             current_route.extend(path_to_end[1:])
#                             current_length += additional_length
#                             if verbose:
#                                 print(f"Final path to end depot: {path_to_end[1:]}")
#                     except nx.NetworkXNoPath:
#                         pass
#                 break
            
#             # Sort candidates: prioritize new required edges
#             candidates.sort(key=lambda x: (
#                 not x['is_new_required'],
#                 -x['score_data']['final_score']
#             ))
            
#             best = candidates[0]
#             current_route.append(best['neighbor'])
#             current_length += best['score_data']['edge_weight']
            
#             if best['is_new_required']:
#                 required_covered.add(tuple(sorted(best['edge'])))
#                 if verbose:
#                     print(f"✓ Covered required edge: {best['edge']} ({len(required_covered)}/{len(required_edges)})")
            
#             if verbose:
#                 print(f"Step: {best['edge']} -> Node {best['neighbor']}, "
#                     f"Length: {current_length:.2f}, Required: {len(required_covered)}/{len(required_edges)}")
        
#         # Ensure end at depot
#         if current_route[-1] != self.end_depot:
#             try:
#                 path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
#                 additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
#                 current_route.extend(path_to_end[1:])
#                 current_length += additional_length
#             except nx.NetworkXNoPath:
#                 return None, float('inf'), len(required_covered)
        
#         if verbose:
#             print(f"Final route: {current_route}, Length: {current_length:.2f}")
#             print(f"Required covered: {len(required_covered)}/{len(required_edges)}")
        
#         # print(current_route, current_length, required_covered, required_edges, len(required_covered))
#         return current_route, current_length, len(required_covered)

class MagneticFieldRouter:
    """
    Parameter-free router that avoids oscillations and 'walks to nowhere'.
    Decisions are purely lexicographic over distances (no tuned scalars).

    Public API:
      find_trip_with_magnetic_scoring(required_edges, verbose=False)
        -> (trip:list[int], length:float, covered_count:int)
    """

    def __init__(self, graph, start_depot, end_depot, capacity):
        self.G = graph
        self.s = start_depot
        self.t = end_depot
        self.C = capacity

        # Cache single-source distances we need a lot:
        # - from every node to end depot (fast feasibility/back-to-depot checks)
        self.dist_to_t = nx.single_source_dijkstra_path_length(self.G, self.t, weight="weight")

        # Quick edge weight lookups (undirected graph assumed)
        self.w = {(u, v): d["weight"] for u, v, d in self.G.edges(data=True)}
        self.w.update({(v, u): w for (u, v), w in list(self.w.items())})

        # On-demand single-source cache to avoid all-pairs blowup.
        self._sssp_cache = {}

    # -------------------- distances & helpers --------------------

    def _d(self, src, dst):
        """Distance(src,dst) with on-demand single-source cache."""
        if src not in self._sssp_cache:
            self._sssp_cache[src] = nx.single_source_dijkstra_path_length(self.G, src, weight="weight")
        return self._sssp_cache[src].get(dst, float("inf"))

    def _edge_w(self, u, v):
        return self.w.get((u, v), float("inf"))

    def _node_to_edge_dist(self, node, edge):
        a, b = edge
        return min(self._d(node, a), self._d(node, b))

    def _edge_to_t_dist(self, edge):
        a, b = edge
        return min(self.dist_to_t.get(a, float("inf")), self.dist_to_t.get(b, float("inf")))

    def _H(self, node, remaining):
        """Monotone potential to 'finish the remaining work and reach t'."""
        if not remaining:
            return 0.0
        best = float("inf")
        for e in remaining:
            dn = self._node_to_edge_dist(node, e)
            best = min(best, dn + self._edge_to_t_dist(e))
        return best

    def _feasible_step(self, cur_len, u, v):
        """After stepping (u,v), can we still reach t within capacity?"""
        w = self._edge_w(u, v)
        back = self.dist_to_t.get(v, float("inf"))
        return (cur_len + w + back) <= self.C + 1e-9

    def _append_shortest_path(self, route, target):
        """Append sp(route[-1] -> target), return added length (inf if none)."""
        src = route[-1]
        if src == target:
            return 0.0
        try:
            path = nx.shortest_path(self.G, src, target, weight="weight")
        except nx.NetworkXNoPath:
            return float("inf")
        added = 0.0
        for a, b in zip(path, path[1:]):
            route.append(b)
            added += self._edge_w(a, b)
        return added

    # -------------------- main --------------------

    def find_trip_with_magnetic_scoring(self, required_edges, verbose=False):
        # work with undirected required set
        req = {tuple(sorted(e)) for e in required_edges}
        covered = set()

        route = [self.s]
        L = 0.0

        # short tabu on UNDIRECTED recent edges (prevents ping-pong)
        tabu = []
        TABU_LEN = 3

        steps = 0
        max_steps = max(10, 10 * self.G.number_of_edges())

        while steps < max_steps:
            steps += 1
            u = route[-1]

            remaining = [e for e in req if e not in covered]
            if not remaining:
                break  # all required covered; we'll go to depot after loop

            # Candidates that are feasible, and NOT immediate backtrack unless necessary
            neigh = []
            prev = route[-2] if len(route) >= 2 else None
            for v in self.G.neighbors(u):
                if not self._feasible_step(L, u, v):
                    continue
                e = tuple(sorted((u, v)))
                # Soft no-backtrack: skip u->prev unless it covers a remaining required edge
                # and no other neighbor can cover or reduce H.
                neigh.append((v, e))

            if not neigh:
                # Try a direct 'jump' to the closest endpoint of any remaining edge
                best = (float("inf"), None)  # (to_edge, endpoint)
                for a, b in remaining:
                    for x in (a, b):
                        to_x = self._d(u, x)
                        back = self.dist_to_t.get(x, float("inf"))
                        if L + to_x + back <= self.C + 1e-9 and to_x < best[0]:
                            best = (to_x, x)
                if best[1] is not None:
                    add = self._append_shortest_path(route, best[1])
                    if not math.isfinite(add):
                        break
                    L += add
                    continue

                # else: go home if possible
                add = self._append_shortest_path(route, self.t)
                if not math.isfinite(add) or L + add > self.C + 1e-9:
                    return None, float("inf"), len(covered)
                L += add
                break

            # Tag neighbors by attributes we care about
            Hu = self._H(u, remaining)

            cand = []
            for v, e in neigh:
                is_req = (e in remaining)
                w = self._edge_w(u, v)
                Hv = self._H(v, remaining if not is_req else [r for r in remaining if r != e])

                # immediate backtrack?
                backtrack = (prev is not None and v == prev)
                # recently used edge?
                in_tabu = (e in tabu)

                # Lexicographic score (minimize):
                #  0) prefer edges that cover a remaining required (is_req: False>True → (not is_req, ...) )
                #  1) prefer decreasing potential H (Hv < Hu), then equal, then increasing
                #  2) tabu?  (non-tabu before tabu)
                #  3) avoid backtrack unless it covers required or decreases H
                #  4) edge weight (shorter first)
                #  5) tie-breaker on node id (determinism)
                score = (
                    0 if is_req else 1,
                    0 if Hv < Hu - 1e-12 else (1 if abs(Hv - Hu) <= 1e-12 else 2),
                    0 if not in_tabu else 1,
                    0 if (not backtrack or is_req or Hv < Hu - 1e-12) else 1,
                    w,
                    v
                )
                cand.append((score, v, e, is_req, w, Hv))

            # Choose best neighbor
            cand.sort(key=lambda z: z[0])
            _, v, e, is_req, w, Hv = cand[0]

            # If the best move neither covers required nor decreases H,
            # and it's a tabu/backtrack edge while there exists a non-tabu neighbor
            # that at least doesn't worsen H, prefer that neighbor instead.
            if (not is_req and Hv > Hu + 1e-12):
                alt = [z for z in cand if (z[3] or z[5] <= Hu + 1e-12)]  # covers req or not-worsen H
                if alt:
                    cand2 = [z for z in alt if z[0][2] == 0]  # non-tabu in alt
                    pick = cand2[0] if cand2 else alt[0]
                    _, v, e, is_req, w, Hv = pick

            # Step
            route.append(v)
            L += w

            # update tabu
            tabu.append(e)
            if len(tabu) > TABU_LEN:
                tabu.pop(0)

            # mark coverage if required
            if is_req:
                covered.add(e)
                if verbose:
                    print(f"✓ cover {e}, len={L:.1f}, rem={len(req)-len(covered)}")

        # close to end depot
        if route[-1] != self.t:
            add = self._append_shortest_path(route, self.t)
            if not math.isfinite(add) or L + add > self.C + 1e-9:
                return None, float("inf"), len(covered)
            L += add

        # compress non-required stretches (keeps required coverage identical)
        route = self._compress(route, covered)
        # recompute length after compression
        L2 = 0.0
        for a, b in zip(route, route[1:]):
            L2 += self._edge_w(a, b)
        if L2 <= self.C + 1e-9:
            L = L2
        return route, L, len(covered)

    # -------------------- post-processing: path compression --------------------

    def _compress(self, route, covered_required):
        """
        Replace non-required detours with shortest paths between the same 'breakpoints'.
        Breakpoints = sequence of nodes where a required edge is entered/exited plus endpoints s,t.
        """
        if len(route) < 2:
            return route[:]

        reqE = set(covered_required)
        segs = []
        cur = [route[0]]
        prev = route[0]

        # Mark positions where a REQUIRED undirected edge is traversed.
        def is_req_edge(a, b):
            return tuple(sorted((a, b))) in reqE

        # Build a list of 'breakpoints' such that between two consecutive
        # breakpoints there is no required edge (only non-required).
        breakpoints = [route[0]]
        for a, b in zip(route, route[1:]):
            if is_req_edge(a, b):
                # end the current non-required chunk at 'a' (if any), then add 'a', 'b'
                if breakpoints[-1] != a:
                    breakpoints.append(a)
                breakpoints.append(b)
            else:
                # continue
                pass
        if breakpoints[-1] != route[-1]:
            breakpoints.append(route[-1])

        # Now reconnect consecutive breakpoints with exact shortest paths
        out = [breakpoints[0]]
        for a, b in zip(breakpoints, breakpoints[1:]):
            if a == out[-1]:
                # append path a->b except first node
                try:
                    p = nx.shortest_path(self.G, a, b, weight="weight")
                except nx.NetworkXNoPath:
                    return route  # fallback, keep original
                out.extend(p[1:])
            else:
                # we shouldn't be here, but robustly fix
                try:
                    p = nx.shortest_path(self.G, out[-1], b, weight="weight")
                except nx.NetworkXNoPath:
                    return route
                out.extend(p[1:])
        return out

def _eligible_vehicle_indices(V, vehicle_status=None, failure_history=None):
    failed = set(failure_history or [])
    if vehicle_status is None:
        return [k for k in range(V) if k not in failed]
    return [k for k in range(V) if (vehicle_status[k] is True) and (k not in failed)]

def _edge_key(u, v):
    return tuple(sorted((u, v)))

def _edge_in_trip(trip, e):
    a, b = tuple(sorted(e))
    for u, v in zip(trip, trip[1:]):
        if tuple(sorted((u, v))) == (a, b):
            return True
    return False

def _find_edge_index_in_trip(trip, e):
    a, b = tuple(sorted(e))
    for i, (u,v) in enumerate(zip(trip, trip[1:])):
        if tuple(sorted((u,v))) == (a,b):
            return i  # edge is (trip[i], trip[i+1])
    return None

def _seller_trip_index(vehicle_trips, e):
    for idx, trip in enumerate(vehicle_trips):
        if _edge_in_trip(trip, e):
            return idx
    return None

def _vehicle_completion_time(times_for_vehicle, recharge_time):
    if not times_for_vehicle:
        return 0.0
    return sum(times_for_vehicle) + (len(times_for_vehicle) - 1) * recharge_time

def _mission_time(vehicle_trip_times, recharge_time):
    return max(_vehicle_completion_time(v, recharge_time) for v in vehicle_trip_times if v is not None)

def _shortest_path_trip(G, s, t):
    if s == t:
        return None  # no connector needed
    return nx.shortest_path(G, s, t, weight='weight')

def _append_trip(routes, k, trip):
    if trip and len(trip) > 1:
        routes[k].append(trip)

def _insert_trip(routes, k, idx, trip):
    if trip and len(trip) > 1:
        routes[k].insert(idx, trip)

def _copy_routes(routes):
    return [[list(t) for t in veh] for veh in routes]

def _last_depot_of_vehicle(routes, k, start_depots):
    """End depot of the last trip, or start depot if no trips yet."""
    if routes[k]:
        return routes[k][-1][-1]
    return start_depots[k]

def _heal_after_sell(G, routes, k, removed_idx, start_depots):
    """
    After removing routes[k][removed_idx], heal continuity with a connector trip if needed:
    - If removed FIRST trip: connect start_depots[k] -> new first trip start.
    - If removed MIDDLE trip: connect prev trip end -> next trip start.
    - If removed LAST: nothing to do.
    """
    if not routes[k]:
        return
    if removed_idx == 0:
        start_next = routes[k][0][0]
        sp = _shortest_path_trip(G, start_depots[k], start_next)
        if sp is not None:
            _insert_trip(routes, k, 0, sp)
        return
    if removed_idx < len(routes[k]):
        prev_end = routes[k][removed_idx - 1][-1]
        next_start = routes[k][removed_idx][0]
        sp = _shortest_path_trip(G, prev_end, next_start)
        if sp is not None:
            _insert_trip(routes, k, removed_idx, sp)
        return
    # removed last: nothing

def _nearest_depot(G, node, depot_nodes):
    """Return the (depot, path) with minimum weighted distance from node."""
    best = None
    best_path = None
    best_cost = float("inf")
    for d in depot_nodes:
        try:
            path = nx.shortest_path(G, node, d, weight='weight')
            cost = nx.shortest_path_length(G, node, d, weight='weight')
            if cost < best_cost:
                best, best_path, best_cost = d, path, cost
        except nx.NetworkXNoPath:
            continue
    return best, best_path

def _split_trip_by_edge_and_remove_segment(G, trip, e, depot_nodes):
    """
    Remove only edge e=(u,v) from trip [n0,...,nm]. Keep prefix and suffix,
    and ALWAYS add the connector needed so the resulting pieces can be stitched
    together into depot→depot trips.

    Returns: replacement_trips (list of depot→depot-compatible trips), removed_segment
    """
    i = _find_edge_index_in_trip(trip, e)
    if i is None:
        return [trip], None

    u, v = trip[i], trip[i+1]
    prefix = trip[:i+1]      # includes u
    suffix = trip[i+1:]      # starts at v

    repl = []

    # Case analysis:
    # A) prefix and suffix both exist
    has_prefix = len(prefix) > 1 or (len(prefix) == 1)  # node exists (start depot node at least)
    has_suffix = len(suffix) > 1 or (len(suffix) == 1)

    # We want final trips to start/end at depots; we do it by inserting explicit connector paths as separate trips.
    if has_prefix and has_suffix:
        # If prefix can be a proper trip (>=2 nodes), keep it; otherwise it is a single node (start depot marker).
        if len(prefix) > 1:
            repl.append(prefix)

        # Connect prefix_end -> suffix_start (even if suffix is a single depot node)
        pe = prefix[-1]
        ss = suffix[0]
        conn = _shortest_path_trip(G, pe, ss)
        if conn is not None:
            repl.append(conn)

        # If suffix has length >1, keep it; if it is a single node and NOT a depot, push it to the nearest depot.
        if len(suffix) > 1:
            repl.append(suffix)
        else:
            # suffix is [node]; if it's not a depot, append a connector to nearest depot
            node = suffix[0]
            if node not in depot_nodes:
                _, path = _nearest_depot(G, node, depot_nodes)
                if path and len(path) > 1:
                    repl.append(path)

        return repl, trip[i:i+2]

    # B) only prefix exists
    if has_prefix and not has_suffix:
        if len(prefix) > 1:
            repl.append(prefix)
            # ensure it ends at depot
            end = prefix[-1]
            if end not in depot_nodes:
                _, path = _nearest_depot(G, end, depot_nodes)
                if path and len(path) > 1:
                    repl.append(path)
        else:
            # just a single node; push it to nearest depot (usually already a depot)
            node = prefix[0]
            if node not in depot_nodes:
                _, path = _nearest_depot(G, node, depot_nodes)
                if path and len(path) > 1:
                    repl.append(path)
        return repl, trip[i:i+2]

    # C) only suffix exists
    if has_suffix and not has_prefix:
        if len(suffix) > 1:
            # make sure it starts at a depot; if not, add connector from nearest depot -> start
            start = suffix[0]
            if start not in depot_nodes:
                # choose nearest depot INTO start
                # (we add depot->start path as a separate trip, then keep suffix)
                best = None; bestp = None; bestc = float("inf")
                for d in depot_nodes:
                    try:
                        path = nx.shortest_path(G, d, start, weight='weight')
                        cost = nx.shortest_path_length(G, d, start, weight='weight')
                        if cost < bestc:
                            best, bestp, bestc = d, path, cost
                    except nx.NetworkXNoPath:
                        continue
                if bestp and len(bestp) > 1:
                    repl.append(bestp)
            repl.append(suffix)
        else:
            # single node; if not depot, push it to nearest depot
            node = suffix[0]
            if node not in depot_nodes:
                _, path = _nearest_depot(G, node, depot_nodes)
                if path and len(path) > 1:
                    repl.append(path)
        return repl, trip[i:i+2]

    # D) neither exists (shouldn’t happen)
    return [], trip[i:i+2]

def _normalize_depot_endpoints(G, routes, start_depots, depot_nodes):
    """
    Ensure every trip starts/ends at a depot by inserting connector trips:
      - if trip starts at non-depot: add nearest-depot -> start as a new trip BEFORE it
      - if trip ends at non-depot: add end -> nearest-depot as a new trip AFTER it
    Also ensure the very first trip is reachable from the vehicle's start depot.
    """
    V = len(routes)
    out = [[] for _ in range(V)]
    for k in range(V):
        prev_end = start_depots[k]
        for trip in routes[k]:
            # connect from prev_end to trip start, if needed (explicit connector trip)
            if prev_end != trip[0]:
                sp = _shortest_path_trip(G, prev_end, trip[0])
                if sp is not None and len(sp) > 1:
                    out[k].append(sp)

            # if trip starts at non-depot, add connector from nearest depot to its start AS A SEPARATE TRIP
            if trip[0] not in depot_nodes:
                best, path = None, None
                # pick a depot that leads cheaply into trip[0]
                best = None; path = None; bestc = float("inf")
                for d in depot_nodes:
                    try:
                        p = nx.shortest_path(G, d, trip[0], weight='weight')
                        c = nx.shortest_path_length(G, d, trip[0], weight='weight')
                        if c < bestc:
                            bestc = c; path = p
                    except nx.NetworkXNoPath:
                        continue
                if path and len(path) > 1:
                    out[k].append(path)

            out[k].append(trip)

            # if trip ends at non-depot, append connector to nearest depot
            if trip[-1] not in depot_nodes:
                _, path = _nearest_depot(G, trip[-1], depot_nodes)
                if path and len(path) > 1:
                    out[k].append(path)
                prev_end = path[-1] if path else trip[-1]
            else:
                prev_end = trip[-1]
    return out

def _insert_at_best_position(G, routes_base, times_base, buyer, new_trip, recharge_time, start_depots, depot_nodes, required_edges):
    """
    Try inserting 'new_trip' at several indices in buyer’s route.
    For each candidate index:
      - add connector from prev_end_depot -> new_trip[0]
      - add connector from new_trip[-1] -> next_start_depot
      - recompute times with calculate_route_times()
    Returns (best_routes, best_times, best_mission) or (None, None, +inf) if infeasible.
    """
    best_routes, best_times, best_mission = None, None, float('inf')
    base_routes = _copy_routes(routes_base)
    route = base_routes[buyer]
    N = len(route)

    # candidate indices: 0, N, and the two positions nearest to the shortest connector
    candidate_idxs = set([0, N])
    # Heuristic: try to insert near the end for most cases
    if N >= 1:
        candidate_idxs.add(N-1)

    for idx in sorted(candidate_idxs):
        test_routes = _copy_routes(base_routes)

        # connector from "left depot" to new_trip start
        left_depot = start_depots[buyer] if idx == 0 else test_routes[buyer][idx-1][-1]
        c1 = _shortest_path_trip(G, left_depot, new_trip[0])
        _insert_trip(test_routes, buyer, idx, c1) if c1 else None
        _insert_trip(test_routes, buyer, idx + (1 if c1 else 0), new_trip)

        # connector from new_trip end to the next trip start (if any)
        idx_after = idx + (1 if c1 else 0) + 1
        if idx < len(route):  # there's a next original trip at this position
            right_start = route[idx][0]
            c2 = _shortest_path_trip(G, new_trip[-1], right_start)
            if c2:
                _insert_trip(test_routes, buyer, idx_after, c2)

        # normalize depot endpoints (keeps CA happy)
        test_routes = _normalize_depot_endpoints(G, test_routes, start_depots, depot_nodes)

        # coverage check (hard constraint)
        ok_cov, missing = verify_required_coverage(test_routes, required_edges)
        if not ok_cov:
            continue

        test_times = calculate_route_times(G, test_routes)
        test_mission = _mission_time(test_times, recharge_time)

        if test_mission < best_mission:
            best_routes, best_times, best_mission = test_routes, test_times, test_mission

    return best_routes, best_times, best_mission

def _nearest_k_depots(G, node, depot_nodes, k=2):
    cand = []
    for d in depot_nodes:
        try:
            c = nx.shortest_path_length(G, node, d, weight='weight')
            cand.append((c, d))
        except nx.NetworkXNoPath:
            pass
    cand.sort()
    return [d for _, d in cand[:k]]

def covered_required_edges(vehicle_routes, required_edges):
    rset = {tuple(sorted(e)) for e in required_edges}
    covered = set()
    for veh in vehicle_routes:
        for trip in veh:
            for u, v in zip(trip, trip[1:]):
                ek = _edge_key(u, v)
                if ek in rset:
                    covered.add(ek)
    return covered

def verify_required_coverage(vehicle_routes, required_edges):
    rset = {tuple(sorted(e)) for e in required_edges}
    cov = covered_required_edges(vehicle_routes, rset)
    missing = rset - cov
    return (len(missing) == 0), missing


def peer_auction_after_failure(
    G,
    vehicle_routes,
    vehicle_trip_times,
    depot_nodes,
    required_edges,
    vehicle_capacity,
    recharge_time,
    start_depots,            # per-vehicle start depot (uavLocation)
    failure_history=None,    # list of failed vehicle indices
    vehicle_status=None,     # list/array of booleans (True=healthy)
    max_rounds=2,
    try_multi_edge_bundle=True,   # NEW: allow MF to target a tiny bundle, not just one edge
    verbose=False,
):
    """
    Peer-auction among NON-FAILED vehicles with TRIP-SPLITTING SELL and coverage-safe BUY.

    SELL: split the seller's trip at edge e (remove only that edge segment), keep prefix/suffix,
          and add a connector trip between them if both exist.
    BUY : mint MF depot->depot trip for e (or a tiny bundle), add explicit connector from buyer_end
          to MF start, splice at END.
    HARD CONSTRAINT: Required-edges coverage must remain complete after each candidate move.
    """
    V = len(vehicle_routes)
    improved = False
    required_set = {tuple(sorted(e)) for e in required_edges}
    eligible = _eligible_vehicle_indices(V, vehicle_status, failure_history)
    if not eligible:
        return vehicle_routes, vehicle_trip_times, False

    # Optional: warn if CA output already misses something
    ok0, miss0 = verify_required_coverage(vehicle_routes, required_edges)
    if verbose and not ok0:
        print("[WARN] Baseline after CA is missing required edges:", miss0)

    for _ in range(max_rounds):
        # 1) pick bottleneck among eligible
        per_vehicle = [
            _vehicle_completion_time(vehicle_trip_times[k], recharge_time) if k in eligible else -float("inf")
            for k in range(V)
        ]
        bottleneck = max(range(V), key=lambda k: per_vehicle[k])
        if bottleneck not in eligible:
            break

        # 2) bundles = required edges on bottleneck's trips
        bundles = []
        for trip in vehicle_routes[bottleneck]:
            for u, v in zip(trip, trip[1:]):
                ek = _edge_key(u, v)
                if ek in required_set:
                    bundles.append(ek)
        bundles = list(dict.fromkeys(bundles))
        if not bundles:
            break

        base_mission = _mission_time(vehicle_trip_times, recharge_time)
        best_delta, best_state = 0.0, None

        for e in bundles:
            # locate seller trip
            seller, seller_trip_idx = None, None
            for k in eligible:
                idx = _seller_trip_index(vehicle_routes[k], e)
                if idx is not None:
                    seller, seller_trip_idx = k, idx
                    break
            if seller is None:
                continue

            # Prepare a tiny local "bundle" around e (optional, improves MF quality)
            bundle = [e]
            if try_multi_edge_bundle:
                trip = vehicle_routes[seller][seller_trip_idx]
                ei = _find_edge_index_in_trip(trip, e)
                # try to include the immediate predecessor/successor edges if they are required
                if ei is not None:
                    if ei > 0:
                        prev_edge = _edge_key(trip[ei-1], trip[ei])
                        if prev_edge in required_set:
                            bundle.append(prev_edge)
                    if ei+1 < len(trip)-1:
                        next_edge = _edge_key(trip[ei+1], trip[ei+2])
                        if next_edge in required_set:
                            bundle.append(next_edge)
                # unique
                bundle = list(dict.fromkeys(bundle))

            for buyer in eligible:
                if buyer == seller:
                    continue

                # SELL (split) on a working copy
                routes_try = _copy_routes(vehicle_routes)
                old_trip = routes_try[seller].pop(seller_trip_idx)

                replacement_trips, removed_seg = _split_trip_by_edge_and_remove_segment(
                    G, old_trip, e, depot_nodes
                )
                # Insert the replacement pieces back in place of the old trip
                ins_at = seller_trip_idx
                for nt in replacement_trips:
                    _insert_trip(routes_try, seller, ins_at, nt)
                    ins_at += 1

                # No extra healing needed beyond the split (we already added a connector inside replacement_trips if needed)

                # --- BUY: a few MF candidates around buyer's end/home depots
                buyer_end = _last_depot_of_vehicle(routes_try, buyer, start_depots)
                home = depot_nodes[buyer % len(depot_nodes)]
                near = _nearest_k_depots(G, buyer_end, depot_nodes, k=2)
                candidate_depots = set([buyer_end, home] + near)
                mf_candidates = []

                for s_d in candidate_depots:
                    for e_d in candidate_depots:
                        router = MagneticFieldRouter(G, s_d, e_d, vehicle_capacity)#, alpha=1.0, gamma=1.0)
                        mf_trip, mf_len, covered = router.find_trip_with_magnetic_scoring(bundle, verbose=False)
                        print(mf_trip, mf_len, covered)
                        if (mf_trip is None) or (covered < 1):
                            continue
                        if mf_len <= vehicle_capacity + 1e-9:
                            mf_candidates.append((s_d, e_d, mf_trip, mf_len))

                if not mf_candidates:
                    # restore and continue
                    routes_try = _copy_routes(vehicle_routes)
                    continue

                # --- Evaluate MF candidates: connector buyer_end -> s_d; splice at end; coverage check → times
                best_for_buyer, best_buyer_mission = None, float("inf")
                for s_d, e_d, mf_trip, mf_len in mf_candidates:
                    routes_eval = _copy_routes(routes_try)

                    # # buyer connector (buyer_end -> s_d)
                    # connector = _shortest_path_trip(G, buyer_end, s_d)
                    # _append_trip(routes_eval, buyer, connector)
                    # _append_trip(routes_eval, buyer, mf_trip)

                    # # *** NEW: enforce depot endpoints before coverage+timing ***
                    # routes_eval = _normalize_depot_endpoints(G, routes_eval, start_depots, depot_nodes)

                    # # coverage check
                    # ok_cov, missing = verify_required_coverage(routes_eval, required_edges)
                    # if not ok_cov:
                    #     if verbose:
                    #         print(f"[skip] trade {seller}->{buyer} on edge {e} breaks coverage:", missing)
                    #     continue

                    # times_eval = calculate_route_times(G, routes_eval)
                    # new_mission = _mission_time(times_eval, recharge_time)

                    # Try best insertion index (this internally adds the necessary connectors)
                    routes_eval, times_eval, new_mission = _insert_at_best_position(
                        G, routes_try, vehicle_trip_times, buyer, mf_trip,
                        recharge_time, start_depots, depot_nodes, copy.deepcopy(required_edges)
                    )
                    if routes_eval is None:
                        continue


                    if new_mission < best_buyer_mission:
                        best_buyer_mission = new_mission
                        best_for_buyer = (routes_eval, times_eval)

                if best_for_buyer and best_buyer_mission + 1e-9 < base_mission:
                    delta = base_mission - best_buyer_mission
                    if delta > best_delta:
                        best_delta, best_state = delta, best_for_buyer

        if best_state:
            vehicle_routes[:], vehicle_trip_times[:] = best_state
            improved = True
        else:
            # --- tiny paired look-ahead on the same seller ---
            for i in range(len(bundles)-1):
                e1, e2 = bundles[i], bundles[i+1]
                # locate contiguous in the same trip
                si = _seller_trip_index(vehicle_routes[bottleneck], e1)
                sj = _seller_trip_index(vehicle_routes[bottleneck], e2)
                if si is None or sj is None or si != sj:
                    continue

    # Final coverage sanity
    ok, missing = verify_required_coverage(vehicle_routes, required_edges)
    if verbose and not ok:
        print("[ERROR] Peer-auction ended with missing required edges:", missing)

    return vehicle_routes, vehicle_trip_times, improved

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
            
            print(f'required edge - {required_edges}')
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

                    print(f'vehicle routes after CA - {vehicle_routes}')
                    print(f'vehicle route times after CA - {vehicle_trip_times}')

                    # Implement peer-auction after failure here
                    # After centralized_auction_optimized(...)
                    vehicle_routes, vehicle_trip_times, _ = peer_auction_after_failure(
                                                                    G=G,
                                                                    vehicle_routes=vehicle_routes,
                                                                    vehicle_trip_times=vehicle_trip_times,
                                                                    depot_nodes=depot_nodes,
                                                                    required_edges=copy.deepcopy(required_edges),
                                                                    vehicle_capacity=vehicle_capacity,
                                                                    recharge_time=recharge_time,
                                                                    start_depots=uavLocation,          # <— IMPORTANT
                                                                    failure_history=failure_history,
                                                                    vehicle_status=vehicle_status,
                                                                    max_rounds=20,
                                                                    verbose=True,
                                                                )
                    # VERIFY coverage after peer auction
                    ok, missing = verify_required_coverage(vehicle_routes, copy.deepcopy(required_edges))
                    if not ok:
                        print("[ERROR] Required edges missing after peer-auction:", missing)
                    else:
                        print("[OK] All required edges are covered after peer-auction.")
                    print(f'vehicle routes after peer auction - {vehicle_routes}')
                    print(f'vehicle route times after peer auction - {vehicle_trip_times}')

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