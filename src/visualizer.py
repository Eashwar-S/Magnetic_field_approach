"""
Enhanced Magnetic Field Router with Step-by-Step Visualization
============================================================
This module adds comprehensive visualization to the magnetic field routing algorithm.
Shows route building on the left and edge ranking matrix on the right.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from math import exp, sqrt
import time

class VisualizedMagneticFieldRouter:
    """
    Enhanced Magnetic Field Router with step-by-step visualization
    """
    
    def __init__(self, graph, start_depot, end_depot, capacity, alpha=1.0, gamma=1.0):
        self.graph = graph
        self.start_depot = start_depot
        self.end_depot = end_depot
        self.capacity = capacity
        self.alpha = alpha
        self.gamma = gamma
        self.pos = self._create_layout()
        self.max_edge_weight = max(d['weight'] for u, v, d in graph.edges(data=True))
        
        # Visualization settings
        self.node_size = 800
        self.edge_width = 3
        self.font_size = 12
        self.fig_size = (16, 8)
        
    def _create_layout(self):
        """Create consistent layout for visualizations"""
        return nx.spring_layout(self.graph, seed=42, k=2, iterations=50)
    
    def calculate_distances(self):
        """Calculate all shortest path distances"""
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='weight'))
    
    def calculate_required_edge_influence(self, required_edges):
        """Calculate influence of required edges on all other edges"""
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            influences[edge] = {}
            influences[edge[::-1]] = {}
            
            for i, req_edge in enumerate(required_edges):
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
    
    def calculate_depot_influence(self):
        """Calculate influence of depots on all edges"""
        distances = self.calculate_distances()
        influences = {}
        
        for edge in self.graph.edges():
            u, v = edge
            
            d_u = min(distances[u].get(self.start_depot, float('inf')),
                     distances[u].get(self.end_depot, float('inf')))
            d_v = min(distances[v].get(self.start_depot, float('inf')),
                     distances[v].get(self.end_depot, float('inf')))
            
            if d_u != float('inf') and d_v != float('inf'):
                influence = 0.5 * (exp(-self.gamma * d_u / self.capacity) + 
                                 exp(-self.gamma * d_v / self.capacity))
            else:
                influence = 0.1
            
            influences[edge] = influence
            influences[edge[::-1]] = influence
        
        return influences
    
    def calculate_edge_score(self, edge, required_edges, current_length, is_new_required=False):
        """Calculate the magnetic field score for an edge"""
        req_influences = self.calculate_required_edge_influence(required_edges)
        depot_influences = self.calculate_depot_influence()
        
        P = max(req_influences[edge].values()) if req_influences[edge] else 0.0
        D = depot_influences[edge]
        w = current_length / self.capacity if self.capacity > 0 else 0
        S = (1 - w) * P + w * D
        
        return {
            'P': P,
            'D': D,
            'w': w,
            'S': S,
            'final_score': S,
            'edge_weight': self.graph[edge[0]][edge[1]]['weight'],
            'normalized_weight': self.graph[edge[0]][edge[1]]['weight'] / self.max_edge_weight
        }
    
    def create_edge_ranking_matrix(self, current_node, required_edges, current_length, visited_edges):
        """Create NxN matrix showing edge rankings from current node"""
        n_nodes = len(self.graph.nodes())
        matrix = np.full((n_nodes, n_nodes), np.nan)
        
        # Get all possible edges from current node
        candidates = []
        for neighbor in self.graph.neighbors(current_node):
            edge = (current_node, neighbor)
            edge_sorted = tuple(sorted(edge))
            
            # Skip visited edges
            if edge_sorted in visited_edges:
                continue
                
            edge_weight = self.graph[current_node][neighbor]['weight']
            
            # Check capacity constraint
            try:
                if neighbor == self.end_depot:
                    path_to_end_length = 0
                else:
                    path_to_end_length = nx.shortest_path_length(
                        self.graph, neighbor, self.end_depot, weight='weight'
                    )
                
                if current_length + edge_weight + path_to_end_length > self.capacity:
                    continue
            except nx.NetworkXNoPath:
                continue
            
            is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges])
            score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
            
            candidates.append({
                'edge': edge,
                'neighbor': neighbor,
                'score': score_data['final_score'],
                'is_required': is_new_required
            })
        
        # Sort candidates by score (higher is better)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Fill matrix with rankings
        for rank, candidate in enumerate(candidates, 1):
            edge = candidate['edge']
            u, v = edge
            matrix[u, v] = rank
        
        return matrix, candidates
    
    def visualize_step(self, step_num, current_route, current_length, required_edges, 
                      visited_edges, required_covered, candidates_info, selected_edge=None):
        """Visualize a single step of the route building process"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.fig_size)
        
        # Left side: Route visualization
        self._draw_route_graph(ax1, current_route, visited_edges, required_edges, 
                              required_covered, selected_edge, current_length)
        
        # Right side: Edge ranking matrix
        if candidates_info:
            matrix, candidates = candidates_info
            self._draw_ranking_matrix(ax2, matrix, candidates, current_route)
        else:
            ax2.text(0.5, 0.5, 'No valid candidates', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=16)
            ax2.set_title('Edge Rankings')
        
        # Overall title
        current_node = current_route[-1] if current_route else self.start_depot
        fig.suptitle(f'Step {step_num}: Current Node = {current_node}, '
                    f'Route Length = {current_length:.2f}, '
                    f'Required Covered = {len(required_covered)}/{len(required_edges)}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Wait for user to close the figure
        plt.waitforbuttonpress()
        plt.close(fig)
    
    def _draw_route_graph(self, ax, current_route, visited_edges, required_edges, 
                         required_covered, selected_edge, current_length):
        """Draw the route graph on the left side"""
        ax.clear()
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(self.graph, self.pos, ax=ax, 
                              edge_color='lightgray', width=1, alpha=0.3)
        
        # Draw required edges
        required_edge_list = [tuple(sorted(edge)) for edge in required_edges]
        covered_edges = []
        uncovered_edges = []
        
        for edge in required_edge_list:
            if edge in required_covered:
                covered_edges.append(edge)
            else:
                uncovered_edges.append(edge)
        
        # Draw uncovered required edges in red
        if uncovered_edges:
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=uncovered_edges,
                                  ax=ax, edge_color='red', width=3, alpha=0.7)
        
        # Draw covered required edges in green
        if covered_edges:
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=covered_edges,
                                  ax=ax, edge_color='green', width=3, alpha=0.8)
        
        # Draw visited edges (route so far) in blue
        route_edges = []
        for i in range(len(current_route) - 1):
            edge = tuple(sorted([current_route[i], current_route[i+1]]))
            route_edges.append(edge)
        
        if route_edges:
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=route_edges,
                                  ax=ax, edge_color='blue', width=4, alpha=0.8)
        
        # Highlight the selected edge for this step
        if selected_edge:
            selected_edge_sorted = tuple(sorted(selected_edge))
            nx.draw_networkx_edges(self.graph, self.pos, edgelist=[selected_edge_sorted],
                                  ax=ax, edge_color='orange', width=6, alpha=0.9)
        
        # Draw nodes
        node_colors = []
        for node in self.graph.nodes():
            if node == self.start_depot:
                node_colors.append('lightgreen')
            elif node == self.end_depot:
                node_colors.append('lightcoral')
            elif node in current_route:
                node_colors.append('yellow')
            else:
                node_colors.append('white')
        
        nx.draw_networkx_nodes(self.graph, self.pos, ax=ax,
                              node_color=node_colors, node_size=self.node_size,
                              edgecolors='black', linewidths=2)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, self.pos, ax=ax, 
                               font_size=self.font_size, font_weight='bold')
        
        # Draw edge weights
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels, ax=ax, 
                                    font_size=10)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', lw=3, label='Required (uncovered)'),
            plt.Line2D([0], [0], color='green', lw=3, label='Required (covered)'),
            plt.Line2D([0], [0], color='blue', lw=4, label='Current route'),
            plt.Line2D([0], [0], color='orange', lw=6, label='Selected edge'),
            plt.scatter([], [], c='lightgreen', s=100, label='Start depot'),
            plt.scatter([], [], c='lightcoral', s=100, label='End depot'),
            plt.scatter([], [], c='yellow', s=100, label='Current path')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        ax.set_title(f'Route Building (Length: {current_length:.2f}/{self.capacity})')
        ax.axis('off')
    
    def _draw_ranking_matrix(self, ax, matrix, candidates, current_route):
        """Draw the edge ranking matrix on the right side"""
        ax.clear()
        
        n_nodes = len(self.graph.nodes())
        current_node = current_route[-1] if current_route else self.start_depot
        
        # Create a masked array to handle NaN values
        masked_matrix = np.ma.masked_invalid(matrix)
        
        # Create colormap
        cmap = plt.cm.RdYlBu_r
        cmap.set_bad(color='lightgray', alpha=0.3)
        
        # Draw the matrix
        if not np.all(np.isnan(matrix)):
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                vmin, vmax = valid_values.min(), valid_values.max()
                im = ax.imshow(masked_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                              aspect='equal', interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.6)
                cbar.set_label('Rank (1=Best)', rotation=270, labelpad=15)
        else:
            # If all values are NaN, show empty matrix
            ax.imshow(np.ones((n_nodes, n_nodes)), cmap='gray', alpha=0.2)
        
        # Add text annotations for rankings
        for i in range(n_nodes):
            for j in range(n_nodes):
                if not np.isnan(matrix[i, j]):
                    rank = int(matrix[i, j])
                    # Find if this edge is required
                    is_required = any(c['is_required'] for c in candidates 
                                    if tuple(sorted(c['edge'])) == tuple(sorted([i, j])))
                    
                    color = 'red' if is_required else 'black'
                    weight = 'bold' if is_required else 'normal'
                    ax.text(j, i, str(rank), ha='center', va='center', 
                           color=color, fontweight=weight, fontsize=12)
        
        # Highlight current node row and column
        ax.axhline(y=current_node, color='blue', linewidth=3, alpha=0.5)
        ax.axvline(x=current_node, color='blue', linewidth=3, alpha=0.5)
        
        # Set ticks and labels
        ax.set_xticks(range(n_nodes))
        ax.set_yticks(range(n_nodes))
        ax.set_xticklabels(range(n_nodes))
        ax.set_yticklabels(range(n_nodes))
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, n_nodes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_nodes, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.3)
        
        ax.set_xlabel('To Node', fontweight='bold')
        ax.set_ylabel('From Node', fontweight='bold')
        ax.set_title(f'Edge Rankings from Node {current_node}\n(Red = Required Edge)', 
                    fontweight='bold')
        
        # Add candidates info as text
        if candidates:
            info_text = "Top 3 Candidates:\n"
            for i, candidate in enumerate(candidates[:3]):
                edge = candidate['edge']
                score = candidate['score']
                req_mark = "★" if candidate['is_required'] else ""
                info_text += f"{i+1}. {edge}: {score:.3f} {req_mark}\n"
            
            ax.text(1.05, 0.95, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def find_route_with_visualization(self, required_edges, verbose=False):
        """Find route with step-by-step visualization"""
        current_route = [self.start_depot]
        current_length = 0
        required_covered = set()
        visited_edges = set()
        
        max_iterations = len(self.graph.edges()) * 10
        iteration_count = 0
        step_num = 0
        
        print("Starting visualization. Close each figure to proceed to the next step.")
        print("Press any key in the figure window to continue.")
        
        while len(required_covered) < len(required_edges) and iteration_count < max_iterations:
            current_node = current_route[-1]
            candidates = []
            iteration_count += 1
            step_num += 1
            
            # Get all possible next edges
            for neighbor in self.graph.neighbors(current_node):
                edge = (current_node, neighbor)
                edge_sorted = tuple(sorted(edge))
                edge_weight = self.graph[current_node][neighbor]['weight']
                
                # Capacity check
                try:
                    if neighbor == self.end_depot:
                        path_to_end_length = 0
                    else:
                        path_to_end_length = nx.shortest_path_length(
                            self.graph, neighbor, self.end_depot, weight='weight'
                        )
                    
                    if current_length + edge_weight + path_to_end_length > self.capacity:
                        continue
                except nx.NetworkXNoPath:
                    continue
                
                is_new_required = (edge_sorted in [tuple(sorted(req)) for req in required_edges] 
                                and edge_sorted not in required_covered)
                
                score_data = self.calculate_edge_score(edge, required_edges, current_length, is_new_required)
                
                candidates.append({
                    'edge': edge,
                    'neighbor': neighbor,
                    'is_new_required': is_new_required,
                    'score_data': score_data
                })
            
            # Create ranking matrix
            matrix, candidate_list = self.create_edge_ranking_matrix(
                current_node, required_edges, current_length, visited_edges
            )
            
            candidates_info = (matrix, candidate_list) if candidate_list else None
            
            # Visualize current state (before making the move)
            self.visualize_step(step_num, current_route, current_length, required_edges,
                              visited_edges, required_covered, candidates_info)
            
            # Make the move
            if not candidates:
                # Force movement toward uncovered required edges or end depot
                if verbose:
                    print(f"No direct candidates - seeking uncovered required edges...")
                
                uncovered_required = [req for req in required_edges 
                                    if tuple(sorted(req)) not in required_covered]
                
                if uncovered_required and current_node != self.end_depot:
                    # Try to reach uncovered required edges
                    best_path = None
                    best_length = float('inf')
                    
                    for req_edge in uncovered_required:
                        for target_node in req_edge:
                            try:
                                path = nx.shortest_path(self.graph, current_node, target_node, weight='weight')
                                path_length = nx.shortest_path_length(self.graph, current_node, target_node, weight='weight')
                                
                                final_path_length = nx.shortest_path_length(self.graph, target_node, self.end_depot, weight='weight')
                                total_length = current_length + path_length + final_path_length
                                
                                if total_length <= self.capacity and path_length < best_length:
                                    best_path = path[1:]
                                    best_length = path_length
                            except nx.NetworkXNoPath:
                                continue
                    
                    if best_path:
                        for next_node in best_path:
                            edge_weight = self.graph[current_route[-1]][next_node]['weight']
                            selected_edge = (current_route[-1], next_node)
                            
                            # Visualize the forced move
                            step_num += 1
                            self.visualize_step(step_num, current_route, current_length, 
                                              required_edges, visited_edges, required_covered,
                                              None, selected_edge)
                            
                            current_route.append(next_node)
                            current_length += edge_weight
                            visited_edges.add(tuple(sorted(selected_edge)))
                        continue
                
                # Go to end depot
                if current_node != self.end_depot:
                    try:
                        path_to_end = nx.shortest_path(self.graph, current_node, self.end_depot, weight='weight')
                        additional_length = nx.shortest_path_length(self.graph, current_node, self.end_depot, weight='weight')
                        if current_length + additional_length <= self.capacity:
                            # Visualize path to end
                            for i in range(1, len(path_to_end)):
                                step_num += 1
                                selected_edge = (path_to_end[i-1], path_to_end[i])
                                self.visualize_step(step_num, current_route, current_length,
                                                  required_edges, visited_edges, required_covered,
                                                  None, selected_edge)
                                current_route.append(path_to_end[i])
                                current_length += self.graph[path_to_end[i-1]][path_to_end[i]]['weight']
                    except nx.NetworkXNoPath:
                        pass
                break
            
            # Sort and select best candidate
            candidates.sort(key=lambda x: (
                not x['is_new_required'],
                -x['score_data']['final_score']
            ))
            
            best = candidates[0]
            selected_edge = best['edge']
            
            # Visualize the selected move
            matrix, candidate_list = self.create_edge_ranking_matrix(
                current_node, required_edges, current_length, visited_edges
            )
            candidates_info = (matrix, candidate_list) if candidate_list else None
            
            step_num += 1
            self.visualize_step(step_num, current_route, current_length, required_edges,
                              visited_edges, required_covered, candidates_info, selected_edge)
            
            # Execute the move
            current_route.append(best['neighbor'])
            current_length += best['score_data']['edge_weight']
            visited_edges.add(tuple(sorted(selected_edge)))
            
            if best['is_new_required']:
                required_covered.add(tuple(sorted(selected_edge)))
                if verbose:
                    print(f"✓ Covered required edge: {selected_edge}")
        
        # Ensure we end at the end depot
        if current_route[-1] != self.end_depot:
            try:
                path_to_end = nx.shortest_path(self.graph, current_route[-1], self.end_depot, weight='weight')
                additional_length = nx.shortest_path_length(self.graph, current_route[-1], self.end_depot, weight='weight')
                
                if current_length + additional_length <= self.capacity:
                    for i in range(1, len(path_to_end)):
                        step_num += 1
                        selected_edge = (path_to_end[i-1], path_to_end[i])
                        self.visualize_step(step_num, current_route, current_length,
                                          required_edges, visited_edges, required_covered,
                                          None, selected_edge)
                        current_route.append(path_to_end[i])
                        current_length += self.graph[path_to_end[i-1]][path_to_end[i]]['weight']
            except nx.NetworkXNoPath:
                pass
        
        # Final visualization
        step_num += 1
        self.visualize_step(step_num, current_route, current_length, required_edges,
                          visited_edges, required_covered, None)
        
        print(f"\nVisualization complete!")
        print(f"Final route: {current_route}")
        print(f"Final length: {current_length:.2f}")
        print(f"Required edges covered: {len(required_covered)}/{len(required_edges)}")
        
        return current_route, current_length, len(required_covered)


def run_visualization_demo():
    """Run the visualization demo with the simple graph"""
    from magnetic_field import SIMPLE_GRAPH, START_DEPOT, END_DEPOT, REQUIRED_EDGES, FAILED_EDGES, MAX_VEHICLE_CAPACITY
    
    print("Magnetic Field Route Visualization Demo")
    print("=" * 50)
    
    # Create visualized router
    capacity = MAX_VEHICLE_CAPACITY  # Adjust as needed
    router = VisualizedMagneticFieldRouter(
        SIMPLE_GRAPH, START_DEPOT, END_DEPOT, capacity, alpha=1.0, gamma=1.0
    )
    
    print(f"Graph: {len(SIMPLE_GRAPH.nodes())} nodes, {len(SIMPLE_GRAPH.edges())} edges")
    print(f"Required edges: {REQUIRED_EDGES + FAILED_EDGES}")
    print(f"Vehicle capacity: {capacity}")
    print(f"Alpha: {router.alpha}, Gamma: {router.gamma}")
    print("\nStarting step-by-step visualization...")
    print("Close each figure window to proceed to the next step.")
    
    # Run with visualization
    route, cost, covered = router.find_route_with_visualization(
        REQUIRED_EDGES + FAILED_EDGES, verbose=True
    )
    
    return router, route, cost, covered


if __name__ == "__main__":
    router, route, cost, covered = run_visualization_demo()