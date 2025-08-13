#!/usr/bin/env python
# coding: utf-8

# In[34]:


import osmnx as ox
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import tqdm
import pandas as pd
import os


# In[18]:


def graphFromMap(aoI, north, east, south, west, roadType):
    state = ox.geocode_to_gdf(aoI) 
#     ox.plot_footprints(ox.project_gdf(state))
#     north, east, south, west = 45.785, -110.953, 45.558, -111.255 
    # Downloading the map as a graph object 
    G = ox.graph_from_bbox(north, south, east, west, retain_all=False, custom_filter=roadType)
    return G


# In[22]:


def createAndSaveOriginalGraph():
    aoI = 'Montana, US'
    north, east, south, west = 45.785, -110.953, 45.558, -111.255
    roadType = ['["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]',
                '["highway"~"residential|tertiary_link|service"]',
                '["highway"~"primary|secindary|service"]',
                '["highway"~"primary|tertiary|tertiary_link|service"]',
                '["highway"~"residential"]',
                '["highway"~"primary|secondary|tertiary|motorway"]',
                '["highway"~"tertiary"]',
                '["highway"~"service"]',
                '["highway"~"primary|secondary|tertiary|living_street"]',
                '["highway"~"primary|tertiary|tertiary_link"]',
                '["highway"~"secondary"]']
    savePath = '../../dataset/osmnx graph files/original graph/'
    for i in tqdm.tqdm(range(len(roadType))):
        
        graph = graphFromMap(aoI, north, east, south, west, roadType[i])
        nx.write_gpickle(graph, savePath + str(i) + '.pkl')


# In[31]:


# createAndSaveOriginalGraph()


# In[3]:


def visualizeGraph(depotNodes ,requiredEdges, pos, numNodes, s, t, weights, show=True):
    G = nx.Graph()
    edges = []
    
    for i in range(len(s)):
        edges.append((s[i], t[i], weights[i]))
    
    for i in range(1, numNodes+1):
        G.add_node(i)

    node_color = ['y']*int(G.number_of_nodes())
    depot_node_color = node_color
    for i in range(1, len(node_color)+1):
        if i in depotNodes:
            depot_node_color[i-1] = 'g'
            
    G.add_weighted_edges_from(edges)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx(G,pos, node_color = node_color)
    nx.draw_networkx(G,pos, node_color = depot_node_color)
    nx.draw_networkx_edges(G, pos, edgelist=requiredEdges, width=3, alpha=0.5,
                                        edge_color="r")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if show:
        plt.figure(1)
#         plt.show()
    return G,pos, node_color, depot_node_color, edges


# In[26]:


def creatingIcyRoadInstanceFromOsmnx(graph):
    dic = {}
    startNode = []
    endNode = []
    edgeWeight = []
    depotNodes = []
    
    requiredEdges = []
    pos = {}
    print('abc')
    
    for i in tqdm.tqdm(range(len(list(graph.nodes(data=True)))), position=0):
      dic[list(graph.nodes(data=True))[i][0]] = i+1
      pos[i+1] = np.array([list(graph.nodes(data=True))[i][1]['x'], list(graph.nodes(data=True))[i][1]['y']])

    
    for i in tqdm.tqdm(range(len(list(graph.edges(data=True)))), position=0):
      startNode.append(dic[list(graph.edges(data=True))[i][0]])
      endNode.append(dic[list(graph.edges(data=True))[i][1]])
      edgeWeight.append(list(graph.edges(data=True))[i][2]['length'])

    newDepotNodes = []
    requiredEdgeIndexes = []
    newRequiredEdges = []
    newNonRequiredEdges = []
    depotNodesLat = []
    depotNodesLong = []
    normalNodesLat = []
    normalNodesLong = []
    requiredEdges = []
    nonrequiredEdges = []

    count = 0
    while count <= (len(list(graph.nodes(data=True)))//5):
        node = rd.randint(1, len(list(graph.nodes(data=True))))
        if node not in newDepotNodes:
            newDepotNodes.append(node)
            count += 1
    print('abcd')
    count = 0
    while count <= (len(list(graph.edges(data=True)))//3):
        edge = rd.randint(0, len(list(graph.edges(data=True)))-1)
        if edge not in requiredEdgeIndexes:
            requiredEdgeIndexes.append(edge)
            count += 1
    print('abcd1')
    for i in tqdm.tqdm(range(len(list(graph.edges(data=True)))), position=0):
        if i in requiredEdgeIndexes:
            newRequiredEdges.append([startNode[i], endNode[i]])
        else:
            newNonRequiredEdges.append([startNode[i], endNode[i]])

    print('abcd2')
    for i in tqdm.tqdm(range(len(newDepotNodes)), position=0):
        for node in list(graph.nodes(data=True)):
            if node[0] == list(dic.keys())[list(dic.values()).index(newDepotNodes[i])]:
                depotNodesLat.append(node[1]['y'])
                depotNodesLong.append(node[1]['x'])
    
    for key, val in dic.items():
        if val not in newDepotNodes:
            for node in list(graph.nodes(data=True)):
                if node[0] == key:
                    normalNodesLat.append(node[1]['y'])
                    normalNodesLong.append(node[1]['x'])
    print('abc1')
    for u, v in newRequiredEdges:
        requiredEdges.append([list(dic.keys())[list(dic.values()).index(u)], list(dic.keys())[list(dic.values()).index(v)]])


    for u, v in newNonRequiredEdges:
        nonrequiredEdges.append([list(dic.keys())[list(dic.values()).index(u)], list(dic.keys())[list(dic.values()).index(v)]])
    print('abc2')
    G,pos, node_color, depot_node_color, edges = visualizeGraph(newDepotNodes , newRequiredEdges, pos, graph.number_of_nodes(), startNode, endNode, edgeWeight)
#     plt.show()
    print('abc3')
    depotNodesLatLong = [depotNodesLat, depotNodesLong]
    normalNodesLatLong = [normalNodesLat, normalNodesLong]
    
    return G, pos, node_color, depot_node_color, edges, newDepotNodes, newRequiredEdges, 2*max(edgeWeight),\
G.number_of_nodes(), depotNodesLatLong, normalNodesLatLong, requiredEdges, nonrequiredEdges, dic


# In[45]:


# for edge in newRequiredEdges:
#     if edge in newNonRequiredEdges:
#         print('Eashwar')


# In[10]:


# print(rd.randint(0,2))


# # New Section

# In[32]:


# print(list(G.nodes(data=True))[0])
# print(list(G.edges(data=True))[0])


# In[33]:


# print(list(G.edges(data=True))[0])


# In[13]:


# Displaying the shape of edge using the geometry
# list(G.edges(data=True))[5][2]['geometry']


# In[14]:


def getLatLongofRoutes(G, routes):
    latLong = []
    for node_list in routes:
        edge_nodes = list(zip(node_list[:-1], node_list[1:]))
        lines = []
        for u, v in edge_nodes:
            # if there are parallel edges, select the shortest in length
            if G.get_edge_data(u, v) is not None:
                data = min(G.get_edge_data(u, v).values(), 
                           key=lambda x: x['length'])
                # if it has a geometry attribute
                if 'geometry' in data:
                    # add them to the list of lines to plot
                    xs, ys = data['geometry'].xy
                    lines.append(list(zip(xs, ys)))
                else:
                    # if it doesn't have a geometry attribute,
                    # then the edge is a straight line from node to node
                    x1 = G.nodes[u]['x']
                    y1 = G.nodes[u]['y']
                    x2 = G.nodes[v]['x']
                    y2 = G.nodes[v]['y']
                    line = [(x1, y1), (x2, y2)]
                    lines.append(line)

        long2 = []
        lat2 = []
        for i in range(len(lines)):
            z = list(lines[i])
            l1 = list(list(zip(*z))[0])
            l2 = list(list(zip(*z))[1])
            for j in range(len(l1)):
                long2.append(l1[j])
                lat2.append(l2[j])
    
        latLong.append([lat2, long2])
    return latLong


# In[68]:


def latLongofEdges(G, requiredEdges):
    lines = []
    for u, v in requiredEdges:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), 
                   key=lambda x: x['length'])
        # if it has a geometry attribute
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute,
            # then the edge is a straight line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
        
    longi = []
    lati = []
    for i in range(len(lines)):
        long2 = []
        lat2 = []
        z = list(lines[i])
        l1 = list(list(zip(*z))[0])
        l2 = list(list(zip(*z))[1])
        for j in range(len(l1)):
            long2.append(l1[j])
            lat2.append(l2[j])
        longi.append(long2)
        lati.append(lat2)
    
    return longi, lati

        


# In[69]:


# getting coordinates of the nodes
# we will store the longitudes and latitudes in following list 
# long = [] 
# lat = []  
# for i in route:
#      point = G.nodes[i]
#      long.append(point['x'])
#      lat.append(point['y'])
# print([lat[0], lat[1]])
# print(long)


# In[136]:


def plot_graph(desiredEdgeLatLong, latLongNonreqEdges, depotNodesLatLong, normalNodesLatLong, routeLatLong=None): 
    longi = []
    lati = []
    depotNodesLat, depotNodesLong = depotNodesLatLong[0], depotNodesLatLong[1]
    normalNodesLat, normalNodesLong = normalNodesLatLong[0], normalNodesLatLong[1]
    latitu, longitu = desiredEdgeLatLong[0], desiredEdgeLatLong[1]
    
#     colors = ['red', 'pink']
    fig = go.Figure(go.Scattermapbox(
        name = "depots",
        mode = "markers",
        lon = depotNodesLong,
        lat = depotNodesLat,
        marker = {'size': 10, 'color':"green"}))
    
    fig.add_trace(go.Scattermapbox(
        name = "Normal Nodes",
        mode = "markers",
        lon = normalNodesLong,
        lat = normalNodesLat,
        marker = {'size': 8, 'color':"yellowgreen"}))
    
#     if routeLatLong is not None:
#         for latit, longit in routeLatLong:
#             fig.add_trace(go.Scattermapbox(
#             name = "Paths",
#             mode = "lines",
#             lon = longit,
#             lat = latit,
#             marker = {'size': 10},
#             line = dict(width = 4, color = 'blue')))
    
    for i in range(len(latitu)): 
        longi += longitu[i]
        lati += latitu[i]
        fig.add_trace(go.Scattermapbox(
            name = "Desired Edge",
            mode = "lines",
            lon = longitu[i],
            lat = latitu[i],
            marker = {'size': 10},
            line = dict(width = 3, color = 'red')))
    
    nlat, nlong = latLongNonreqEdges[0], latLongNonreqEdges[1]
    
    for i in range(len(nlat)): 
#         longi += nlong[i]
#         lati += nlat[i]
        fig.add_trace(go.Scattermapbox(
            name = "Undesired Edge",
            mode = "lines",
            lon = nlong[i],
            lat = nlat[i],
            marker = {'size': 10},
            line = dict(width = 1, color = 'black')))
    
   
    if routeLatLong is not None:
        for latit, longit in routeLatLong:
            fig.add_trace(go.Scattermapbox(
            name = "Paths",
            mode = "lines",
            lon = longit,
            lat = latit,
            marker = {'size': 10},
            line = dict(width = 4, color = 'blue')))
#     fig = go.Figure(go.Scattermapbox(
#         name = "Desired Edge 1",
#         mode = "lines",
#         lon = long[1],
#         lat = lat[1],
#         marker = {'size': 10},
#         line = dict(width = 4.5, color = 'blue')))
    
#     fig = go.Figure(go.Scattermapbox(
#         name = "Path1",
#         mode = "lines",
#         lon = [long[3], long[4]],
#         lat = [lat[3], lat[4]],
#         marker = {'size': 10},
#         line = dict(width = 4.5, color = 'blue')))

#     fig.add_annotation(
#               x=lat[1],  # arrows' head
#               y=long[1],  # arrows' head
#               ax=lat[0],
#               ay=long[0],
#               xref='x',
#               yref='y',
#               axref='x',
#               ayref='y',
#               text='',  # if you want only the arrow
#               showarrow=True,
#               arrowhead=3,
#               arrowsize=1,
#               arrowwidth=1,
#               arrowcolor='black'
#             )

    
   
     
#     adding destination marker
#     fig.add_trace(go.Scattermapbox(
#         name = "Destination",
#         mode = "markers",
#         lon = [destination_point[1]],
#         lat = [destination_point[0]],
#         marker = {'size': 12, 'color':'green'}))
    
    # getting center for plots:
    lat_center = np.mean(lati)
    long_center = np.mean(longi)
    # defining the layout using mapbox_style
    fig.update_layout(mapbox_style="open-street-map",
        mapbox_center_lat = 30, mapbox_center_lon=-80)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                      mapbox = {
                          'center': {'lat': lat_center, 
                          'lon': long_center},
                          'zoom': 10})
#     fig.write_image("fig1.png")
    fig.show()


# In[28]:


# define origin and desination locations  
# origin_point = (depotNodesLat[0], depotNodesLong[0]) 
# destination_point = (depotNodesLat[2], depotNodesLong[2])

# # origin_point1 = (list(G.nodes(data=True))[5][1]['y'], list(G.nodes(data=True))[5][1]['x']) 
# # destination_point1 = (list(G.nodes(data=True))[10][1]['y'], list(G.nodes(data=True))[10][1]['x'])

# origin_point1 = (depotNodesLat[5], depotNodesLong[5]) 
# destination_point1 = (depotNodesLat[7], depotNodesLong[7])


# # get the nearest nodes to the locations 
# # origin_node = ox.get_nearest_node(G, origin_point) 
# # destination_node = ox.get_nearest_node(G, destination_point)
# origin_node = ox.distance.nearest_nodes(G, origin_point[1], origin_point[0]) 
# destination_node = ox.distance.nearest_nodes(G, destination_point[1], destination_point[0])
# # printing the closest node id to origin and destination points 

# origin_node1 = ox.distance.nearest_nodes(G, origin_point1[1], origin_point1[0]) 
# destination_node1 = ox.distance.nearest_nodes(G, destination_point1[1], destination_point1[0])


# In[29]:


# Finding the optimal path 
# route1 = nx.shortest_path(G, origin_node, destination_node, weight = 'length')

# route2 = nx.shortest_path(G, origin_node1, destination_node1, weight = 'length')

# route = [route1, route2]


# In[30]:


# reqlongi, reqlati = latLongofEdges(G, requiredEdges)
# nreqlongi, nreqlati = latLongofEdges(G, nonrequiredEdges)
# routeLatLong = getLatLongofRoutes(G, route)


# In[144]:


# plot_graph([reqlati reqlongi], [nreqlati, nreqlongi], [depotNodesLat, depotNodesLong], [normalNodesLat, normalNodesLong], routeLatLong)


# In[30]:


# # Getting the start and end node of this part 
# start_node=route[-7] 
# end_node=route[-6]
# # Getting the edge connecting these nodes and storing it as a list in z to maintain the data structure of G.edges 
# z = []  
# for i in list(G.edges(data=True)):
#      if (i[0]==start_node) & (i[1]==end_node):
#          z.append(i)
 
# z[0][2]['geometry']


# In[27]:


# plot_path(lat2, long2, origin_point, destination_point)


# In[13]:


# def plot_path(lat, long, origin_point, destination_point):
    
#     """
#     Given a list of latitudes and longitudes, origin 
#     and destination point, plots a path on a map
    
#     Parameters
#     ----------
#     lat, long: list of latitudes and longitudes
#     origin_point, destination_point: co-ordinates of origin
#     and destination
#     Returns
#     -------
#     Nothing. Only shows the map.
#     """
#     # adding the lines joining the nodes
#     fig = go.Figure(go.Scattermapbox(
#         name = "Path",
#         mode = "lines",
#         lon = long,
#         lat = lat,
#         marker = {'size': 10},
#         line = dict(width = 4.5, color = 'blue')))
#     # adding source marker
#     fig.add_trace(go.Scattermapbox(
#         name = "Source",
#         mode = "markers",
#         lon = [origin_point[1]],
#         lat = [origin_point[0]],
#         marker = {'size': 12, 'color':"red"}))
     
#     # adding destination marker
#     fig.add_trace(go.Scattermapbox(
#         name = "Destination",
#         mode = "markers",
#         lon = [destination_point[1]],
#         lat = [destination_point[0]],
#         marker = {'size': 12, 'color':'green'}))
    
#     # getting center for plots:
#     lat_center = np.mean(lat)
#     long_center = np.mean(long)
#     # defining the layout using mapbox_style
#     fig.update_layout(mapbox_style="stamen-terrain",
#         mapbox_center_lat = 30, mapbox_center_lon=-80)
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
#                       mapbox = {
#                           'center': {'lat': lat_center, 
#                           'lon': long_center},
#                           'zoom': 13})
#     fig.show()

