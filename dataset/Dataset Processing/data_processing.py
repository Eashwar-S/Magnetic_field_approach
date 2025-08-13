#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random as rd


# In[11]:


def readFile(folderPath):    
    with open(folderPath, 'r') as f:
        fileContents = f.readlines()
    return fileContents


# In[12]:


def fillInfoFromContents(fileContents, info):
    for i, content in enumerate(fileContents):
        if i == 0:
            info['Instance Name'].append(content.split()[2])
        elif i == 1:
            info['Number of Nodes'].append([int(word) for word in content.split() if word.isdigit()][0])
        elif i == 2:
            info['Required Edges'].append([int(word) for word in content.split() if word.isdigit()][0])
        elif i == 3:
            c = [int(word) for word in content.split() if word.isdigit()][0]
        elif i == 6:
            info['Capacity'].append([int(word) for word in content.split() if word.isdigit()][0])
        elif i == 9:
            info['Depot Nodes'].append([int(word) for word in content.split() if word.isdigit()])
            info['Number of Depot Nodes'].append(len(info['Depot Nodes'][-1]))
        
    info['Number of Edges'].append(c + info['Required Edges'][-1])


# In[13]:


def readAndStoreInstanceInfo(folderPath):
    info = {'Instance Name' : [],
            'Number of Nodes' : [],
            'Number of Edges' : [],
            'Required Edges' : [],
            'Capacity' : [],
            'Number of Depot Nodes' : [],
            'Depot Nodes' : []}
    for i, file in enumerate(os.listdir(folderPath)):
        if file.endswith(".txt"):
            file_path = f"{folderPath}/{file}"
            fileContents = readFile(file_path)
            fillInfoFromContents(fileContents, info)

    df = pd.DataFrame(data=info,columns=['Instance Name','Number of Nodes', 'Number of Edges',
                                     'Required Edges', 'Capacity', 'Number of Depot Nodes', 'Depot Nodes'])
    print(df.columns)
    df.to_csv("DeArmon_dataset_info.csv")
    df.sort_values(by='Number of Edges', ascending=False)
    return info


# In[14]:


def createGraphfromFile(file, info, index):
    fileContents = readFile(file)
    s = ["LIST_REQ_EDGES :\n", "LIST_NOREQ_EDGES :\n"]
    startProcessing = False
    startNode = []
    endNode = []
    edgeWeight = []
    i = 0
    for contents in fileContents:
        if contents == s[i] and startProcessing:
            startProcessing = False
            break

        if startProcessing:
            startNode.append([int(letters) for word in contents.split() for letters in word.split(",") if letters.isdigit()][0])
            endNode.append([int(letters) for word in contents.split() for letters in word.split(",") if letters.isdigit()][1])
            edgeWeight.append([int(letters) for word in contents.split() for letters in word.split(",") if letters.isdigit()][2])

        if contents == s[i]:
            startProcessing = True
            i += 1
    requiredEdges = []
    for i in range(info['Required Edges'][index]):
        requiredEdges.append([startNode[i], endNode[i]])
        
    return startNode, endNode, edgeWeight


# In[15]:


def plotGraph(depotNodes ,requiredEdges, numNodes, s, t, weights, show=True):
    G = nx.Graph()
    edges = []    
    for i in range(len(s)):
        edges.append((s[i], t[i], weights[i]))
    
    for i in range(1, numNodes+1):
        G.add_node(i)
    pos = nx.spring_layout(G)
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
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if show:
        plt.figure(1)
        plt.show()
    return G,pos, node_color, depot_node_color, edges


# In[16]:


def creatingIcyRoadInstance(file, info, index, startNode, endNode, edgeWeight):
    newDepotNodes = []
    requiredEdgeIndexes = []
    newRequiredEdges = []
    count = 0
    while count <= (info['Number of Nodes'][index]//5):
        node = rd.randint(1, info['Number of Nodes'][index])
        if node not in newDepotNodes:
            newDepotNodes.append(node)
            count += 1
    count = 0
    while count <= (info['Number of Edges'][index]//3):
        edge = rd.randint(0, info['Number of Edges'][index])
        if edge not in requiredEdgeIndexes:
            requiredEdgeIndexes.append(edge)
            count += 1
    for i in range(info['Number of Edges'][index]):
        if i in requiredEdgeIndexes:
            newRequiredEdges.append([startNode[i], endNode[i]])
    G,pos, node_color, depot_node_color, edges = plotGraph(newDepotNodes, newRequiredEdges, info['Number of Nodes'][index], startNode, endNode, edgeWeight)
    #     plt.savefig('../IcyRoad Instances from DeArmon\icy_road_' + info['Instance Name'][index] + '.png')
    #     plt.show()
    return G,pos, node_color, depot_node_color, edges, newDepotNodes, newRequiredEdges, 2*max(edgeWeight), G.number_of_nodes()


# In[25]:


# def createGraph(inputType = 'txt'):
# #     folderPath = '../CARP_datasets/DeArmon_gdb-IF'
# #     for i, file in enumerate(os.listdir(folderPath)):
# #         if file.endswith(".txt"):
# #             file_path = f"{folderPath}/{file}"
#     file_path = '../CARP_datasets/DeArmon_gdb-IF/gdb-IF-01.txt'
#     info = readAndStoreInstanceInfo('../../../CARP_datasets/DeArmon_gdb-IF')
#     startNode, endNode, edgeWeight = createGraphfromFile(file_path, info, 0)
#     G, depotNodes, requiredNodes, vehicleCapacity, numNodes = creatingIcyRoadInstance(file_path, info, 0, startNode, endNode, edgeWeight)
#     return G, depotNodes, requiredNodes, vehicleCapacity, numNodes


# In[ ]:




