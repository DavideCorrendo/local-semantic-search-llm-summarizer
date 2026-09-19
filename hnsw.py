import math 
import heapq
import random

def euclidean_distance(v1, v2):
     return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

import heapq

def prune(nodes, vectors, anchor_id, M_max):
    nodes_dist = [(euclidean_distance(vectors[anchor_id], vectors[nid]), nid) for nid in nodes]
    nodes_dist.sort() 

    return set(nid for _, nid in nodes_dist[:M_max])

def phase2(vectors, graph, entry_point, q, ef_construction, layer, node_id, M, M_max):
    nearest = search_layer(vectors, graph, entry_point, q, ef_construction, layer)
    new_entry_point = nearest[0]
            
    selected_neighbors = nearest[:M]
            
    graph[node_id][layer] = set(selected_neighbors)
    
    for n in selected_neighbors:
        nodes = graph[n].get(layer, set())
        nodes.add(node_id) 
        if len(nodes) > M_max:
            nodes = prune(nodes, vectors, n, M_max)
        graph[n][layer] = nodes

    return new_entry_point

def search_layer(vectors, graph, entry_point, q, ef, layer):

    visited = set([entry_point])
    
    queue = [(euclidean_distance(vectors[entry_point], q), entry_point)]
    
    results = list(queue)

    while queue:
        dist, current = heapq.heappop(queue)

        if len(results) >= ef and dist > results[-1][0]:
            break

        neighbours = graph[current].get(layer, set())
        for n in neighbours:
            if n not in visited:
                visited.add(n)
                d = euclidean_distance(vectors[n], q)
                
                if len(results) < ef or d < results[-1][0]:
                    heapq.heappush(queue, (d, n))
                    
                    results.append((d, n))
                    results.sort()
                    if len(results) > ef:
                        results.pop()

    return [idx for _, idx in results]


def insert(q, vectors, graph, node_levels, state, M = 16, M_max = 16, M_max0= 32, ef_construction=200, m_L=None):

    """
    q               : the new vector being inserted
    vectors         : list of all vectors, index = node id
    graph           : list where graph[node_id] = {layer: set(neighbor_ids)}
    node_levels     : list where node_levels[node_id] = top layer that node reaches
    state           : dict holding {"entry_point": id_or_None, "max_level": int}
    """

    node_id = len(vectors)
    vectors.append(q)
    graph.append({})

    if m_L is None:
        m_L = 1 / math.log(M)
    tl = math.floor(-math.log(random.uniform(0, 1)) * m_L)
    node_levels.append(tl)

    if (state["entry_point"] == None):
        state["entry_point"] = node_id
        state["max_level"] = tl
        graph[node_id] = {layer: set() for layer in range(tl+1)}
        return node_id

    entry_point = state["entry_point"]
    L = state["max_level"]

    # phase 1
    for layer in range(L,tl,-1):
        nearest = search_layer(vectors, graph, entry_point, q, 1, layer)
        entry_point = nearest[0]

    # phase 2
    # search must start from the layer where there are other nodes
    for layer in range(min(L, tl), 0, -1):
        entry_point = phase2(vectors, graph, entry_point, q, ef_construction, layer, node_id, M, M_max)

    phase2(vectors, graph, entry_point, q, ef_construction, 0, node_id, M, M_max0)

    if tl > L:
        state["entry_point"] = node_id
        state["max_level"] = tl
        

