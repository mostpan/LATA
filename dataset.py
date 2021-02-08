import networkx as nx
import random

def load_data(file_name,fraction):
    g_t = nx.Graph()

    with open('data/'+file_name+'.txt') as f:
        nodes,edges=set(),[]
        for line in f.readlines():
            strs=line.strip().split(" ")
            u,v=int(strs[0]),int(strs[1])
            edges.append((u,v))
            nodes.add(u)
            nodes.add(v)
        nodes=list(nodes)
        nodes.sort()
        id_map=dict(zip(nodes,range(1,1+len(nodes))))

        for u in nodes:
            g_t.add_node(id_map[u])

        for u,v in edges:
            if u!=v:
                g_t.add_edge(id_map[u],id_map[v],label=1)

    n_node, n_edge = len(g_t.nodes()), len(g_t.edges())


    n_add_edge=int(n_edge*fraction)
    g_o=g_t.copy()
    for i in range(n_add_edge):
        u,v=random.randint(1,n_node),random.randint(1,n_node)
        while(u==v or g_o.has_edge(u,v)):
            u, v = random.randint(1, n_node), random.randint(1, n_node)
        g_o.add_edge(u, v,label=0)

    return g_o
