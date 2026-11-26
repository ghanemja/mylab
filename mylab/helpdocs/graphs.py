"""Help text for the functions in mylab.builtins.graphs."""

HELP = {
    "graph": (["G = graph", "G = graph(A)", "G = graph(A, nodenames)", "G = graph(s, t)",
               "G = graph(s, t, weights)", "G = graph(s, t, weights, nodenames | num | NodeTable)",
               "G = graph(EdgeTable)", "G = graph(EdgeTable, NodeTable)", "G = graph(..., 'omitselfloops')",
               "G = graph(A, ..., 'upper' | 'lower')"],
              "Undirected graph. Build it from a symmetric adjacency matrix A (nonzeros are edges and "
              "their weights), from lists of end nodes s and t (numbers or names), or from tables. "
              "G.Edges is a table with EndNodes (and Weight); G.Nodes is a table with one row per "
              "node (variable Name when the nodes are named). Both tables can take extra variables."),
    "digraph": (["G = digraph", "G = digraph(A)", "G = digraph(A, nodenames)", "G = digraph(s, t)",
                 "G = digraph(s, t, weights)", "G = digraph(s, t, weights, nodenames | num | NodeTable)",
                 "G = digraph(EdgeTable)", "G = digraph(EdgeTable, NodeTable)", "G = digraph(..., 'omitselfloops')"],
                "Directed graph. A(i, j) nonzero is an edge from node i to node j; in the list form "
                "each edge goes from s(k) to t(k). See graph for the tables G.Edges and G.Nodes."),
    "numnodes": ("n = numnodes(G)", "Number of nodes of a graph or digraph."),
    "numedges": ("m = numedges(G)", "Number of edges of a graph or digraph."),
    "findnode": ("k = findnode(G, nodeID)",
                 "Numeric index of each node given by name or number; 0 for a node that is not in the graph."),
    "findedge": (["idx = findedge(G, s, t)", "[s, t] = findedge(G)", "[s, t] = findedge(G, idx)",
                  "[idx, m] = findedge(G, s, t)"],
                 "Edge index of the edge between s and t (0 if there is none), or the end nodes of the "
                 "edges with the given indices. With two outputs and node pairs it lists every "
                 "matching edge of a multigraph and the pair each one belongs to."),
    "edgecount": ("c = edgecount(G, s, t)", "Number of edges between the node pairs (s, t)."),
    "ismultigraph": ("tf = ismultigraph(G)", "True if some pair of nodes is joined by more than one edge."),
    "addedge": (["H = addedge(G, s, t)", "H = addedge(G, s, t, weights)", "H = addedge(G, EdgeTable)"],
                "Adds edges. Nodes that do not exist yet are created. A weighted graph needs a weight "
                "for each new edge."),
    "rmedge": (["H = rmedge(G, s, t)", "H = rmedge(G, idx)"],
               "Removes the edges between the node pairs (s, t), or the edges with the given indices."),
    "addnode": (["H = addnode(G, n)", "H = addnode(G, names)", "H = addnode(G, NodeTable)"],
                "Adds n isolated nodes, or nodes with the given names or table rows."),
    "rmnode": ("H = rmnode(G, nodeIDs)", "Removes nodes and every edge that touches them. The remaining "
                                          "nodes are renumbered."),
    "flipedge": (["H = flipedge(G)", "H = flipedge(G, s, t)", "H = flipedge(G, idx)"],
                 "Reverses the direction of all edges of a digraph, or of the chosen ones."),
    "subgraph": ("H = subgraph(G, nodeIDs)",
                 "The graph formed by the listed nodes (numbers, names or a logical mask) and the edges "
                 "between them."),
    "reordernodes": (["H = reordernodes(G, order)", "[H, idx] = reordernodes(G, order)"],
                     "Renumbers the nodes so that node order(k) becomes node k. idx maps the edges of "
                     "H to the edges of G."),
    "neighbors": ("N = neighbors(G, nodeID)", "Nodes joined to the given node by an edge (undirected graphs)."),
    "successors": ("N = successors(G, nodeID)", "Nodes reached by an edge leaving the given node of a digraph."),
    "predecessors": ("N = predecessors(G, nodeID)", "Nodes with an edge that enters the given node of a digraph."),
    "degree": (["d = degree(G)", "d = degree(G, nodeIDs)"],
               "Number of edges at each node of an undirected graph. A self-loop counts twice."),
    "indegree": (["d = indegree(G)", "d = indegree(G, nodeIDs)"], "Number of edges entering each node of a digraph."),
    "outdegree": (["d = outdegree(G)", "d = outdegree(G, nodeIDs)"], "Number of edges leaving each node of a digraph."),
    "outedges": (["eid = outedges(G, nodeID)", "[eid, nid] = outedges(G, nodeID)"],
                 "Indices of the edges leaving a node (all its edges in an undirected graph), and the "
                 "nodes at their other ends."),
    "inedges": (["eid = inedges(G, nodeID)", "[eid, nid] = inedges(G, nodeID)"],
                "Indices of the edges entering a node of a digraph, and the nodes they come from."),
    "adjacency": (["A = adjacency(G)", "A = adjacency(G, 'weighted')", "A = adjacency(G, weights)"],
                  "Sparse adjacency matrix. A(i, j) counts the edges from i to j; 'weighted' or a weight "
                  "vector puts edge weights there instead."),
    "incidence": ("I = incidence(G)",
                  "Sparse node-by-edge incidence matrix: column j has -1 at the source node and +1 at the "
                  "target node of edge j."),
    "shortestpath": (["P = shortestpath(G, s, t)", "[P, d, edgepath] = shortestpath(G, s, t)",
                      "... = shortestpath(G, s, t, 'Method', m)"],
                     "Shortest path between two nodes: the nodes on it, its length and its edges. d is "
                     "Inf and P is empty when t cannot be reached. Method is 'auto', 'unweighted', "
                     "'positive' (Dijkstra), 'mixed' (Bellman-Ford) or 'acyclic'."),
    "shortestpathtree": (["TR = shortestpathtree(G, s)", "TR = shortestpathtree(G, s, t)",
                          "[TR, D, E] = shortestpathtree(..., 'OutputForm', form, 'Method', m)"],
                         "Shortest paths from one node to all others (or to the nodes t; s may be 'all' "
                         "with a single t). OutputForm 'tree' gives a digraph of the tree edges, 'cell' "
                         "the paths as a cell array, 'vector' the predecessor of each node. D holds the "
                         "distances and E marks the tree edges."),
    "distances": (["d = distances(G)", "d = distances(G, s)", "d = distances(G, s, t)",
                   "... = distances(..., 'Method', m)"],
                  "Shortest-path distances between all pairs of nodes, or from the nodes s to the nodes t. "
                  "Unreachable pairs are Inf."),
    "nearest": (["ids = nearest(G, s, d)", "[ids, dist] = nearest(G, s, d, 'Direction', dir, 'Method', m)"],
                "Nodes within distance d of node s, nearest first, with their distances."),
    "conncomp": (["bins = conncomp(G)", "[bins, binsizes] = conncomp(G)",
                  "bins = conncomp(G, 'Type', 'weak' | 'strong', 'OutputForm', 'vector' | 'cell')"],
                 "Connected components: bins(k) is the component number of node k. For a digraph the "
                 "default is strongly connected components, numbered in topological order; 'weak' "
                 "ignores edge directions."),
    "bfsearch": (["v = bfsearch(G, s)", "T = bfsearch(G, s, events)", "... = bfsearch(..., 'Restart', tf)"],
                 "Breadth-first search from node s: the nodes in the order they are discovered. events "
                 "is 'discovernode', 'finishnode', 'startnode', 'edgetonew', 'edgetodiscovered', "
                 "'edgetofinished', 'allevents' or a cell array of several; several events give a table."),
    "dfsearch": (["v = dfsearch(G, s)", "T = dfsearch(G, s, events)", "... = dfsearch(..., 'Restart', tf)"],
                 "Depth-first search from node s. The events are the same as for bfsearch."),
    "minspantree": (["T = minspantree(G)", "[T, pred] = minspantree(G)",
                     "T = minspantree(G, 'Method', 'dense' | 'sparse', 'Root', r, 'Type', 'tree' | 'forest')"],
                    "Minimum spanning tree of an undirected graph (of the component of the root, or of "
                    "every component with Type 'forest'). pred holds the predecessor of each node."),
    "toposort": (["n = toposort(G)", "[n, H] = toposort(G)", "n = toposort(G, 'Order', 'fast' | 'stable')"],
                 "Topological order of a directed acyclic graph: every edge goes from an earlier to a "
                 "later node. 'stable' gives the smallest possible node numbers first. H is the graph "
                 "with its nodes in that order."),
    "isdag": ("tf = isdag(G)", "True if a digraph has no cycles."),
    "hascycles": ("tf = hascycles(G)", "True if the graph contains a cycle."),
    "maxflow": (["mf = maxflow(G, s, t)", "[mf, GF, cs, ct] = maxflow(G, s, t)", "... = maxflow(G, s, t, algorithm)"],
                "Maximum flow from s to t, with the edge weights as capacities (1 without weights). GF is "
                "a digraph of the edges that carry flow; cs and ct are the two sides of a minimum cut. "
                "algorithm is 'searchtrees', 'augmentpath' or 'pushrelabel'."),
    "centrality": (["c = centrality(G, type)", "c = centrality(G, type, 'Importance', w)",
                    "c = centrality(G, type, 'Cost', w)", "c = centrality(G, 'pagerank', 'FollowProbability', p)"],
                   "Importance of each node. For a graph: 'degree', 'closeness', 'betweenness', "
                   "'pagerank', 'eigenvector'. For a digraph: 'indegree', 'outdegree', 'incloseness', "
                   "'outcloseness', 'betweenness', 'pagerank', 'hubs', 'authorities'. Tolerance and "
                   "MaxIterations control the iterative types."),
    "transclosure": ("H = transclosure(G)",
                     "Transitive closure of a digraph: an edge from i to j whenever j can be reached from i."),
    "transreduction": ("H = transreduction(G)",
                       "Transitive reduction of a digraph: a graph with the fewest edges that has the same "
                       "reachability."),
    "condensation": ("C = condensation(G)",
                     "The digraph whose nodes are the strongly connected components of G (numbered as "
                     "by conncomp), with an edge where G has an edge between two components."),
    "allpaths": (["paths = allpaths(G, s, t)", "[paths, edgepaths] = allpaths(G, s, t)",
                  "... = allpaths(..., 'MaxNumPaths', n, 'MaxPathLength', L, 'MinPathLength', L)"],
                 "All paths from s to t that visit no node twice, as a cell array of node lists."),
    "dmperm": (["p = dmperm(A)", "[p, q, r, s, cc, rr] = dmperm(A)"],
               "Dulmage-Mendelsohn decomposition. With one output, p is a maximum matching: p(j) is the "
               "row matched to column j (0 if none). With more outputs A(p, q) is block upper "
               "triangular; r and s mark the block boundaries, cc and rr the coarse (under-, well- "
               "and over-determined) parts."),
    "etree": (["p = etree(A)", "p = etree(A, 'col' | 'row' | 'lo' | 'sym')", "[p, post] = etree(...)"],
              "Elimination tree of a symmetric sparse matrix (or of A'*A with 'col', A*A' with 'row'): "
              "p(k) is the parent of column k, 0 for a root. post is a postorder of the tree."),
    "colperm": ("p = colperm(A)", "Column permutation that orders the columns by increasing number of nonzeros."),
    "gplot": (["gplot(A, xy)", "gplot(A, xy, linespec)", "[x, y] = gplot(A, xy)"],
              "Draws the graph with adjacency matrix A and node coordinates xy (one row per node). "
              "With outputs it returns the line coordinates instead, separated by NaN."),
    "treelayout": (["[x, y] = treelayout(parent)", "[x, y, h, s] = treelayout(parent, post)"],
                   "Coordinates for drawing a tree given by its parent vector (0 marks a root). h is the "
                   "height of the tree and s the largest number of children."),
    "treeplot": (["treeplot(parent)", "treeplot(parent, nodespec, edgespec)"],
                 "Draws the tree given by a parent vector (0 marks a root)."),
    "etreeplot": (["etreeplot(A)", "etreeplot(A, nodespec, edgespec)"],
                  "Draws the elimination tree of a sparse matrix."),
}
