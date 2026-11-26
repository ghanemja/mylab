"""graph and digraph objects: construction, tables, node names, editing and structural queries."""

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


def test_construction_from_edge_lists(m):
    ok(m, """
        G = graph([1 2 3], [2 3 1]);
        assert(isa(G, 'graph') && numnodes(G) == 3 && numedges(G) == 3);
        % edges are stored with the smaller node first and sorted by source, then target
        assert(isequal(G.Edges.EndNodes, [1 2; 1 3; 2 3]) && isequal(size(G.Nodes), [3 0]));
        G = graph([3 1], [1 2], [7 9]);
        assert(isequal(G.Edges.EndNodes, [1 2; 1 3]) && isequal(G.Edges.Weight, [9; 7]));
        G = graph([1 2], [2 3], [], 5); assert(numnodes(G) == 5 && numedges(G) == 2);
        D = digraph([2 1], [1 3]); assert(isa(D, 'digraph') && isequal(D.Edges.EndNodes, [1 3; 2 1]));
        E = graph; assert(numnodes(E) == 0 && numedges(E) == 0);
        L = graph([1 1 2], [1 2 2], 'omitselfloops'); assert(numedges(L) == 1);
    """)
    with pytest.raises(MatlabError):
        m.run("graph([1 2], [2 3 4])")
    with pytest.raises(MatlabError):
        m.run("graph([1 2], [2 3], [], 2)")


def test_construction_from_adjacency_matrices(m):
    ok(m, """
        % a numeric adjacency matrix gives weighted edges, a logical one unweighted edges
        G = graph([0 1 1; 1 0 0; 1 0 0]);
        assert(isequal(G.Edges.EndNodes, [1 2; 1 3]) && isequal(G.Edges.Weight, [1; 1]));
        G = graph([0 2 5; 2 0 0; 5 0 0]); assert(isequal(G.Edges.Weight, [2; 5]));
        G = graph(sparse([0 2; 2 0])); assert(numedges(G) == 1 && G.Edges.Weight == 2);
        G = graph([0 1 2; 9 0 3; 9 9 0], 'upper');
        assert(isequal(G.Edges.EndNodes, [1 2; 1 3; 2 3]) && isequal(G.Edges.Weight, [1; 2; 3]));
        G = graph([0 1 2; 9 0 3; 8 7 0], 'lower'); assert(isequal(G.Edges.Weight, [9; 8; 7]));
        D = digraph([0 2; 0 0], {'x', 'y'}); assert(isequal(D.Nodes.Name, {'x'; 'y'}) && D.Edges.Weight == 2);
        D = digraph([0 1; 1 0]); assert(numedges(D) == 2);
        G = graph(true(2) & ~eye(2)); assert(numedges(G) == 1 && width(G.Edges) == 1);
    """)
    with pytest.raises(MatlabError):
        m.run("graph([0 1; 0 0])")
    with pytest.raises(MatlabError):
        m.run("graph([0 1 0; 1 0 0])")


def test_node_names_and_tables(m):
    ok(m, """
        G = graph({'a', 'b', 'c'}, {'b', 'c', 'a'});
        assert(isequal(G.Nodes.Name, {'a'; 'b'; 'c'}) && isequal(G.Edges.EndNodes, {'a' 'b'; 'a' 'c'; 'b' 'c'}));
        assert(findnode(G, 'c') == 3 && findnode(G, 'zz') == 0 && findedge(G, 'a', 'c') == 2);
        assert(isequal(neighbors(G, 'a'), {'b'; 'c'}) && isequal(shortestpath(G, 'a', 'c'), {'a', 'c'}));
        T = table([1 2; 2 3; 3 1], [5; 6; 7], 'VariableNames', {'EndNodes', 'Weight'});
        H = graph(T); assert(isequal(H.Edges.EndNodes, [1 2; 1 3; 2 3]) && isequal(H.Edges.Weight, [5; 7; 6]));
        N = table({'p'; 'q'; 'r'}, [10; 20; 30], 'VariableNames', {'Name', 'Size'});
        H = graph(T, N); assert(isequal(H.Nodes.Size, [10; 20; 30]) && findnode(H, 'q') == 2);
        H = graph([1 2], [2 3], [4 5], {'u', 'v', 'w'}); assert(isequal(H.Nodes.Name, {'u'; 'v'; 'w'}));
    """)


def test_assigning_to_the_tables(m):
    ok(m, """
        G = graph([1 1 2], [2 3 3], [1 4 2]);
        G.Edges.Weight(2) = 10; assert(isequal(G.Edges.Weight, [1; 10; 2]));
        G.Edges.Label = {'a'; 'b'; 'c'}; assert(isequal(G.Edges.Label, {'a'; 'b'; 'c'}));
        G.Nodes.Name = {'x'; 'y'; 'z'}; assert(findnode(G, 'z') == 3);
        G.Nodes.Value = [1; 2; 3]; H = rmnode(G, 'x');
        assert(isequal(H.Nodes.Value, [2; 3]) && isequal(H.Edges.Label, {'c'}) && H.Edges.Weight == 2);
        G2 = G; G2.Edges.Weight(1) = 99; assert(G.Edges.Weight(1) == 1);   % value semantics
    """)
    with pytest.raises(MatlabError):
        m.run("G = graph([1 2], [2 3]); G.Nodes.Name = {'a'; 'a'; 'b'};")
    with pytest.raises(MatlabError):
        m.run("G = graph([1 2], [2 3]); G.Edges.EndNodes = [1 3; 2 3];")


def test_adding_and_removing(m):
    ok(m, """
        G = graph([1 2], [2 3]);
        G = addedge(G, 3, 4); assert(numnodes(G) == 4 && isequal(G.Edges.EndNodes, [1 2; 2 3; 3 4]));
        G = addnode(G, 2); assert(numnodes(G) == 6 && degree(G, 6) == 0);
        G = rmedge(G, 1, 2); assert(numedges(G) == 2 && findedge(G, 1, 2) == 0);
        G = rmedge(G, 1); assert(isequal(G.Edges.EndNodes, [3 4]));
        G = rmnode(G, [1 2]); assert(numnodes(G) == 4 && isequal(G.Edges.EndNodes, [1 2]));
        W = graph([1 2], [2 3], [5 6]); W = addedge(W, 1, 3, 7); assert(isequal(W.Edges.Weight, [5; 7; 6]));
        N = graph({'a'}, {'b'}); N = addedge(N, 'b', 'c'); N = addnode(N, 'd');
        assert(isequal(N.Nodes.Name, {'a'; 'b'; 'c'; 'd'}) && numedges(N) == 2);
        D = digraph([1 2], [2 3]); F = flipedge(D); assert(isequal(F.Edges.EndNodes, [2 1; 3 2]));
    """)
    with pytest.raises(MatlabError):
        m.run("W = graph([1 2], [2 3], [5 6]); addedge(W, 1, 3)")


def test_neighbours_and_degrees(m):
    ok(m, """
        G = graph([1 1 2 3 4], [2 3 4 4 5]);
        assert(isequal(neighbors(G, 4), [2; 3; 5]) && isequal(degree(G), [2; 2; 2; 3; 1]) && degree(G, 4) == 3);
        assert(isequal(degree(G, [1 4]), [2 3]));
        L = graph([1 1], [1 2]); assert(isequal(degree(L), [3; 1]));          % a self-loop counts twice
        D = digraph([1 1 2 3], [2 3 4 4]);
        assert(isequal(successors(D, 1), [2; 3]) && isequal(predecessors(D, 4), [2; 3]));
        assert(isequal(indegree(D), [0; 1; 1; 2]) && isequal(outdegree(D), [2; 1; 1; 0]));
        [eid, nid] = outedges(D, 1); assert(isequal(eid, [1; 2]) && isequal(nid, [2; 3]));
        [eid, nid] = inedges(D, 4); assert(isequal(eid, [3; 4]) && isequal(nid, [2; 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("neighbors(digraph([1 2], [2 3]), 1)")
    with pytest.raises(MatlabError):
        m.run("degree(graph([1 2], [2 3]), 7)")


def test_matrices_of_a_graph(m):
    ok(m, """
        G = graph([1 1 2], [2 3 3], [5 6 7]);
        A = adjacency(G); assert(issparse(A) && isequal(full(A), [0 1 1; 1 0 1; 1 1 0]));
        assert(isequal(full(adjacency(G, 'weighted')), [0 5 6; 5 0 7; 6 7 0]));
        assert(isequal(full(adjacency(G, [1 2 3])), [0 1 2; 1 0 3; 2 3 0]));
        I = incidence(G); assert(issparse(I) && isequal(full(I), [-1 -1 0; 1 0 -1; 0 1 1]));
        Lp = laplacian(G); assert(isequal(full(Lp), [2 -1 -1; -1 2 -1; -1 -1 2]) && isequal(full(I * I'), full(Lp)));
        D = digraph([1 2], [2 3]); assert(isequal(full(adjacency(D)), [0 1 0; 0 0 1; 0 0 0]));
        assert(isequal(full(incidence(D)), [-1 0; 1 -1; 0 1]));
    """)


def test_subgraph_reorder_and_queries(m):
    ok(m, """
        G = graph([1 1 2 3 4], [2 3 4 4 5], [1 4 2 1 3]);
        H = subgraph(G, [1 2 4]); assert(isequal(H.Edges.EndNodes, [1 2; 2 3]) && isequal(H.Edges.Weight, [1; 2]));
        H = subgraph(G, logical([1 1 0 1 0])); assert(numnodes(H) == 3 && numedges(H) == 2);
        [R, idx] = reordernodes(G, [5 4 3 2 1]);
        assert(isequal(R.Edges.EndNodes, [1 2; 2 3; 2 4; 3 5; 4 5]) && isequal(R.Edges.Weight, G.Edges.Weight(idx)));
        M = graph([1 1 2], [2 2 3]); assert(ismultigraph(M) && ~ismultigraph(G) && edgecount(M, 1, 2) == 2);
        S = simplify(M); assert(numedges(S) == 2 && ~ismultigraph(S));
        [s, t] = findedge(G, [2 4]); assert(isequal([s t], [1 3; 3 4]));
        assert(isequal(graph([1 2], [2 3]), graph([2 1], [3 2])) && ...
            ~isequal(graph([1 2], [2 3]), digraph([1 2], [2 3])));
    """)


def test_graph_functions_reject_other_inputs(m):
    with pytest.raises(MatlabError):
        m.run("numnodes(5)")
    with pytest.raises(MatlabError):
        m.run("shortestpath([1 2 3], 1, 2)")


def test_display(m):
    out = m.run("G = graph([1 2], [2 3])")
    assert "graph with properties" in out and "Edges: [2×1 table]" in out and "Nodes: [3×0 table]" in out
