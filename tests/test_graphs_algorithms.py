"""Graph algorithms, graph plots and the sparse structure functions. Expected values are worked out
by hand on small graphs."""

import os

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


# The weighted test graph:   1 --1-- 2 --2-- 4 --3-- 5
#                            |               |
#                            +--5-- 3 --1----+
# Every shortest path in it is unique (1 to 3 goes through 2 and 4, length 4).
WG = "G = graph([1 1 2 3 4], [2 3 4 4 5], [1 5 2 1 3]);"


def test_shortest_paths(m):
    ok(m, WG + """
        [P, d, ep] = shortestpath(G, 1, 5); assert(isequal(P, [1 2 4 5]) && d == 6 && isequal(ep, [1 3 5]));
        [P, d] = shortestpath(G, 1, 3); assert(isequal(P, [1 2 4 3]) && d == 4);
        [P, d] = shortestpath(G, 1, 3, 'Method', 'unweighted'); assert(isequal(P, [1 3]) && d == 1);
        D = digraph([1 2], [2 3]); assert(isequal(shortestpath(D, 1, 3), [1 2 3]));
        [P, d] = shortestpath(D, 3, 1); assert(isempty(P) && isinf(d));
        N = digraph([1 1 2], [2 3 3], [4 1 -2]);           % a negative weight needs Bellman-Ford
        [P, d] = shortestpath(N, 1, 3); assert(isequal(P, [1 3]) && d == 1);
        [P, d] = shortestpath(N, 1, 3, 'Method', 'acyclic'); assert(d == 1);
        N2 = digraph([1 2 3], [2 3 1], [1 -3 1]);
    """)
    with pytest.raises(MatlabError):
        m.run("N2 = digraph([1 2 3], [2 3 1], [1 -3 1]); shortestpath(N2, 1, 3)")
    with pytest.raises(MatlabError):
        m.run("shortestpath(graph([1 2], [2 3], [1 -1]), 1, 3, 'Method', 'positive')")


def test_distances_and_trees(m):
    ok(m, WG + """
        d = distances(G); assert(isequal(d(1, :), [0 1 4 3 6]) && isequal(d, d') && isequal(distances(G, 1), d(1, :)));
        assert(isequal(distances(G, [1 2], [4 5]), [3 6; 2 5]));
        du = distances(G, 'Method', 'unweighted'); assert(isequal(du(1, :), [0 1 1 2 3]));
        [TR, D, E] = shortestpathtree(G, 1);
        assert(isa(TR, 'digraph') && isequal(TR.Edges.EndNodes, [1 2; 2 4; 4 3; 4 5]) && isequal(D, [0 1 4 3 6]));
        assert(isequal(E', logical([1 0 1 1 1])));
        C = shortestpathtree(G, 1, 'OutputForm', 'cell'); assert(isequal(C{5}, [1 2 4 5]) && isequal(C{1}, 1));
        v = shortestpathtree(G, 1, 'OutputForm', 'vector'); assert(isequal(v, [0 1 4 2 4]));
        [TR, D] = shortestpathtree(G, 1, [3 5]); assert(isequal(D, [4 6]) && numedges(TR) == 4);
        [ids, dist] = nearest(G, 1, 3); assert(isequal(ids, [2; 4]) && isequal(dist, [1; 3]));
        I = digraph([1 2], [2 3]); d = distances(I); assert(isinf(d(3, 1)) && d(1, 3) == 2);
    """)


def test_connected_components(m):
    ok(m, """
        G = graph([1 2 4], [2 3 5]); [bins, sizes] = conncomp(G);
        assert(isequal(bins, [1 1 1 2 2]) && isequal(sizes, [3 2]));
        c = conncomp(G, 'OutputForm', 'cell'); assert(isequal(c{1}, [1 2 3]) && isequal(c{2}, [4 5]));
        D = digraph([1 2 3 3 4], [2 3 1 4 5]);            % a 3-cycle followed by a chain
        b = conncomp(D); assert(b(1) == b(2) && b(2) == b(3) && numel(unique(b)) == 3 && b(4) < b(5) && b(3) < b(4));
        assert(isequal(conncomp(D, 'Type', 'weak'), [1 1 1 1 1]));
        C = condensation(D); assert(numnodes(C) == 3 && isdag(C) && numedges(C) == 2);
    """)


def test_searches(m):
    ok(m, WG + """
        assert(isequal(bfsearch(G, 1), [1; 2; 3; 4; 5]) && isequal(dfsearch(G, 1), [1; 2; 4; 3; 5]));
        assert(isequal(bfsearch(G, 5), [5; 4; 2; 3; 1]));
        e = dfsearch(G, 1, 'edgetonew'); assert(isequal(e, [1 2; 2 4; 4 3; 4 5]));
        f = dfsearch(G, 1, 'finishnode'); assert(isequal(f, [3; 5; 4; 2; 1]));
        T = bfsearch(G, 1, 'allevents');
        assert(istable(T) && isequal(T.Properties.VariableNames, {'Event', 'Node', 'Edge', 'EdgeIndex'}));
        H = graph([1 3], [2 4]);
        assert(isequal(bfsearch(H, 1), [1; 2]) && isequal(bfsearch(H, 1, 'Restart', true), [1; 2; 3; 4]));
        D = digraph([1 1 2], [2 3 3]); assert(isequal(dfsearch(D, 1), [1; 2; 3]) && isequal(dfsearch(D, 3), 3));
    """)


def test_spanning_tree_and_ordering(m):
    ok(m, WG + """
        [T, pred] = minspantree(G); assert(isequal(T.Edges.EndNodes, [1 2; 2 4; 3 4; 4 5]) && sum(T.Edges.Weight) == 7);
        assert(isequal(pred, [0 1 4 2 4]));
        T2 = minspantree(G, 'Method', 'sparse'); assert(isequal(T2.Edges.EndNodes, T.Edges.EndNodes));
        [T3, p3] = minspantree(G, 'Root', 5); assert(p3(5) == 0 && p3(4) == 5 && sum(T3.Edges.Weight) == 7);
        F = graph([1 3], [2 4], [1 2]); [T4, p4] = minspantree(F); assert(numedges(T4) == 1 && all(isnan(p4(3:4))));
        T5 = minspantree(F, 'Type', 'forest'); assert(numedges(T5) == 2);
        D = digraph([1 1 2 3], [2 3 4 4]);
        n = toposort(D); pos(n) = 1:4; assert(all(pos(D.Edges.EndNodes(:, 1)) < pos(D.Edges.EndNodes(:, 2))));
        assert(isequal(toposort(D, 'Order', 'stable'), [1 2 3 4]) && isdag(D) && ~hascycles(D));
        C = digraph([1 2 3], [2 3 1]); assert(~isdag(C) && hascycles(C));
        assert(hascycles(G) && ~hascycles(graph([1 2], [2 3])));
    """)
    with pytest.raises(MatlabError):
        m.run("toposort(digraph([1 2 3], [2 3 1]))")
    with pytest.raises(MatlabError):
        m.run("toposort(graph([1 2], [2 3]))")


def test_maxflow(m):
    # capacities: 1->2 (3), 1->3 (2), 2->3 (1), 2->4 (2), 3->4 (3); the cut {1} has capacity 5
    ok(m, """
        D = digraph([1 1 2 2 3], [2 3 3 4 4], [3 2 1 2 3]);
        [mf, GF, cs, ct] = maxflow(D, 1, 4);
        assert(mf == 5 && isequal(cs, 1) && isequal(ct, [2; 3; 4]) && isa(GF, 'digraph'));
        % flow conservation at the inner nodes
        A = full(adjacency(GF, 'weighted')); assert(sum(A(:, 2)) == sum(A(2, :)) && sum(A(:, 3)) == sum(A(3, :)));
        assert(sum(A(1, :)) == 5 && all(GF.Edges.Weight <= 3));
        assert(maxflow(D, 1, 4, 'augmentpath') == 5 && maxflow(D, 1, 4, 'pushrelabel') == 5);
        G = graph([1 1 2 3], [2 3 4 4]); assert(maxflow(G, 1, 4) == 2);    % unit capacities, two disjoint paths
        assert(maxflow(D, 4, 1) == 0);
    """)


def test_centrality(m):
    ok(m, """
        P = graph([1 2 3], [2 3 4]);                      % a path with four nodes
        assert(isequal(centrality(P, 'degree'), [1; 2; 2; 1]));
        assert(isequal(centrality(P, 'betweenness'), [0; 2; 2; 0]));
        c = centrality(P, 'closeness'); assert(norm(c - [1/6; 1/4; 1/4; 1/6]) < 1e-14);
        pr = centrality(P, 'pagerank'); assert(abs(sum(pr) - 1) < 1e-12 && pr(2) > pr(1) && abs(pr(1) - pr(4)) < 1e-6);
        ev = centrality(P, 'eigenvector');
        assert(abs(sum(ev) - 1) < 1e-12 && abs(ev(2) / ev(1) - (1 + sqrt(5)) / 2) < 1e-3);
        S = graph([1 1 1], [2 3 4]);                      % a star: the centre lies on all three pairs' paths
        assert(isequal(centrality(S, 'betweenness'), [3; 0; 0; 0]));
        W = graph([1 2], [2 3], [1 5]);
        assert(isequal(centrality(W, 'degree', 'Importance', W.Edges.Weight), [1; 6; 5]));
        c = centrality(W, 'closeness', 'Cost', W.Edges.Weight); assert(abs(c(1) - 1 / 7) < 1e-14);
        D = digraph([1 1 2], [2 3 3]);
        assert(isequal(centrality(D, 'indegree'), [0; 1; 2]) && isequal(centrality(D, 'outdegree'), [2; 1; 0]));
        h = centrality(D, 'hubs'); a = centrality(D, 'authorities');
        assert(abs(sum(h) - 1) < 1e-10 && h(3) < 1e-10 && a(1) < 1e-10 && a(3) > a(2));
        oc = centrality(D, 'outcloseness'); assert(abs(oc(1) - 0.5) < 1e-14 && oc(3) == 0);
    """)
    with pytest.raises(MatlabError):
        m.run("centrality(graph([1 2], [2 3]), 'hubs')")
    with pytest.raises(MatlabError):
        m.run("centrality(graph([1 2], [2 3]), 'degree', 'Cost', [1 2])")


def test_closures_and_paths(m):
    ok(m, """
        D = digraph([1 2], [2 3]);
        assert(isequal(transclosure(D).Edges.EndNodes, [1 2; 1 3; 2 3]));
        R = transreduction(digraph([1 1 2], [2 3 3])); assert(isequal(R.Edges.EndNodes, [1 2; 2 3]));
        G = graph([1 1 2 3], [2 3 4 4]);
        p = allpaths(G, 1, 4); assert(numel(p) == 2 && isequal(p{1}, [1 2 4]) && isequal(p{2}, [1 3 4]));
        p = allpaths(G, 1, 4, 'MaxNumPaths', 1); assert(numel(p) == 1);
    """)


def test_plot_draws_a_node_link_diagram(m, tmp_path):
    png = os.path.join(tmp_path, "g.png").replace("\\", "/")
    ok(m, f"""
        G = graph([1 1 2 3 4], [2 3 4 4 5], [1 4 2 1 3]);
        h = plot(G);
        assert(isequal(size(h.XData), [1 5]) && isequal(size(h.YData), [1 5]) && all(isfinite([h.XData h.YData])));
        % no two nodes are drawn at the same place
        d = hypot(h.XData' - h.XData, h.YData' - h.YData) + eye(5); assert(min(d(:)) > 1e-3);
        assert(isequal(h.NodeLabel, {{'1', '2', '3', '4', '5'}}));
        highlight(h, [1 2 4 5], 'EdgeColor', 'r', 'LineWidth', 2); highlight(h, 3, 'NodeColor', [0 1 0]);
        assert(isequal(h.LineWidth, [2 0.5 2 0.5 2]) && isequal(h.NodeColor(3, :), [0 1 0]));
        h.MarkerSize = 6; assert(h.MarkerSize == 6);
        labelnode(h, 1, 'start'); assert(strcmp(h.NodeLabel{{1}}, 'start'));
        layout(h, 'circle'); assert(norm(hypot(h.XData, h.YData) - 1) < 1e-12);
        print('-dpng', '{png}'); close all
        D = digraph([1 1 2 3], [2 3 4 4]);
        hl = plot(D, 'Layout', 'layered');
        assert(hl.YData(1) > hl.YData(2) && hl.YData(2) == hl.YData(3) && hl.YData(3) > hl.YData(4));
        hf = plot(G, 'XData', 1:5, 'YData', [0 1 0 1 0], 'EdgeLabel', G.Edges.Weight, 'NodeColor', 'k');
        assert(isequal(hf.XData, 1:5) && isequal(hf.EdgeLabel, {{'1', '4', '2', '1', '3'}}));
        close all
    """)
    assert os.path.getsize(png) > 2000
    with pytest.raises(MatlabError):
        m.run("plot(graph([1 2], [2 3]), 'Layout', 'nosuchlayout')")
    with pytest.raises(MatlabError):
        m.run("plot(graph([1 2], [2 3]), 'NoSuchOption', 1)")


def test_sparse_structure_functions(m):
    ok(m, """
        p = dmperm(sparse([1 0; 1 1])); assert(isequal(sort(p), [1 2]));
        A = sparse([0 1 1; 1 0 0; 0 0 1]);
        [p, q, r, s] = dmperm(A); B = full(A(p, q));
        assert(istriu(B) && all(diag(B) ~= 0) && isequal(r, [1 2 3 4]) && isequal(s, [1 2 3 4]));
        assert(isequal(dmperm(sparse([1 1; 0 0])) > 0, [true false]) && sprank(sparse([1 1; 0 0])) == 1);
        T = sparse([4 1 0 1; 1 4 1 0; 0 1 4 1; 1 0 1 4]);
        [par, post] = etree(T); assert(isequal(par, [2 3 4 0]) && isequal(post, [1 2 3 4]));
        assert(isequal(etree(sparse(diag([1 2 3]))), [0 0 0]));
        assert(isequal(colperm(sparse([1 1 1; 0 1 0; 0 1 1])), [1 3 2]));
        [x, y] = gplot([0 1; 1 0], [0 0; 1 2]);
        assert(isequal(x(1:2)', [0 1]) && isequal(y(1:2)', [0 2]) && isnan(x(3)));
        [x, y, h] = treelayout([0 1 1 2]); assert(h == 3 && y(1) > y(2) && y(2) == y(3) && y(2) > y(4));
    """)


def test_sparse_scalar_indexing_stays_sparse(m):
    ok(m, """
        S = sparse([1 0; 0 2]);
        assert(issparse(S(1, 1)) && issparse(S(4)) && full(S(2, 2)) == 2 && S(2, 2) == 2);
        x = zeros(1, 2); x(1) = S(2, 2); assert(~issparse(x) && isequal(x, [2 0]));
        assert(~issparse(S(2, 2) + 1) && S(2, 2) + 1 == 3 && isequal(size(zeros(S(2, 2))), [2 2]));
        if S(1, 2), error('zero entry must be false'); end
    """)


def test_eigs_and_svds_options(m):
    ok(m, """
        A = sparse(diag(1:10));
        assert(isequal(eigs(A, 2), [10; 9]) && isequal(eigs(A, 2, 'smallestabs'), [1; 2]));
        assert(norm(eigs(A, 3, 4.2) - [4; 5; 3]) < 1e-10);
        d = eigs(A, 2, 'largestabs', 'Tolerance', 1e-8, 'MaxIterations', 500); assert(isequal(d, [10; 9]));
        opts.tol = 1e-8; opts.maxit = 300; assert(isequal(eigs(A, 2, 'lm', opts), [10; 9]));
        d = eigs(@(x) A * x, 10, 2); assert(norm(d - [10; 9]) < 1e-10);
        d = eigs(@(x) A \\ x, 10, 2, 'smallestabs'); assert(norm(d - [1; 2]) < 1e-10);
        [V, D, flag] = eigs(A, 2); assert(flag == 0 && norm(A * V - V * D) < 1e-10);
        assert(svds(A, 1) == 10 && isequal(svds(A, 2, 'smallest'), [2; 1]));
        s = svds(A, 2, 'largest', 'Tolerance', 1e-10); assert(isequal(s, [10; 9]));
    """)
    with pytest.raises(MatlabError):
        m.run("eigs(sparse(diag(1:10)), 2, 'largestabs', 'NoSuchOption', 1)")
    with pytest.raises(MatlabError):
        m.run("svds(sparse(diag(1:10)), 2, 'largest', 'NoSuchOption', 1)")
