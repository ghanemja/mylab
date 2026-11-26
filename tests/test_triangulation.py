"""triangulation and delaunayTriangulation objects.

The 2-D mesh is made up for these tests: an open fan of four counterclockwise triangles around vertex 1.
Expected values are worked out by hand from the coordinates or computed here with NumPy.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError

FAN = """
P = [0 0; 3 0; 2 2; -1 2; -2 -1; 1 -2];
T = [1 2 3; 1 3 4; 1 4 5; 1 5 6];
TR = triangulation(T, P);
"""
P_FAN = np.array([[0, 0], [3, 0], [2, 2], [-1, 2], [-2, -1], [1, -2]], dtype=float)
T_FAN = np.array([[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 5, 6]])
BOUNDARY = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)}  # every edge but the three inner spokes 1-3, 1-4, 1-5


@pytest.fixture
def fan(m):
    m.run(FAN)
    return m


def _cells(m, expr, n):
    return [sorted(m.num(f"{expr}{{{k}}}").astype(int).ravel().tolist()) for k in range(1, n + 1)]


def test_construct_and_properties(fan):
    m = fan
    assert m.str("class(TR)") == "triangulation"
    assert m.num("isa(TR, 'triangulation')").item()
    assert m.num("isobject(TR)").item()
    np.testing.assert_array_equal(m.num("TR.Points"), P_FAN)
    np.testing.assert_array_equal(m.num("TR.ConnectivityList"), T_FAN)
    np.testing.assert_array_equal(m.num("TR.ConnectivityList(2, :)"), [[1, 3, 4]])
    np.testing.assert_array_equal(m.num("TR(2, :)"), [[1, 3, 4]])
    np.testing.assert_array_equal(m.num("TR(4, 2)"), [[5]])
    m.run("TR2 = triangulation(T, P(:, 1), P(:, 2));")
    assert m.num("isequal(TR, TR2)").item()
    out = m.run("TR")
    assert "triangulation with properties:" in out
    assert "Points: [6×2 double]" in out and "ConnectivityList: [4×3 double]" in out
    assert "Constraints" not in out


def test_size(fan):
    m = fan
    np.testing.assert_array_equal(m.num("size(TR)"), [[4, 3]])
    assert m.num("size(TR, 1)").item() == 4
    m.run("[a, b] = size(TR);")
    assert m.num("a").item() == 4 and m.num("b").item() == 3
    assert m.num("numel(TR)").item() == 1


def test_edges_sorted_unique(fan):
    m = fan
    expected = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [3, 4], [4, 5], [5, 6]]
    np.testing.assert_array_equal(m.num("edges(TR)"), expected)
    np.testing.assert_array_equal(m.num("TR.edges()"), expected)
    np.testing.assert_array_equal(m.num("TR.edges"), expected)


def test_free_boundary_is_a_counterclockwise_loop(fan):
    m = fan
    m.run("F = freeBoundary(TR);")
    F = m.num("F").astype(int)
    assert F.shape == (6, 2)
    assert all(F[k, 1] == F[(k + 1) % 6, 0] for k in range(6))  # one closed loop
    assert {tuple(sorted(e)) for e in F.tolist()} == BOUNDARY
    loop = P_FAN[F[:, 0] - 1]
    area2 = np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    assert area2 == pytest.approx(22.0)  # counterclockwise, twice the area 6 + 6 + 5 + 5 of the triangles
    m.run("[F2, P2] = freeBoundary(TR);")
    np.testing.assert_array_equal(m.num("P2(F2)"), m.num("P(F)"))


def test_neighbors(fan):
    m = fan
    nan = np.nan
    # column j: the triangle across the edge opposite vertex j
    expected = np.array([[nan, 2, nan], [nan, 3, 1], [nan, 4, 2], [nan, nan, 3]])
    np.testing.assert_array_equal(m.num("neighbors(TR)"), expected)
    np.testing.assert_array_equal(m.num("neighbors(TR, [2 3])"), expected[[1, 2]])
    np.testing.assert_array_equal(m.num("TR.neighbors(4)"), expected[[3]])


def test_vertex_and_edge_attachments(fan):
    m = fan
    m.run("V = vertexAttachments(TR); V1 = vertexAttachments(TR, 3); E = edgeAttachments(TR, 1, 3);")
    assert m.num("size(V)").tolist() == [[6, 1]]
    assert m.num("iscell(V)").item()
    assert _cells(m, "V", 6) == [[1, 2, 3, 4], [1], [1, 2], [2, 3], [3, 4], [4]]
    assert _cells(m, "V1", 1) == [[1, 2]]
    assert _cells(m, "E", 1) == [[1, 2]]
    m.run("E2 = edgeAttachments(TR, [2 3; 2 4]);")
    np.testing.assert_array_equal(m.num("E2{1}"), [[1]])
    assert m.num("isempty(E2{2})").item()  # 2-4 is not an edge


def test_is_connected(fan):
    m = fan
    np.testing.assert_array_equal(m.num("isConnected(TR, [1 3; 2 4])"), [[True], [False]])
    assert m.num("isConnected(TR, 6, 1)").item()


def _circumcenter(a, b, c):
    """Circumcentre from the perpendicular bisectors: |x - a|^2 = |x - b|^2 = |x - c|^2."""
    A = 2 * np.array([b - a, c - a])
    rhs = np.array([b @ b - a @ a, c @ c - a @ a])
    return np.linalg.solve(A, rhs)


def test_incenter_and_circumcenter(fan):
    m = fan
    m.run("[C, r] = incenter(TR, 1); [CC, rr] = circumcenter(TR);")
    a, b, c = P_FAN[[0, 1, 2]]  # triangle 1: (0,0), (3,0), (2,2)
    la, lb, lc = np.linalg.norm(b - c), np.linalg.norm(c - a), np.linalg.norm(a - b)  # sqrt(5), 2*sqrt(2), 3
    np.testing.assert_allclose(m.num("C"), [(la * a + lb * b + lc * c) / (la + lb + lc)])
    np.testing.assert_allclose(m.num("r"), [[2 * 3.0 / (la + lb + lc)]])  # area 3
    centres = np.array([_circumcenter(*P_FAN[t - 1]) for t in T_FAN])
    np.testing.assert_allclose(m.num("CC"), centres)
    np.testing.assert_allclose(m.num("rr").ravel(), np.linalg.norm(centres - P_FAN[T_FAN[:, 0] - 1], axis=1))
    # by hand: x = 1.5 (the base's bisector), then |(1.5, y)| = |(0.5, y - 2)| gives y = 0.5
    np.testing.assert_allclose(centres[0], [1.5, 0.5])


def test_barycentric_round_trip_and_point_location(fan):
    m = fan
    X = m.num("barycentricToCartesian(TR, [1; 2], [1/3 1/3 1/3; 1 0 0])")
    np.testing.assert_allclose(X, [[5 / 3, 2 / 3], [0.0, 0.0]])
    B = m.num("cartesianToBarycentric(TR, [1; 2], [5/3 2/3; 0 0])")
    np.testing.assert_allclose(B, [[1 / 3, 1 / 3, 1 / 3], [1, 0, 0]], atol=1e-12)
    m.run("[ti, Bq] = pointLocation(TR, [2 1; 5 5; -1 1]);")
    np.testing.assert_array_equal(m.num("ti"), [[1], [np.nan], [3]])
    np.testing.assert_allclose(m.num("sum(Bq(1, :))"), 1.0)
    np.testing.assert_allclose(m.num("Bq(1, :) * P(T(1, :), :)"), [[2, 1]])
    np.testing.assert_allclose(m.num("Bq(3, :) * P(T(3, :), :)"), [[-1, 1]])
    assert np.isnan(m.num("Bq(2, :)")).all()
    np.testing.assert_array_equal(m.num("pointLocation(TR, 2, 1)"), [[1]])


def test_nearest_neighbor(fan):
    m = fan
    m.run("[id, d] = nearestNeighbor(TR, [2.9 0.2; -2 -1.4]);")
    np.testing.assert_array_equal(m.num("id"), [[2], [5]])
    np.testing.assert_allclose(m.num("d"), [[np.hypot(0.1, 0.2)], [0.4]])


def test_feature_edges_2d_are_the_boundary(fan):
    m = fan
    np.testing.assert_array_equal(m.num("featureEdges(TR, pi/4)"), sorted(map(list, BOUNDARY)))


def test_read_only_properties(fan):
    m = fan
    with pytest.raises(MatlabError, match="read-only"):
        m.run("TR.Points = [0 0; 1 0; 0 1];")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("TR.ConnectivityList = [1 2 3];")
    with pytest.raises(MatlabError, match="Unrecognized method, property, or field 'Foo'"):
        m.run("x = TR.Foo;")


def test_constructor_errors(m):
    with pytest.raises(MatlabError, match="range 1 to 3"):
        m.run("triangulation([1 2 4], [0 0; 1 0; 0 1])")
    with pytest.raises(MatlabError, match="3 columns"):
        m.run("triangulation([1 2], [0 0; 1 0; 0 1])")
    with pytest.raises(MatlabError, match="3-D points"):
        m.run("triangulation([1 2 3 4], [0 0; 1 0; 0 1; 1 1])")


def test_method_on_wrong_type(m):
    with pytest.raises(MatlabError, match="type 'double'"):
        m.run("freeBoundary(5)")
    with pytest.raises(MatlabError, match="No method 'convexHull'|Unrecognized|delaunayTriangulation"):
        m.run("convexHull(triangulation([1 2 3], [0 0; 1 0; 0 1]))")


# -- 3-D --------------------------------------------------------------------------------------------------------

TET = "TT = triangulation([1 2 3 4], [0 0 0; 1 0 0; 0 1 0; 0 0 1]);"


def test_tetrahedron(m):
    m.run(TET)
    F = m.num("freeBoundary(TT)").astype(int) - 1
    assert F.shape == (4, 3)
    P = m.num("TT.Points")
    centroid = P.mean(axis=0)
    for f in F:  # outward normals
        a, b, c = P[f]
        assert np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - centroid) > 0
    m.run("[C, r] = circumcenter(TT); [I, ri] = incenter(TT);")
    np.testing.assert_allclose(m.num("C"), [[0.5, 0.5, 0.5]])
    np.testing.assert_allclose(m.num("r"), [[np.sqrt(3) / 2]])
    r_in = 0.5 / (1.5 + np.sqrt(3) / 2)
    np.testing.assert_allclose(m.num("I"), [[r_in, r_in, r_in]])
    np.testing.assert_allclose(m.num("ri"), [[r_in]])
    assert m.num("size(edges(TT), 1)").item() == 6
    np.testing.assert_array_equal(m.num("neighbors(TT)"), [[np.nan] * 4])
    m.run("[id, B] = pointLocation(TT, [0.1 0.1 0.1; 1 1 1]);")
    np.testing.assert_array_equal(m.num("id"), [[1], [np.nan]])
    np.testing.assert_allclose(m.num("B(1, :)"), [[0.7, 0.1, 0.1, 0.1]])
    with pytest.raises(MatlabError, match="3-D surface"):
        m.run("faceNormal(TT)")


def test_surface_normals_and_feature_edges(m):
    # two triangles folded along the edge 1-3 by 90 degrees
    m.run("TS = triangulation([1 2 3; 1 3 4], [0 0 0; 1 0 0; 1 1 0; 0 1 1]);")
    n2 = np.array([1.0, -1.0, 1.0]) / np.sqrt(3)
    np.testing.assert_allclose(m.num("faceNormal(TS)"), [[0, 0, 1], n2], atol=1e-12)
    np.testing.assert_allclose(m.num("TS.faceNormal(2)"), [n2], atol=1e-12)
    V = m.num("vertexNormal(TS)")
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0)
    np.testing.assert_allclose(V[1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(V[3], n2, atol=1e-12)
    # the fold angle is acos(1/sqrt(3)) = 0.955 rad
    np.testing.assert_array_equal(m.num("featureEdges(TS, 0.5)"), [[1, 2], [1, 3], [1, 4], [2, 3], [3, 4]])
    np.testing.assert_array_equal(m.num("featureEdges(TS, 1.0)"), [[1, 2], [1, 4], [2, 3], [3, 4]])
    with pytest.raises(MatlabError, match="pointLocation"):
        m.run("pointLocation(TS, [0 0 0])")


# -- delaunayTriangulation -----------------------------------------------------------------------------------------

SQUARE = "x = [0 1 0 1 0.5]'; y = [0 0 1 1 0.5]'; DT = delaunayTriangulation(x, y);"


def test_delaunay_basic(m):
    m.run(SQUARE)
    assert m.str("class(DT)") == "delaunayTriangulation"
    assert m.num("isa(DT, 'triangulation')").item()
    assert m.num("isa(DT, 'delaunayTriangulation')").item()
    T = m.num("DT.ConnectivityList").astype(int) - 1
    P = m.num("DT.Points")
    assert T.shape == (4, 3)
    assert all(4 in t for t in T)  # every triangle uses the centre point
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert (area2 > 0).all()  # counterclockwise
    assert m.num("isempty(DT.Constraints)").item()
    assert m.num("size(edges(DT), 1)").item() == 8
    np.testing.assert_array_equal(m.num("size(DT)"), [[4, 3]])
    out = m.run("DT")
    assert "delaunayTriangulation with properties:" in out and "Constraints: []" in out
    m.run("DT2 = delaunayTriangulation([x y]);")
    assert m.num("isequal(DT, DT2)").item()


def test_delaunay_convex_hull_and_voronoi(m):
    m.run(SQUARE)
    m.run("[K, A] = convexHull(DT);")
    np.testing.assert_array_equal(m.num("K"), [[1], [2], [4], [3], [1]])
    assert m.num("A").item() == pytest.approx(1.0)
    np.testing.assert_array_equal(m.num("DT.convexHull()"), [[1], [2], [4], [3], [1]])
    m.run("[V, r] = voronoiDiagram(DT);")
    V = m.num("V")
    assert np.isinf(V[0]).all() and V.shape == (5, 2)
    assert m.num("size(r)").tolist() == [[5, 1]]
    centre = m.num("r{5}").astype(int).ravel()
    assert 1 not in centre and len(centre) == 4
    np.testing.assert_allclose(sorted(map(tuple, V[centre - 1])), [(0, 0.5), (0.5, 0), (0.5, 1), (1, 0.5)])
    corner = m.num("r{1}").astype(int).ravel()
    assert corner[0] == 1 and sorted(map(tuple, V[corner[1:] - 1])) == [(0, 0.5), (0.5, 0)]


def test_delaunay_3d(m):
    m.run("DT = delaunayTriangulation([0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 1]);")
    T = m.num("DT.ConnectivityList")
    assert T.shape[1] == 4
    m.run("[K, v] = convexHull(DT);")
    assert m.num("size(K, 2)").item() == 3
    assert m.num("v").item() == pytest.approx(1 / 6 + 1 / 3)  # corner tet plus the far tet
    assert m.num("pointLocation(DT, [0.1 0.1 0.1])").shape == (1, 1)
    m.run("DT3 = delaunayTriangulation([0 1 0 0 1]', [0 0 1 0 1]', [0 0 0 1 1]');")
    assert m.num("isequal(DT3.ConnectivityList, DT.ConnectivityList)").item()


def test_delaunay_duplicates_and_empty(m):
    out = m.run("DT = delaunayTriangulation([0 0; 1 0; 0 1; 1 0; 1 1]);")
    assert "Duplicate data points have been detected and removed" in out
    np.testing.assert_array_equal(m.num("DT.Points"), [[0, 0], [1, 0], [0, 1], [1, 1]])
    assert m.num("size(DT, 1)").item() == 2
    m.run("E = delaunayTriangulation();")
    assert m.num("isempty(E.Points)").item() and m.num("size(E, 1)").item() == 0


def test_delaunay_points_assignment_retriangulates(m):
    m.run(SQUARE)
    m.run("DT.Points(end+1, :) = [2 2];")
    assert m.num("size(DT.Points, 1)").item() == 6
    assert m.num("max(DT.ConnectivityList(:))").item() == 6
    with pytest.raises(MatlabError, match="read-only"):
        m.run("DT.ConnectivityList = [1 2 3];")


def test_constraint_is_recovered(m):
    # the Delaunay triangulation of this rhombus uses the short diagonal 2-4
    m.run("P = [0 0; 2 -1; 4 0; 2 1]; D0 = delaunayTriangulation(P); DC = delaunayTriangulation(P, [1 3]);")
    assert m.num("ismember([2 4], edges(D0), 'rows')").item()
    assert m.num("ismember([1 3], edges(DC), 'rows')").item()
    assert not m.num("ismember([2 4], edges(DC), 'rows')").item()
    np.testing.assert_array_equal(m.num("DC.Constraints"), [[1, 3]])
    assert m.num("size(DC, 1)").item() == 2
    m.run("DX = delaunayTriangulation(P(:, 1), P(:, 2), [1 3]);")
    assert m.num("isequal(DX.ConnectivityList, DC.ConnectivityList)").item()


def test_constrained_region_and_is_interior(m):
    m.run("""
        outer = [0 0; 4 0; 4 4; 0 4]; inner = [1 1; 3 1; 3 3; 1 3];
        P = [outer; inner];
        C = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5];
        DT = delaunayTriangulation(P, C);
        TF = isInterior(DT);
    """)
    assert m.num("all(ismember(sort(C, 2), edges(DT), 'rows'))").item()
    TF = m.num("TF")
    assert TF.shape == (10, 1) and TF.dtype == bool
    assert TF.sum() == 8  # the frame between the squares; the hole's 2 triangles are outside
    T = m.num("DT.ConnectivityList").astype(int)
    assert all(set(t) <= {5, 6, 7, 8} for t in T[~TF.ravel()])
    out = m.run("DT")
    assert "Constraints: [8×2 double]" in out


def test_constrained_points_on_a_constraint_and_crossing(m):
    # point 5 lies on the constrained edge 1-3: the constraint is split there
    m.run("DT = delaunayTriangulation([0 0; 2 -1; 4 0; 2 1; 2 0], [1 3]);")
    np.testing.assert_array_equal(m.num("DT.Constraints"), [[1, 5], [5, 3]])
    with pytest.raises(MatlabError, match="intersect"):
        m.run("delaunayTriangulation([0 0; 1 0; 1 1; 0 1], [1 3; 2 4])")
    with pytest.raises(MatlabError, match="2-D"):
        m.run("delaunayTriangulation([0 0 0; 1 0 0; 0 1 0; 0 0 1], [1 2])")


def test_is_interior_needs_constraints(m):
    m.run(SQUARE)
    with pytest.raises(MatlabError, match="constraints"):
        m.run("isInterior(DT)")


def test_triplot_accepts_triangulation(m):
    m.run(SQUARE + " h = triplot(DT); TS = triangulation([1 2 3], [0 0 0; 1 0 0; 0 1 1]); trisurf(TS);")
    assert "Line" in m.str("class(h)")


def test_methods_and_properties_lists(m):
    m.run(SQUARE)
    props = m.run("properties(DT)")
    assert "Constraints" in props and "ConnectivityList" in props
    meths = m.run("methods(DT)")
    for name in ("convexHull", "voronoiDiagram", "isInterior", "freeBoundary", "pointLocation"):
        assert name in meths
