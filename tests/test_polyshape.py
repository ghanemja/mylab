"""polyshape objects, their methods and nsidedpoly.

The shapes are made up for these tests (squares, an L, a frame with a window, random polygons).
Expected values are worked out by hand, with the shoelace formula in NumPy, or by sampling points:
for the set operations the inside test of the result must agree with the Boolean combination of
the inside tests of the inputs at random points away from every boundary.
"""

import numpy as np
import pytest

from mylab.errors import MatlabError


def shoelace(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def mat(P):
    return "[" + "; ".join(f"{float(a)!r} {float(b)!r}" for a, b in np.asarray(P, dtype=float)) + "]"


def boundaries(V):
    """Split an N-by-2 vertex matrix at NaN rows."""
    out, cur = [], []
    for row in V:
        if np.isnan(row).any():
            out.append(np.array(cur))
            cur = []
        else:
            cur.append(row)
    if cur:
        out.append(np.array(cur))
    return out


# -- construction -----------------------------------------------------------------------------------------------

def test_construct_square_is_clockwise(m):
    m.run("p = polyshape([0 2 2 0], [0 0 1 1]);")  # given counterclockwise
    assert m.str("class(p)") == "polyshape"
    assert m.num("isa(p, 'polyshape')").item() and m.num("isobject(p)").item()
    V = m.num("p.Vertices")
    assert V.shape == (4, 2)
    assert shoelace(V) == pytest.approx(-2.0)  # stored clockwise
    np.testing.assert_array_equal(V[0], [0, 0])  # first vertex kept
    assert set(map(tuple, V)) == {(0, 0), (2, 0), (2, 1), (0, 1)}
    assert m.num("p.NumRegions").item() == 1 and m.num("p.NumHoles").item() == 0
    out = m.run("p")
    assert "polyshape with properties:" in out
    assert "Vertices: [4×2 double]" in out and "NumRegions: 1" in out and "NumHoles: 0" in out


def test_clockwise_input_is_kept_as_given(m):
    m.run("p = polyshape([0 0 3 3], [0 1 1 0]);")
    np.testing.assert_array_equal(m.num("p.Vertices"), [[0, 0], [0, 1], [3, 1], [3, 0]])


def test_matrix_and_cell_forms(m):
    m.run("a = polyshape([0 0; 2 0; 2 2; 0 2]);"
          "b = polyshape({[0 2 2 0], [5 6 6]}, {[0 0 2 2], [0 0 1]});"
          "c = polyshape([0 2 2 0 NaN 5 6 6], [0 0 2 2 NaN 0 0 1]);")
    assert m.num("area(a)").item() == pytest.approx(4.0)
    assert m.num("b.NumRegions").item() == 2
    assert m.num("area(b)").item() == pytest.approx(4.5)
    assert m.num("isequal(b, c)").item()
    V = m.num("b.Vertices")
    assert np.isnan(V[4]).all() and V.shape == (8, 2)


def test_empty_polyshape(m):
    m.run("p = polyshape();")
    assert m.num("p.Vertices").shape == (0, 2)
    assert m.num("p.NumRegions").item() == 0
    assert m.num("area(p)").item() == 0
    assert m.num("numboundaries(p)").item() == 0
    m.run("[x, y] = centroid(p);")
    assert np.isnan(m.num("x").item())


def test_hole_from_nesting_and_orientation(m):
    # both boundaries counterclockwise: the inner one is enclosed twice, so it is a hole
    m.run("p = polyshape({[0 6 6 0], [2 4 4 2]}, {[0 0 6 6], [2 2 4 4]});")
    assert m.num("p.NumHoles").item() == 1 and m.num("p.NumRegions").item() == 1
    assert m.num("area(p)").item() == pytest.approx(32.0)
    B = boundaries(m.num("p.Vertices"))
    assert shoelace(B[0]) < 0 < shoelace(B[1])  # solid clockwise, hole counterclockwise
    np.testing.assert_array_equal(m.num("ishole(p)"), [[False], [True]])
    # with 'ccw' the counterclockwise inner square is solid too, so it merges into the outer one
    m.run("q = polyshape({[0 6 6 0], [2 4 4 2]}, {[0 0 6 6], [2 2 4 4]}, 'SolidBoundaryOrientation', 'ccw');")
    assert m.num("q.NumHoles").item() == 0
    assert m.num("area(q)").item() == pytest.approx(36.0)
    # with 'ccw' a clockwise inner square is a hole
    m.run("r = polyshape({[0 6 6 0], [2 2 4 4]}, {[0 0 6 6], [2 4 4 2]}, 'SolidBoundaryOrientation', 'ccw');")
    assert m.num("r.NumHoles").item() == 1


def test_simplify_warns_and_repairs_a_bowtie(m):
    out = m.run("p = polyshape([0 2 2 0], [0 2 0 2]);")
    assert "Warning:" in out
    assert m.str("lastwarn") and m.interp.last_warning_id == "MATLAB:polyshape:repairedBySimplify"
    assert m.num("p.NumRegions").item() == 2
    assert m.num("area(p)").item() == pytest.approx(2.0)  # two triangles of area 1
    assert m.num("issimplified(p)").item()


def test_warning_can_be_turned_off(m):
    out = m.run("warning('off', 'MATLAB:polyshape:repairedBySimplify'); p = polyshape([0 2 2 0], [0 2 0 2]);")
    assert "Warning" not in out


def test_simplify_false_keeps_vertices(m):
    out = m.run("p = polyshape([0 2 2 0], [0 2 0 2], 'Simplify', false);")
    assert "Warning" not in out
    np.testing.assert_array_equal(m.num("p.Vertices"), [[0, 0], [2, 2], [2, 0], [0, 2]])
    assert not m.num("issimplified(p)").item()
    m.run("q = simplify(p);")
    assert m.num("q.NumRegions").item() == 2 and m.num("issimplified(q)").item()


def test_closing_vertex_and_duplicates(m):
    out = m.run("p = polyshape([0 1 1 0 0], [0 0 1 1 0]);")
    assert "Warning" not in out  # repeating the first vertex to close the loop is fine
    assert m.num("numsides(p)").item() == 4
    out = m.run("q = polyshape([0 1 1 1 0], [0 0 1 1 1]);")
    assert "Warning" in out and m.num("numsides(q)").item() == 4


def test_collinear_points(m):
    m.run("p = polyshape([0 1 2 2 0], [0 0 0 1 1]);")
    assert m.num("numsides(p)").item() == 4
    m.run("q = polyshape([0 1 2 2 0], [0 0 0 1 1], 'KeepCollinearPoints', true);")
    assert m.num("numsides(q)").item() == 5
    assert m.num("area(q)").item() == pytest.approx(2.0)


def test_too_few_points_dropped_with_warning(m):
    out = m.run("p = polyshape({[0 1 1], [5 6]}, {[0 0 1], [5 5]});")
    assert "Warning" in out and m.num("numboundaries(p)").item() == 1


def test_bad_inputs(m):
    with pytest.raises(MatlabError, match="NoSuchOption"):
        m.run("polyshape([0 1 1], [0 0 1], 'NoSuchOption', 1);")
    with pytest.raises(MatlabError):
        m.run("polyshape([0 1 1], [0 0]);")
    with pytest.raises(MatlabError):
        m.run("polyshape([0 1 Inf], [0 0 1]);")
    with pytest.raises(MatlabError):
        m.run("perimeter(3);")


# -- measures and queries --------------------------------------------------------------------------------------------

L_SHAPE = np.array([[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]], dtype=float)


def test_area_perimeter_centroid_of_an_l(m):
    m.run(f"p = polyshape({mat(L_SHAPE)});")
    assert m.num("area(p)").item() == pytest.approx(abs(shoelace(L_SHAPE)))
    assert m.num("perimeter(p)").item() == pytest.approx(12.0)
    m.run("[cx, cy] = centroid(p);")
    # two rectangles: 3x1 centred (1.5, 0.5) and 1x2 centred (0.5, 2)
    assert m.num("cx").item() == pytest.approx((3 * 1.5 + 2 * 0.5) / 5)
    assert m.num("cy").item() == pytest.approx((3 * 0.5 + 2 * 2) / 5)
    m.run("[xl, yl] = boundingbox(p);")
    np.testing.assert_array_equal(m.num("xl"), [[0, 3]])
    np.testing.assert_array_equal(m.num("yl"), [[0, 3]])
    assert m.num("p.area()").item() == pytest.approx(5.0)


def test_measures_with_a_hole(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]});")
    assert m.num("area(p)").item() == pytest.approx(15.0)
    np.testing.assert_allclose(m.num("area(p, [1 2])"), [[16], [1]])
    assert m.num("perimeter(p)").item() == pytest.approx(20.0)
    assert m.num("perimeter(p, 2)").item() == pytest.approx(4.0)
    m.run("[cx, cy] = centroid(p);")
    assert m.num("cx").item() == pytest.approx((16 * 2 - 1 * 1.5) / 15)
    assert m.num("cy").item() == pytest.approx((16 * 2 - 1 * 1.5) / 15)
    m.run("[hx, hy] = centroid(p, 2);")
    assert (m.num("hx").item(), m.num("hy").item()) == pytest.approx((1.5, 1.5))
    np.testing.assert_array_equal(m.num("numsides(p, [1 2])"), [[4], [4]])
    assert m.num("numboundaries(p)").item() == 2


def test_isinterior(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]});")
    m.run("[in, on] = isinterior(p, [0.5 1.5 4 5 1], [0.5 1.5 2 2 1.5]);")
    np.testing.assert_array_equal(m.num("in").ravel(), [True, False, True, False, True])
    np.testing.assert_array_equal(m.num("on").ravel(), [False, False, True, False, True])
    assert m.num("in").shape == (5, 1)
    np.testing.assert_array_equal(m.num("isinterior(p, [3 3; 1.5 1.5])").ravel(), [True, False])


def test_nearestvertex(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]});")
    m.run("[vid, bid, ind] = nearestvertex(p, [3.9 1.1], [0.2 1.2]);")
    V = m.num("p.Vertices")
    vid = m.num("vid").astype(int).ravel()
    np.testing.assert_array_equal(V[vid[0] - 1], [4, 0])
    np.testing.assert_array_equal(V[vid[1] - 1], [1, 1])
    np.testing.assert_array_equal(m.num("bid").ravel(), [1, 2])
    B = boundaries(V)
    ind = m.num("ind").astype(int).ravel()
    np.testing.assert_array_equal(B[1][ind[1] - 1], [1, 1])


def test_boundary_regions_holes(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1], [10 11 11]}, {[0 0 4 4], [1 1 2 2], [0 0 1]});")
    assert m.num("p.NumRegions").item() == 2 and m.num("p.NumHoles").item() == 1
    m.run("[x, y] = boundary(p, 1);")
    x, y = m.num("x").ravel(), m.num("y").ravel()
    assert x.size == 5 and x[0] == x[-1] and y[0] == y[-1]  # closed
    m.run("[x, y] = boundary(p);")
    assert np.isnan(m.num("x")).sum() == 2 and m.num("x").shape == (5 + 1 + 5 + 1 + 4, 1)
    m.run("R = regions(p);")
    assert m.num("numel(R)").item() == 2 and tuple(m.num("size(R)").ravel()) == (2, 1)
    areas = sorted(m.num("area(R)").ravel())
    assert areas == pytest.approx([0.5, 15.0])
    m.run("H = holes(p);")
    assert m.num("numel(H)").item() == 1 and m.num("area(H)").item() == pytest.approx(1.0)
    assert m.num("H.NumHoles").item() == 0
    m.run("q = rmholes(p);")
    assert m.num("q.NumHoles").item() == 0 and m.num("area(q)").item() == pytest.approx(16.5)


def test_addboundary_rmboundary(m):
    m.run("p = polyshape([0 4 4 0], [0 0 4 4]); p = addboundary(p, [1 2 2 1], [1 1 2 2]);")
    assert m.num("p.NumHoles").item() == 1 and m.num("area(p)").item() == pytest.approx(15.0)
    m.run("p = addboundary(p, [6 7 7], [0 0 1]);")
    assert m.num("p.NumRegions").item() == 2
    m.run("q = rmboundary(p, 2);")
    assert m.num("q.NumHoles").item() == 0
    assert m.num("area(q)").item() == pytest.approx(16.5)


def test_sortboundaries(m):
    m.run("p = polyshape({[0 1 1], [5 8 8 5], [10 12 12 10]}, {[0 0 1], [0 0 3 3], [0 0 2 2]});")
    m.run("[q, I] = sortboundaries(p, 'area', 'descend');")
    np.testing.assert_allclose(m.num("area(q, 1:3)").ravel(), [9, 4, 0.5])
    np.testing.assert_array_equal(m.num("I").ravel(), [2, 3, 1])
    m.run("q = sortboundaries(p, 'numsides', 'ascend');")
    np.testing.assert_array_equal(m.num("numsides(q, 1:3)").ravel(), [3, 4, 4])
    m.run("q = sortboundaries(p, 'centroid', 'ascend', 'ReferencePoint', [11 1]);")
    np.testing.assert_allclose(m.num("area(q, 1)"), 4.0)


def test_set_vertices(m):
    m.run("p = polyshape([0 1 1], [0 0 1]); p.Vertices = [0 0; 3 0; 3 3; 0 3];")
    assert m.num("area(p)").item() == pytest.approx(9.0)
    with pytest.raises(MatlabError, match="read-only"):
        m.run("p.NumHoles = 2;")


# -- transforms -----------------------------------------------------------------------------------------------------

def test_translate_rotate_scale(m):
    m.run(f"p = polyshape({mat(L_SHAPE)});")
    m.run("t = translate(p, 2, -1); [cx, cy] = centroid(t); [c0x, c0y] = centroid(p);")
    assert m.num("cx - c0x").item() == pytest.approx(2) and m.num("cy - c0y").item() == pytest.approx(-1)
    assert m.num("isequal(translate(p, [2 -1]), t)").item()
    m.run("r = rotate(p, 90);")
    expected = L_SHAPE @ np.array([[0, -1], [1, 0]]).T
    got = m.num("r.Vertices")
    assert {tuple(np.round(v, 12)) for v in got} == {tuple(np.round(v, 12)) for v in expected}
    assert m.num("area(r)").item() == pytest.approx(5.0)
    m.run("r2 = rotate(p, 180, [1 1]); [rx, ry] = boundingbox(r2);")
    np.testing.assert_allclose(m.num("rx"), [[-1, 2]], atol=1e-12)
    m.run("s = scale(p, [2 3]);")
    assert m.num("area(s)").item() == pytest.approx(30.0)
    m.run("f = scale(p, [-1 1]);")  # a mirror image stays clockwise
    assert shoelace(m.num("f.Vertices")) < 0
    assert m.num("area(f)").item() == pytest.approx(5.0)


# -- set operations ----------------------------------------------------------------------------------------------------

SQ = "polyshape([{x} {x}+2 {x}+2 {x}], [{y} {y} {y}+2 {y}+2])"


def sq(x, y):
    return SQ.format(x=x, y=y)


def test_set_operations_on_overlapping_squares(m):
    m.run(f"a = {sq(0, 0)}; b = {sq(1, 1)};")
    assert m.num("area(union(a, b))").item() == pytest.approx(7.0)
    assert m.num("area(intersect(a, b))").item() == pytest.approx(1.0)
    assert m.num("area(subtract(a, b))").item() == pytest.approx(3.0)
    assert m.num("area(xor(a, b))").item() == pytest.approx(6.0)
    assert m.num("numsides(union(a, b))").item() == 8
    assert m.num("numsides(intersect(a, b))").item() == 4
    assert m.num("a.union(b).NumRegions").item() == 1


def test_shared_edge_and_touching_corner(m):
    m.run(f"a = {sq(0, 0)}; b = {sq(2, 0)}; c = {sq(2, 2)};")
    m.run("u = union(a, b);")
    assert m.num("u.NumRegions").item() == 1 and m.num("numsides(u)").item() == 4  # the shared edge vanishes
    assert m.num("area(u)").item() == pytest.approx(8.0)
    assert m.num("area(intersect(a, b))").item() == 0
    m.run("v = union(a, c);")  # only a corner in common: two regions
    assert m.num("v.NumRegions").item() == 2 and m.num("area(v)").item() == pytest.approx(8.0)
    m.run("w = subtract(a, b);")
    assert m.num("area(w)").item() == pytest.approx(4.0)


def test_hole_touching_the_outer_boundary(m):
    m.run(f"a = {sq(0, 0)}; t = polyshape([0 1 0.5], [0 0.5 1]);")
    m.run("d = subtract(a, t);")
    assert m.num("d.NumRegions").item() == 1 and m.num("d.NumHoles").item() == 1
    assert m.num("area(d)").item() == pytest.approx(4 - 0.375)


def test_subtract_makes_a_hole_and_union_fills_it(m):
    m.run("a = polyshape([0 6 6 0], [0 0 6 6]); b = polyshape([2 4 4 2], [2 2 4 4]);")
    m.run("d = subtract(a, b);")
    assert m.num("d.NumHoles").item() == 1 and m.num("area(d)").item() == pytest.approx(32.0)
    m.run("u = union(d, b);")
    assert m.num("u.NumHoles").item() == 0 and m.num("area(u)").item() == pytest.approx(36.0)
    m.run("i = intersect(d, polyshape([3 7 7 3], [3 3 7 7]));")  # an L-shaped piece of the frame
    assert m.num("area(i)").item() == pytest.approx(9 - 1)


def test_union_of_an_array_and_ids(m):
    m.run(f"P = [{sq(0, 0)}, {sq(1, 0)}, {sq(5, 5)}];")
    m.run("u = union(P);")
    assert m.num("area(u)").item() == pytest.approx(6 + 4)
    assert m.num("u.NumRegions").item() == 2
    m.run(f"[q, sid, vid] = union({sq(0, 0)}, {sq(1, 1)});")
    sid, vid = m.num("sid").ravel(), m.num("vid").ravel()
    V = m.num("q.Vertices")
    assert sid.shape == (V.shape[0],)
    assert sorted(sid.tolist()) == [0, 0, 1, 1, 1, 2, 2, 2]
    assert {tuple(V[k]) for k in range(8) if sid[k] == 0} == {(2, 1), (1, 2)}
    m.run(f"A = {sq(0, 0)};")
    for k in range(8):
        if sid[k] == 1:
            np.testing.assert_array_equal(m.num(f"A.Vertices({int(vid[k])}, :)").ravel(), V[k])


def _random_polygon(rng, kind):
    n = int(rng.integers(3, 9))
    if kind == 0:
        return rng.integers(0, 5, size=(n, 2)).astype(float)  # many shared vertices and collinear edges
    if kind == 1:
        return np.round(rng.random((n, 2)) * 4, 3)
    ang = np.sort(rng.random(n)) * 2 * np.pi
    r = 1 + rng.random(n)
    return np.round(np.c_[2 + r * np.cos(ang), 2 + r * np.sin(ang)], 6)


@pytest.mark.parametrize("seed", range(4))
def test_set_operations_against_point_sampling(m, seed):
    """isinterior of the result agrees with the Boolean of the inputs at random points off the boundaries."""
    rng = np.random.default_rng(100 + seed)
    m.run("warning('off', 'MATLAB:polyshape:repairedBySimplify'); warning('off', 'MATLAB:polyshape:boundary3Points');")
    Q = rng.random((1500, 2)) * 4.4 - 0.2
    m.interp.set_var("Q", Q)
    for trial in range(12):
        A, B = _random_polygon(rng, trial % 3), _random_polygon(rng, (trial + 1) % 3)
        m.run(f"a = polyshape({mat(A)}); b = polyshape({mat(B)});")
        if m.num("numboundaries(a)").item() == 0 or m.num("numboundaries(b)").item() == 0:
            continue
        m.run("[ia, oa] = isinterior(a, Q); [ib, ob] = isinterior(b, Q);")
        ia, ib = m.num("ia").ravel(), m.num("ib").ravel()
        far = ~(m.num("oa").ravel() | m.num("ob").ravel())
        # also keep away from the boundaries by a margin, so rounding cannot decide
        m.run("[xa, ya] = boundary(a); [xb, yb] = boundary(b);")
        edges = []
        for nm in ("a", "b"):
            x, y = m.num(f"x{nm}").ravel(), m.num(f"y{nm}").ravel()
            pts = np.c_[x, y]
            for k in range(len(pts) - 1):
                if not (np.isnan(pts[k]).any() or np.isnan(pts[k + 1]).any()):
                    edges.append((pts[k], pts[k + 1]))
        P0 = np.array([e[0] for e in edges])
        P1 = np.array([e[1] for e in edges])
        d = P1 - P0
        t = np.clip(np.einsum("qkj,kj->qk", Q[:, None] - P0[None], d) / np.einsum("kj,kj->k", d, d), 0, 1)
        dist = np.min(np.linalg.norm(Q[:, None] - (P0[None] + t[..., None] * d[None]), axis=2), axis=1)
        far &= dist > 1e-7
        expect = {"union": ia | ib, "intersect": ia & ib, "subtract": ia & ~ib, "xor": ia ^ ib}
        for op, want in expect.items():
            m.run(f"r = {op}(a, b); ir = isinterior(r, Q);")
            got = m.num("ir").ravel()
            assert np.array_equal(got[far], want[far]), (op, A.tolist(), B.tolist())
            # the area agrees with the fraction of sampled points (loosely) and with the boundaries
            V = m.num("r.Vertices")
            hole = m.num("ishole(r)").ravel() if V.size else np.zeros(0, dtype=bool)
            total = sum(-abs(shoelace(b)) if h else abs(shoelace(b)) for b, h in zip(boundaries(V), hole))
            assert m.num("area(r)").item() == pytest.approx(total, abs=1e-9)
            assert m.num("issimplified(r)").item()


def test_intersect_with_a_polyline(m):
    m.run(f"a = {sq(0, 0)}; [in, out] = intersect(a, [-1 1; 3 1]);")
    np.testing.assert_allclose(m.num("in"), [[0, 1], [2, 1]])
    out = m.num("out")
    assert np.isnan(out[2]).all()
    np.testing.assert_allclose(out[[0, 1, 3, 4]], [[-1, 1], [0, 1], [2, 1], [3, 1]])


def test_overlaps(m):
    m.run(f"a = {sq(0, 0)}; b = {sq(1, 1)}; c = {sq(2, 2)}; d = {sq(5, 5)};")
    assert m.num("overlaps(a, b)").item()
    assert m.num("overlaps(a, c)").item()  # touching at a corner
    assert not m.num("overlaps(a, d)").item()
    M = m.num("overlaps([a b d])")
    np.testing.assert_array_equal(M, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])


# -- constructions -----------------------------------------------------------------------------------------------------

def test_convhull(m):
    m.run(f"p = polyshape({mat(L_SHAPE)}); h = convhull(p);")
    hull = np.array([[0, 0], [3, 0], [3, 1], [1, 3], [0, 3]], dtype=float)
    assert m.num("area(h)").item() == pytest.approx(abs(shoelace(hull)))
    assert m.num("numsides(h)").item() == 5


def test_polybuffer(m):
    d = 0.25
    m.run(f"p = {sq(0, 0)}; b = polybuffer(p, {d});")
    n = 64
    disk = 0.5 * n * d * d * np.sin(2 * np.pi / n)  # the round corners are quarters of a 64-gon
    assert m.num("area(b)").item() == pytest.approx(4 + 4 * 2 * d + disk, rel=1e-9)
    m.run(f"s = polybuffer(p, -{d});")
    assert m.num("area(s)").item() == pytest.approx((2 - 2 * d) ** 2, rel=1e-9)
    m.run(f"q = polybuffer(p, {d}, 'JointType', 'miter');")
    assert m.num("area(q)").item() == pytest.approx((2 + 2 * d) ** 2, rel=1e-9)
    m.run(f"q = polybuffer(p, {d}, 'JointType', 'square');")
    h = d * (np.sqrt(2) - 1)  # each corner of the outer square is cut off at distance d from the vertex
    assert m.num("area(q)").item() == pytest.approx((2 + 2 * d) ** 2 - 4 * h * h, rel=1e-9)
    m.run("l = polybuffer([0 0; 4 0], 'lines', 1);")
    assert m.num("area(l)").item() == pytest.approx(8 + 0.5 * n * np.sin(2 * np.pi / n), rel=1e-9)
    with pytest.raises(MatlabError, match="NoSuchOption"):
        m.run("polybuffer(p, 1, 'NoSuchOption', 2);")


def test_triangulation_covers_the_area(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]}); T = triangulation(p);")
    assert m.str("class(T)") == "triangulation"
    P, C = m.num("T.Points"), m.num("T.ConnectivityList").astype(int) - 1
    a, b, c = P[C[:, 0]], P[C[:, 1]], P[C[:, 2]]
    tri = 0.5 * np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    assert tri.sum() == pytest.approx(15.0)
    assert P.shape == (8, 2)


def test_nsidedpoly(m):
    m.run("p = nsidedpoly(6);")
    assert m.num("area(p)").item() == pytest.approx(6 / 2 * np.sin(2 * np.pi / 6))
    V = m.num("p.Vertices")
    assert np.allclose(np.hypot(V[:, 0], V[:, 1]), 1.0)
    assert shoelace(V) < 0
    assert V[0, 1] == pytest.approx(V[-1, 1])  # a flat bottom side
    m.run("q = nsidedpoly(4, 'Center', [5 -1], 'SideLength', 3);")
    assert m.num("area(q)").item() == pytest.approx(9.0)
    m.run("[cx, cy] = centroid(q);")
    assert (m.num("cx").item(), m.num("cy").item()) == pytest.approx((5, -1))
    m.run("r = nsidedpoly(8, 'Radius', 2);")
    assert m.num("perimeter(r)").item() == pytest.approx(8 * 2 * 2 * np.sin(np.pi / 8))
    with pytest.raises(MatlabError):
        m.run("nsidedpoly(2);")
    with pytest.raises(MatlabError):
        m.run("nsidedpoly(5, 'Radius', 1, 'SideLength', 1);")


def test_turningdist(m):
    m.run(f"a = {sq(0, 0)}; b = rotate(scale(a, 3), 30); c = nsidedpoly(3);")
    assert m.num("turningdist(a, b)").item() == pytest.approx(0, abs=1e-9)
    assert m.num("turningdist(a, c)").item() > 0.1


# -- arrays, equality, plotting ----------------------------------------------------------------------------------

def test_object_arrays(m):
    m.run(f"a = {sq(0, 0)}; b = nsidedpoly(3); P = [a b a];")
    np.testing.assert_array_equal(m.num("size(P)"), [[1, 3]])
    assert m.str("class(P)") == "polyshape"
    np.testing.assert_allclose(m.num("area(P)"), [[4, 3 * np.sqrt(3) / 4, 4]])
    assert m.num("isequal(P(3), a)").item()
    m.run("P(5) = b;")
    np.testing.assert_array_equal(m.num("size(P)"), [[1, 5]])
    assert m.num("numboundaries(P(4))").item() == 0  # filled with empty polyshapes
    m.run("P(2) = [];")
    assert m.num("numel(P)").item() == 4
    m.run("Q = P'; n = 0; for k = 1:numel(P), n = n + P(k).NumRegions; end")
    np.testing.assert_array_equal(m.num("size(Q)"), [[4, 1]])
    assert m.num("n").item() == 3
    out = m.run("P")
    assert "1×4 polyshape array with properties:" in out
    m.run("T = translate(P, 1, 1);")
    np.testing.assert_allclose(m.num("area(T)"), m.num("area(P)"))
    m.run("[xl, yl] = boundingbox(T);")
    np.testing.assert_allclose(m.num("xl"), [[1 - np.sqrt(3) / 2, 3]])  # the triangle reaches left of 1


def test_isequal(m):
    m.run(f"a = {sq(0, 0)}; b = {sq(0, 0)}; c = {sq(1, 0)};")
    assert m.num("isequal(a, b)").item() and not m.num("isequal(a, c)").item()
    m.run("h1 = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]}); h2 = h1;")
    assert m.num("isequal(h1, h2)").item()  # NaN separators compare equal
    assert not m.num("isequal(a, 3)").item()


def test_plot_returns_a_polygon(m):
    m.run("p = polyshape({[0 4 4 0], [1 2 2 1]}, {[0 0 4 4], [1 1 2 2]}); h = plot(p, 'FaceColor', 'r');")
    assert m.str("class(h)") == "matlab.graphics.primitive.Polygon"
    assert m.str("h.Type") == "polygon"
    np.testing.assert_allclose(m.num("h.FaceColor"), [[1, 0, 0]])
    assert m.num("h.FaceAlpha").item() == pytest.approx(0.35)
    assert m.num("isequal(h.Shape, p)").item()
    m.run("h.FaceAlpha = 0.8;")
    assert m.num("h.FaceAlpha").item() == pytest.approx(0.8)
    m.run("hs = plot([p translate(p, 5, 0)]);")
    assert m.num("numel(hs)").item() == 2
    with pytest.raises(MatlabError):
        m.run("plot(p, 'NoSuchProperty', 1);")
    m.run("close all")
