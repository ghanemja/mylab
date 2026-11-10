"""N-dimensional arrays."""

import numpy as np
import pytest

from mylab import MatlabError


def test_construct_and_size(m):
    m.run("A = zeros(2, 3, 4);")
    assert m.num("size(A)").tolist() == [[2, 3, 4]]
    assert m.num("ndims(A)").item() == 3
    assert m.num("numel(A)").item() == 24
    assert m.num("size(A, 3)").item() == 4
    assert m.num("size(A, 5)").item() == 1
    m.run("[r, c] = size(A);")
    assert (m.val("r").item(), m.val("c").item()) == (2, 12)
    m.run("[r, c, p] = size(A);")
    assert m.val("p").item() == 4
    assert m.num("size(ones(2, 3, 1))").tolist() == [[2, 3]]
    assert m.num("size(rand([2 2 2]))").tolist() == [[2, 2, 2]]
    assert not m.num("ismatrix(A)").item()


def test_page_indexing_and_assignment(m):
    m.run("A = zeros(2, 2); A(:, :, 2) = [1 2; 3 4];")
    assert m.num("size(A)").tolist() == [[2, 2, 2]]
    np.testing.assert_array_equal(m.num("A(:, :, 2)"), [[1, 2], [3, 4]])
    assert m.num("A(2, 1, 2)").item() == 3
    assert m.num("A(end)").item() == 4
    assert m.num("A(8)").item() == 4  # linear, column-major
    np.testing.assert_array_equal(m.num("A(2, :)"), [[0, 0, 3, 4]])  # trailing dims fold
    np.testing.assert_array_equal(m.num("squeeze(A(1, 2, :))"), [[0], [2]])
    m.run("B = reshape(1:24, 2, 3, 4);")
    assert m.num("B(2, 3, 4)").item() == 24
    assert m.num("B(1, end, end)").item() == 23
    m.run("B(:, :, 1) = [];")
    assert m.num("size(B)").tolist() == [[2, 3, 3]]
    m.run("C = []; C(2, 2, 3) = 7;")
    assert m.num("size(C)").tolist() == [[2, 2, 3]]


def test_cat_permute_squeeze(m):
    m.run("A = cat(3, [1 2], [3 4], [5 6]);")
    assert m.num("size(A)").tolist() == [[1, 2, 3]]
    m.run("P = permute(A, [3 2 1]);")
    np.testing.assert_array_equal(m.val("P"), [[1, 2], [3, 4], [5, 6]])
    m.run("Q = ipermute(P, [3 2 1]);")
    assert m.num("isequal(Q, A)").item()
    assert m.num("size(squeeze(ones(1, 1, 5)))").tolist() == [[5, 1]]
    assert m.num("size(repmat([1 2], [2 1 3]))").tolist() == [[2, 2, 3]]


def test_reductions_along_any_dim(m):
    m.run("A = reshape(1:8, 2, 2, 2);")
    np.testing.assert_array_equal(m.num("sum(A, 3)"), [[6, 10], [8, 12]])
    np.testing.assert_array_equal(m.num("sum(A, 'all')"), [[36]])
    assert m.num("size(sum(A))").tolist() == [[1, 2, 2]]
    np.testing.assert_array_equal(m.num("max(A, [], 3)"), [[5, 7], [6, 8]])
    np.testing.assert_array_equal(m.num("mean(A, 3)"), [[3, 5], [4, 6]])
    assert m.num("size(cumsum(A, 3))").tolist() == [[2, 2, 2]]
    assert m.num("any(A(:) > 6)").item()


def test_elementwise_and_broadcast(m):
    m.run("A = ones(2, 2, 3); B = A .* reshape(1:3, 1, 1, 3);")
    np.testing.assert_array_equal(m.num("B(:, :, 3)"), [[3, 3], [3, 3]])
    m.run("C = A + [10 20];")
    assert m.num("C(1, 2, 3)").item() == 21
    with pytest.raises(MatlabError, match="permute"):
        m.run("A'")
    with pytest.raises(MatlabError, match="2-D"):
        m.run("A * A")


def test_display_pages(m):
    out = m.run("A = cat(3, [1 2], [3 4])")
    assert "A(:,:,1) =" in out and "A(:,:,2) =" in out


def test_for_over_nd_iterates_columns(m):
    m.run("n = 0; for col = ones(2, 3, 4), n = n + 1; end")
    assert m.val("n").item() == 12


def test_nd_cells_and_grids(m):
    m.run("c = cell(2, 2, 2); c{2, 2, 2} = 'x';")
    assert m.num("size(c)").tolist() == [[2, 2, 2]]
    assert m.str("c{8}") == "x"
    m.run("[X, Y, Z] = ndgrid(1:2, 1:3, 1:4);")
    assert m.num("size(Z)").tolist() == [[2, 3, 4]]
    m.run("[X, Y, Z] = meshgrid(1:2, 1:3, 1:4);")
    assert m.num("size(X)").tolist() == [[3, 2, 4]]
    assert m.num("size(pagemtimes(ones(2,3,4), ones(3,2,4)))").tolist() == [[2, 2, 4]]
