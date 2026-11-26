"""Reductions and array helpers: rms, topkrows, discretize, histograms, grouping, moving windows,
outliers, sorting options, normalization and smoothing. Expected values are computed by hand."""

import pytest

from mylab.errors import MatlabError


def ok(m, code: str) -> None:
    m.run(code)


def test_rms(m):
    ok(m, """
        assert(abs(rms([3 4]) - sqrt(12.5)) < 1e-15 && isequal(rms([3 4; 3 4]), [3 4]));
        assert(norm(rms([1 2; 3 4], 2) - [sqrt(2.5); sqrt(12.5)]) < 1e-15 && ...
            abs(rms([1 2; 3 4], 'all') - sqrt(7.5)) < 1e-15);
        assert(isnan(rms([1 NaN 3])) && abs(rms([1 NaN 3], 'omitnan') - sqrt(5)) < 1e-15);
        assert(abs(rms([3+4i 0]) - sqrt(12.5)) < 1e-15 && isa(rms(single([1 2])), 'single'));
        assert(isequal(size(rms(ones(2, 3, 4), [1 3])), [1 3]));
    """)


def test_topkrows_and_issortedrows(m):
    ok(m, """
        X = [1 2; 3 4; 2 9; 3 1];
        assert(isequal(topkrows(X, 1), [3 4]));
        [B, I] = topkrows(X, 2); assert(isequal(B, [3 4; 3 1]) && isequal(I, [2; 4]));
        assert(isequal(topkrows(X, 2, 2), [2 9; 3 4]) && isequal(topkrows(X, 2, [1 -2]), [3 1; 3 4]));
        assert(isequal(topkrows(X, 2, 1, 'ascend'), [1 2; 2 9]) && isequal(topkrows(X, 10), sortrows(X, 'descend')));
        assert(isequal(topkrows([-5 1; 2 2], 1, 1, 'ComparisonMethod', 'abs'), [-5 1]));
        assert(issortedrows([1 2; 1 3; 2 1]) && ...
            ~issortedrows([1 2; 1 1; 2 1]) && issortedrows([1 2; 1 1; 2 1], [1 -2]));
        assert(issortedrows([3 1; 2 1], 'descend') && ~issortedrows([1 2; 2 2], 2, 'strictascend'));
        assert(issortedrows([3 1; 2 1], 'monotonic') && issortedrows([1 5; 2 4], 1, 'strictascend'));
    """)


def test_discretize(m):
    ok(m, """
        assert(isequal(discretize([1 5], [0 2 6]), [1 2]));
        Y = discretize([1 2 3 4 5 6 NaN 7], [1 3 5 6]); assert(isequaln(Y, [1 1 2 2 3 3 NaN NaN]));
        assert(isequal(discretize([1 3 5], [1 3 5], 'IncludedEdge', 'right'), [1 1 2]));
        assert(isequal(discretize([1 2 6], [0 2 4 8], [10 20 30]), [10 20 30]));
        assert(isequal(discretize([1 5], [0 4 8], {'low', 'high'}), {'low', 'high'}));
        c = discretize([1 5 9], [0 4 8 12], 'categorical');
        assert(iscategorical(c) && isequal(cellstr(c), {'[0, 4)', '[4, 8)', '[8, 12]'}));
        c = discretize([1 5 9], [0 4 8 12], 'categorical', {'a', 'b', 'c'});
        assert(isequal(cellstr(c), {'a', 'b', 'c'}));
        [Y, E] = discretize([1 4 7 10], 3);
        assert(numel(E) == 4 && E(1) <= 1 && E(end) >= 10 && ...
            all(abs(diff(E) - (E(2) - E(1))) < 1e-12) && isequal(Y, [1 1 2 3]));
        assert(isequal(size(discretize(magic(3), [0 5 10])), [3 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("discretize([1 2], [3 2 1])")
    with pytest.raises(MatlabError):
        m.run("discretize([1 2], [0 5 10], [1 2 3])")


def test_histcounts_options(m):
    ok(m, """
        [N, e] = histcounts([1 2 2 3 5], 0:2:6); assert(isequal(N, [1 3 1]) && isequal(e, 0:2:6));
        assert(isequal(histcounts([1 2 2 3 5], 'BinMethod', 'integers'), [1 2 1 0 1]));
        [N, e] = histcounts(1:100, 'BinMethod', 'sturges'); assert(numel(N) == 8 && sum(N) == 100);
        [N, e] = histcounts(1:100, 'BinMethod', 'sqrt'); assert(numel(N) == 10);
        [N, e] = histcounts([1 2 2 3 5], 'BinWidth', 2); assert(isequal(e, [0 2 4 6]) && isequal(N, [1 3 1]));
        assert(isequal(histcounts([1 2 2 3], [0 2 4], 'Normalization', 'probability'), [0.25 0.75]));
        assert(isequal(histcounts([1 2 2 3], [0 2 4], 'Normalization', 'percentage'), [25 75]));
        assert(isequal(histcounts([1 2 2 3], [0 2 4], 'Normalization', 'cdf'), [0.25 1]));
        assert(isequal(histcounts([1 2 2 3], [0 2 4], 'Normalization', 'pdf'), [0.125 0.375]));
        [N, e, bin] = histcounts([1 5 9], [0 4 8 12]); assert(isequal(bin, [1 2 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("histcounts([1 2 3], 'BinMethod', 'nosuch')")


def test_histcounts2(m):
    ok(m, """
        [N, xe, ye] = histcounts2([1 2 2 3], [1 1 2 3], [0 2 4], [0 2 4]);
        assert(isequal(N, [1 0; 1 2]) && isequal(xe, [0 2 4]) && isequal(ye, [0 2 4]));
        [N, xe, ye, bx, by] = histcounts2([1 2 2 3 5], [1 1 2 3 5], 2);
        assert(sum(N(:)) == 5 && isequal(size(N), [2 2]) && isequal(bx, by));
        P = histcounts2([1 2 2 3], [1 1 2 3], [0 2 4], [0 2 4], 'Normalization', 'probability');
        assert(isequal(P, [0.25 0; 0.25 0.5]));
        N = histcounts2([1 9], [1 9], 'XBinEdges', [0 5 10], 'YBinEdges', [0 5 10]); assert(isequal(N, eye(2)));
    """)


def test_groupcounts(m):
    ok(m, """
        [B, BG, BP] = groupcounts([1; 2; 2; 3; 3; 3]);
        assert(isequal(B, [1; 2; 3]) && isequal(BG, [1; 2; 3]) && norm(BP - 100 * [1; 2; 3] / 6) < 1e-12);
        [B, BG] = groupcounts({'a'; 'b'; 'a'}); assert(isequal(B, [2; 1]) && isequal(BG, {'a'; 'b'}));
        [B, BG] = groupcounts({[1; 1; 2; 2], [1; 2; 2; 2]});
        assert(isequal(B, [1; 1; 2]) && isequal(BG{1}, [1; 1; 2]) && isequal(BG{2}, [1; 2; 2]));
        T = table([1; 1; 2], [5; 6; 7], 'VariableNames', {'g', 'v'}); G = groupcounts(T, 'g');
        assert(isequal(G.GroupCount, [2; 1]) && norm(G.Percent - [200/3; 100/3]) < 1e-12);
    """)


def test_moving_windows(m):
    ok(m, """
        assert(isequal(movprod([1 2 3 4], 2), [1 2 6 12]) && isequal(movmad([1 2 9 4 5], 3), [0.5 1 2 1 0.5]));
        assert(isequal(movfun(@max, [1 3 2 5 4], 3), [3 3 5 5 5]));
        assert(isequaln(movfun(@(x) sum(x), [1 2 3 4], [1 0], 'Endpoints', 'fill'), [NaN 3 5 7]));
        assert(norm(movfun(@mean, [1 2 3 4], 3, 'Endpoints', 'periodic') - [7/3 2 3 8/3]) < 1e-14);
        assert(isequal(movmean([1 2 3 4 5], 3), [1.5 2 3 4 4.5]) && isequal(movsum([1 2 3 4], 2), [1 3 5 7]));
        assert(isequal(movmedian([1 9 2 8 3], 3), [5 2 8 3 5.5]) && ...
            isequal(movmean([1 2 3 4 5], 3, 'Endpoints', 'discard'), [2 3 4]));
        assert(isequaln(movmean([1 2 3 4 5], 3, 'Endpoints', 'fill'), [NaN 2 3 4 NaN]));
        % sample points 0 1 3 4 with a window of 2: [t-1, t+1)
        assert(isequal(movmean([1 2 3 4], 2, 'SamplePoints', [0 1 3 4]), [1 1.5 3 3.5]));
        assert(isequal(movmean([1 NaN 3], 3, 'omitnan'), [1 2 3]));
    """)
    with pytest.raises(MatlabError):
        m.run("movmean([1 2 3], 3, 'NoSuchOption', 1)")


def test_outliers(m):
    ok(m, """
        [tf, L, U, C] = isoutlier([1 2 3 100]);
        % median 2.5, scaled MAD 1.4826 * 1 -> thresholds 2.5 -+ 3 * 1.4826
        assert(isequal(tf, logical([0 0 0 1])) && C == 2.5 && ...
            abs(U - (2.5 + 3 * 1.482602218505602)) < 1e-12 && abs(L + U - 5) < 1e-12);
        assert(~any(isoutlier([1 2 3 4 100], 'mean')) && ...
            isequal(find(isoutlier([1 2 3 4 5 6 7 8 9 100], 'quartiles')), 10));
        assert(isequal(find(isoutlier([1 2 3 4 5 6 7 8 9 100], 'grubbs')), 10));
        x = [1 2 3 4 5 6 7 8 9 100 -50 4 5 6 3 4 5 6 5 4]; assert(isequal(find(isoutlier(x, 'gesd')), [10 11]));
        assert(isequal(find(isoutlier(1:10, 'percentiles', [10 90])), [1 10]));
        assert(isequal(find(isoutlier([1 1 1 9 1 1 1 1 20 1], 'movmedian', 5)), [4 9]));
        assert(isequal(isoutlier([1 2 3 100; 1 1 1 1], 2), logical([0 0 0 1; 0 0 0 0])));
        assert(~any(isoutlier([1 2 3 100], 'ThresholdFactor', 100)));
        [B, TF] = rmoutliers([1 2 3 100 2]); assert(isequal(B, [1 2 3 2]) && isequal(TF, logical([0 0 0 1 0])));
        assert(isequal(rmoutliers([1 2; 2 3; 3 100; 2 2; 1 3]), [1 2; 2 3; 2 2; 1 3]));
        [B, TF, L, U, C] = filloutliers([1 2 3 100 2], 'center'); assert(isequal(B, [1 2 3 2 2]) && C == 2);
        assert(isequal(filloutliers([1 2 3 100 2], 'linear'), [1 2 3 2.5 2]) && ...
            isequal(filloutliers([1 2 3 100 2], 0), [1 2 3 0 2]));
        assert(isequal(filloutliers([1 2 3 100 2], 'previous'), [1 2 3 3 2]));
        c = filloutliers([1 2 3 100 2], 'clip'); assert(abs(c(4) - U) < 1e-12);
    """)
    with pytest.raises(MatlabError):
        m.run("isoutlier([1 2 3], 'nosuchmethod')")


def test_sort_options(m):
    ok(m, """
        assert(isequal(sort([-3 1 -2], 'ComparisonMethod', 'abs'), [1 -2 -3]));
        assert(isequaln(sort([3 NaN 1], 'MissingPlacement', 'first'), [NaN 1 3]));
        [s, i] = sort([3 NaN 1], 'descend', 'MissingPlacement', 'last');
        assert(isequaln(s, [3 1 NaN]) && isequal(i, [1 3 2]));
        assert(isequal(sort([1+2i 2 -3], 'ComparisonMethod', 'real'), [-3 1+2i 2]));
        assert(isequal(sort([3 1 2; 9 7 8], 2, 'descend'), [3 2 1; 9 8 7]) && ...
            isequal(sort([2 -2], 'ComparisonMethod', 'abs'), [2 -2]));
        assert(isequal(sortrows([1 2; 1 1; 0 5], [1 -2]), [0 5; 1 2; 1 1]));
        assert(isequal(sortrows([1 2; 1 1; 0 5], [1 2], {'ascend', 'descend'}), [0 5; 1 2; 1 1]));
        assert(isequal(sortrows([1 2; 1 1; 0 5], 1, 'descend'), [1 2; 1 1; 0 5]));
        assert(isequal(sortrows([-3 1; 2 2; 1 3], 'ComparisonMethod', 'abs'), [1 3; 2 2; -3 1]));
        [B, I] = sortrows([3 1; 1 2; 2 0]); assert(isequal(I, [2; 3; 1]));
        assert(isequal(maxk([-5 1 3], 2, 'ComparisonMethod', 'abs'), [-5 3]) && isequal(mink([5 1 9 3], 2), [1 3]));
        [v, i] = maxk([4 9 2 9], 2); assert(isequal(v, [9 9]) && isequal(i, [2 4]));
    """)
    with pytest.raises(MatlabError):
        m.run("sort([1 2], 'ComparisonMethod', 'sideways')")


def test_max_min_options(m):
    ok(m, """
        [v, i] = max([1 5; 7 2], [], 'all'); assert(v == 7 && i == 2);
        [v, i] = max([1 5; 7 2], [], 2, 'linear'); assert(isequal(v, [5; 7]) && isequal(i, [3; 2]));
        assert(max([-5 3], [], 'ComparisonMethod', 'abs') == -5 && min([-5 3 1], [], 'ComparisonMethod', 'abs') == 1);
        A = cat(3, [1 2; 3 4], [8 6; 7 5]); [v, i] = max(A, [], [1 2]);
        assert(isequal(v(:)', [4 8]) && isequal(i(:)', [4 5]));
        assert(max([1+1i 1-1i]) == 1+1i && max(-4, 3, 'ComparisonMethod', 'abs') == -4);
        assert(isnan(max([1 NaN 3], [], 'includenan')) && max([1 NaN 3]) == 3);
        [lo, hi] = bounds([3 1 2; 9 7 8], 2); assert(isequal(lo, [1; 7]) && isequal(hi, [3; 9]));
    """)


def test_normalize_methods_and_outputs(m):
    ok(m, """
        assert(isequal(normalize([1 2 3]), [-1 0 1]) && isequal(normalize([1 2 3], 'range'), [0 0.5 1]));
        [N, C, S] = normalize([1 2 3]); assert(C == 2 && S == 1);
        assert(norm(normalize([1 2 3 10], 'medianiqr') - ([1 2 3 10] - 2.5) / 5) < 1e-14);   % median 2.5, iqr 5
        assert(isequal(normalize([1 2 3], 'center', 'median', 'scale', 'mad'), [-1 0 1]));
        [N, C, S] = normalize([1 2; 3 6], 'range', [-1 1]);
        assert(isequal(N, [-1 -1; 1 1]) && isequal(C, [2 4]) && isequal(S, [1 2]));
        assert(isequal(normalize([3 4], 'norm'), [0.6 0.8]) && ...
            isequal(normalize([1 3; 2 2], 2, 'norm', 1), [0.25 0.75; 0.5 0.5]));
        assert(isequal(normalize([2 4 6], 'scale', 2), [1 2 3]) && isequal(normalize([2 4 6], 'center', 4), [-2 0 2]));
        assert(isequal(rescale([1 2 3], -1, 1), [-1 0 1]) && ...
            isequal(rescale([1 2 3], 'InputMin', 0, 'InputMax', 4), [0.25 0.5 0.75]));
    """)
    with pytest.raises(MatlabError):
        m.run("normalize([1 2 3], 'nosuchmethod')")


def test_smoothdata_methods(m):
    ok(m, """
        assert(isequal(smoothdata([1 2 3 4 5], 'movmean', 3), [1.5 2 3 4 4.5]));
        % local linear and quadratic regression reproduce straight lines and parabolas exactly
        x = 2 * (1:9) + 1;
        assert(norm(smoothdata(x, 'lowess', 5) - x) < 1e-10 && norm(smoothdata(x, 'rlowess', 5) - x) < 1e-10);
        q = (1:9).^2; assert(norm(smoothdata(q, 'loess', 5) - q) < 1e-9 && norm(smoothdata(q, 'sgolay', 5) - q) < 1e-9);
        y = [1 5 2 8 3]; g = smoothdata(y, 'gaussian', 3); assert(g(3) > 2 && g(3) < 8 && numel(g) == 5);
        r = smoothdata([1 2 3 100 5 6 7 8 9], 'rlowess', 7); assert(norm(r - (1:9)) < 1e-8);   % the spike is ignored
        p = smoothdata([1 2 3 100 5 6 7 8 9], 'lowess', 7);
        assert(p(4) > 10);                 % plain lowess is not robust
        [B, w] = smoothdata(1:20, 'movmean', 4); assert(w == 4);
        assert(isequal(smoothdata([1 2 3 4 5], 'SmoothingFactor', 0), [1 2 3 4 5]));
        assert(isequal(smoothdata([1 3 2 5], 'movmean', 2, 'SamplePoints', [0 1 3 4]), [1 2 2 3.5]));
    """)
    with pytest.raises(MatlabError):
        m.run("smoothdata([1 2 3], 'movmean', 3, 'SmoothingFactor', 0.5)")


def test_tolerance_sets_and_vector_norms(m):
    ok(m, """
        [tf, loc] = ismembertol([1 2.0000000000001 5], [2 1]);
        assert(isequal(tf, logical([1 1 0])) && isequal(loc, [2 1 0]));
        assert(isequal(ismembertol([1 2; 3 4.1], [3 4], 0.05, 'ByRows', true), logical([0; 1])));
        [tf, loc] = ismembertol(1, [1 1+1e-14 3], 'OutputAllIndices', true); assert(isequal(loc{1}, [1; 2]));
        assert(isequal(uniquetol([1 1+1e-13 2]), [1 2]));
        assert(isequal(vecnorm([3 4; 6 8]), [sqrt(45) sqrt(80)]) && isequal(vecnorm([3 4; 6 8], 2, 2), [5; 10]));
        assert(vecnorm([3 -4], 1) == 7 && ...
            isequal(vecnorm([3 -4; 1 2], Inf), [3 4]) && isequal(vecnorm([3 -4], 1, 1), [3 4]));
        V = vecnorm(cat(3, [3 0], [4 0]), 2, 3); assert(isequal(V, [5 0]) && isa(vecnorm(single([3 4])), 'single'));
        assert(isequal(cummax([1 3 2; 4 1 5], 2), [1 3 3; 4 4 5]) && isequal(cummin([3 1 2], 'reverse'), [1 1 2]));
        assert(isequal(trapz([1 2; 3 4], 2), [1.5; 3.5]) && isequal(cumtrapz([1 2 3]), [0 1.5 4]));
        [c, ia, ic] = unique([3 1 3 2], 'last');
        assert(isequal(c, [1 2 3]) && isequal(ia', [2 4 3]) && isequal(c(ic), [3 1 3 2]));
        [c, ia] = unique([1 2; 1 2; 3 4], 'rows', 'stable'); assert(isequal(c, [1 2; 3 4]) && isequal(ia', [1 3]));
    """)


def test_uniquetol_options(m):
    ok(m, """
        [c, ia, ic] = uniquetol([3 1 1+1e-13 2 3]);
        assert(isequal(c, [1 2 3]) && isequal(ia', [2 4 1]) && isequal(ic', [3 1 1 2 3]));
        assert(isequal(uniquetol([1 2; 1+1e-13 2; 3 4], 'ByRows', true), [1 2; 3 4]));
        [c, ia, ic] = uniquetol([1 1.04 2 1.02], 0.05); assert(isequal(c, [1 2]) && isequal(ic', [1 1 2 1]));
        [c, ia] = uniquetol([1 1+1e-13 2], 'OutputAllIndices', true);
        assert(isequal(ia{1}, [1; 2]) && isequal(ia{2}, 3));
        assert(isequal(uniquetol([10 10.5 20], 0.1, 'DataScale', 10), [10 20]));
        assert(isequal(size(uniquetol([3; 1; 2])), [3 1]));
    """)
    with pytest.raises(MatlabError):
        m.run("uniquetol([1 2 3], 0.1, 'NoSuchOption', 1)")
