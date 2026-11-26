"""Bugs found while comparing the merged work with GNU Octave on the held-out checks."""

import pytest

from mylab.errors import MatlabError


def test_fscanf_reads_one_word_per_field_and_keeps_its_place(m, tmp_path):
    f = tmp_path / "words.txt"
    f.write_text("alpha beta\n1 2 3\n")
    m.run(f"fid = fopen('{f}'); x = fscanf(fid, '%s', 1); y = fscanf(fid, '%s', 1); z = fscanf(fid, '%d'); "
          "fclose(fid);")
    assert m.str("x") == "alpha" and m.str("y") == "beta"
    assert m.num("z").ravel().tolist() == [1, 2, 3]


def test_sscanf_stops_at_the_field_count(m):
    m.run("[a, n, e, nx] = sscanf('alpha beta', '%s', 1); [c, ~, ~, nc] = sscanf('12 34 56', '%d', 2);")
    assert m.str("a") == "alpha" and m.num("n") == 1 and m.num("nx") == 6
    assert m.num("c").ravel().tolist() == [12, 34] and m.num("nc") == 6
    m.run("[d, ~, ~, nd] = sscanf('12 34 56', '%d');")
    assert m.num("d").ravel().tolist() == [12, 34, 56] and m.num("nd") == 9
    assert m.num("sscanf('1 2 3 4 5 6', '%d', [2 2])").tolist() == [[1, 3], [2, 4]]


def test_delete_expands_wildcards_and_warns_about_missing_files(m, tmp_path):
    for name in ("a.dat", "b.dat", "keep.txt"):
        (tmp_path / name).write_text("x")
    out = m.run(f"delete('{tmp_path / '*.dat'}'); delete('{tmp_path / 'none.txt'}');")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]
    assert "Warning" in out and "none.txt" in out


def test_evalc_returns_extra_outputs(m):
    m.run("[t, a, b] = evalc('deal(1, 2)'); [u, v] = evalc('max([3 9 2])');")
    assert m.str("t") == "" and m.num("a") == 1 and m.num("b") == 2 and m.num("v") == 9
    with pytest.raises(MatlabError, match="expression"):
        m.run("[t, a] = evalc('x = 1;');")


@pytest.mark.parametrize("code, ident", [
    ("validateattributes([1 2 3], {'numeric'}, {'scalar'})", "MATLAB:expectedScalar"),
    ("validateattributes('x', {'numeric'}, {})", "MATLAB:invalidType"),
    ("validateattributes(-1, {'numeric'}, {'positive'})", "MATLAB:expectedPositive"),
    ("validateattributes([1 2], {'numeric'}, {'numel', 3})", "MATLAB:incorrectNumel"),
    ("validateattributes(5, {'numeric'}, {'<', 3})", "MATLAB:notLess"),
    ("validateattributes([], {'numeric'}, {'nonempty'})", "MATLAB:expectedNonempty"),
])
def test_validateattributes_error_identifiers(m, code, ident):
    m.run(f"try, {code}; id = ''; catch e, id = e.identifier; end")
    assert m.str("id") == ident


def test_sparse_scalar_into_full_array(m):
    m.run("S = sparse([0 7; 0 0]); x = zeros(1, 3); x(2) = S(1, 2);")
    assert m.num("x").tolist() == [[0, 7, 0]] and m.str("class(x)") == "double"


def test_writetable_row_names_append_and_unknown_options(m, tmp_path):
    f = tmp_path / "rn.csv"
    m.run(f"T = table([1; 2], [3; 4], 'VariableNames', {{'a', 'b'}}, 'RowNames', {{'r1', 'r2'}}); "
          f"writetable(T, '{f}', 'WriteRowNames', true); R = readtable('{f}', 'ReadRowNames', true);")
    assert f.read_text().splitlines()[0] == "Row,a,b"
    assert m.str("strjoin(R.Properties.RowNames, ',')") == "r1,r2" and m.num("R.b").ravel().tolist() == [3, 4]
    m.run(f"writetable(T, '{f}', 'WriteMode', 'append');")
    assert f.read_text().splitlines()[-1] == "2,4"
    with pytest.raises(MatlabError, match="Bogus"):
        m.run(f"writetable(T, '{tmp_path / 'x.csv'}', 'Bogus', 1);")


def test_delete_of_empty_values_and_figure_numbers(m):
    m.run("delete(gcp('nocreate')); delete([]); figure(7); delete(7); n = numel(findobj('Type', 'figure'));")
    assert m.num("n") == 0


def test_do_is_not_a_keyword_because_of_a_comment(m, tmp_path):
    # a function named 'do', called as a statement, in a file whose comment says ", until"
    (tmp_path / "do.m").write_text("function do()\nglobal DONE\nDONE = 1;\nend\n")
    (tmp_path / "script.m").write_text("% repeat, until done\nglobal DONE\nDONE = 0;\ndo\n")
    m.run(f"addpath('{tmp_path}'); script; global DONE; x = DONE;")
    assert m.num("x") == 1


def test_persistent_store_belongs_to_its_definition(m, tmp_path):
    f = tmp_path / "counter_p.m"
    f.write_text("function r = counter_p()\npersistent n\nif isempty(n), n = 0; end\nn = n + 1; r = n;\nend\n")
    m.run(f"addpath('{tmp_path}'); counter_p(); counter_p();")
    assert m.num("counter_p()") == 3
    m.run("clear counter_p")
    assert m.num("counter_p()") == 1
