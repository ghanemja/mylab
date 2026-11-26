"""imageDatastore, and transform and combine on image and tabular text datastores."""

import numpy as np
import pytest

from mylab.errors import MatlabError


@pytest.fixture
def pics(m, tmp_path, monkeypatch):
    """pics/round holds 3 PNG files, pics/square 5 JPEG files and a text file; pics/extra/deep 1 BMP file."""
    monkeypatch.chdir(tmp_path)
    m.run("mkdir('pics'); mkdir('pics/round'); mkdir('pics/square'); mkdir('pics/extra'); mkdir('pics/extra/deep');"
          "for k = 1:3, imwrite(uint8(10*k*ones(4, 6)), sprintf('pics/round/r%d.png', k)); end;"
          "for k = 1:5, imwrite(uint8(repmat(40*k, 5, 5, 3)), sprintf('pics/square/s%d.jpg', k), 'Quality', 100);"
          "end; imwrite(uint8(zeros(2, 3)), 'pics/extra/deep/z.bmp');"
          "fid = fopen('pics/square/readme.txt', 'w'); fprintf(fid, 'not an image'); fclose(fid);")
    return m


def test_construction_and_properties(pics, tmp_path):
    m = pics
    m.run("ds = imageDatastore('pics', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');"
          "top = imageDatastore('pics/round'); one = imageDatastore('pics/round/r2.png');"
          "wild = imageDatastore('pics/square/s*.jpg'); mix = imageDatastore({'pics/round', \"pics/square/s1.jpg\"});"
          "pngs = imageDatastore('pics', 'IncludeSubfolders', 1, 'FileExtensions', '.png');"
          "both = imageDatastore('pics', 'IncludeSubfolders', 1, 'FileExtensions', {'png', '.BMP'});")
    assert m.str("class(ds)") == "matlab.io.datastore.ImageDatastore"
    assert m.num("isa(ds, 'matlab.io.Datastore')") and m.num("isa(ds, 'handle')")
    assert m.num("numel(ds.Files)") == 9 and m.num("size(ds.Files, 2)") == 1 and m.str("class(ds.Files)") == "cell"
    assert m.str("ds.Files{1}") == str(tmp_path / "pics" / "extra" / "deep" / "z.bmp")
    assert m.str("class(ds.Labels)") == "categorical" and m.num("numel(ds.Labels)") == 9
    assert m.str("char(ds.Labels(1))") == "deep" and m.str("char(ds.Labels(end))") == "square"
    assert m.str("strjoin(categories(ds.Labels), ',')") == "deep,round,square"
    assert m.str("ds.Folders{1}") == str(tmp_path / "pics") and m.num("ds.ReadSize") == 1
    assert m.str("func2str(ds.ReadFcn)") in ("imread", "@imread")
    assert m.num("numel(top.Files)") == 3 and m.num("isempty(top.Labels)")
    assert m.num("numel(one.Files)") == 1 and m.num("numel(wild.Files)") == 5 and m.num("numel(mix.Files)") == 4
    assert m.num("numel(mix.Folders)") == 2
    assert m.num("numel(pngs.Files)") == 3 and m.num("numel(both.Files)") == 4
    out = m.run("ds")
    assert "ImageDatastore with properties" in out and "Labels" in out and "ReadSize" in out


def test_read_is_shared_and_ordered(pics, tmp_path):
    m = pics
    m.run("ds = imageDatastore('pics/round', 'LabelSource', 'foldernames'); alias = ds;"
          "[a, info] = read(ds); b = read(alias); c = read(ds); more = hasdata(ds);"
          "reset(alias); again = read(ds); p = progress(ds);")
    np.testing.assert_array_equal(m.num("a"), 10 * np.ones((4, 6)))
    assert m.str("class(a)") == "uint8"
    assert m.num("b(1)") == 20 and m.num("c(1)") == 30 and not m.num("more") and m.num("again(1)") == 10
    assert m.str("info.Filename") == str(tmp_path / "pics" / "round" / "r1.png")
    assert m.num("info.FileSize") == (tmp_path / "pics" / "round" / "r1.png").stat().st_size
    assert m.str("char(info.Label)") == "round"
    assert m.num("p") == pytest.approx(1 / 3)
    with pytest.raises(MatlabError, match="No more data"):
        m.run("read(ds); read(ds); read(ds);")
    m.run("ok = 0; reset(ds); while hasdata(ds), img = read(ds); ok = ok + 1; end")
    assert m.num("ok") == 3


def test_readimage_readall_preview_and_readsize(pics):
    m = pics
    m.run("ds = imageDatastore('pics/square'); im = readimage(ds, 2); [~, inf2] = readimage(ds, 5);"
          "all = readall(ds); r = read(ds); first = preview(ds);"
          "ds.ReadSize = 2; reset(ds); [blk, binfo] = read(ds); blk2 = read(ds); [blk3, i3] = read(ds);")
    assert m.num("size(im)").tolist() == [[5, 5, 3]] and abs(m.num("im(1)").item() - 80) <= 2
    assert m.str("inf2.Filename").endswith("s5.jpg") and m.num("isempty(inf2.Label)")
    assert m.str("class(all)") == "cell" and m.num("size(all)").tolist() == [[5, 1]]
    assert abs(m.num("first(1)").item() - 40) <= 2  # preview does not depend on the read position
    assert m.str("class(blk)") == "cell" and m.num("size(blk)").tolist() == [[2, 1]]
    assert m.num("numel(binfo.Filename)") == 2 and m.num("numel(binfo.FileSize)") == 2
    assert m.num("size(blk2, 1)") == 2 and m.num("size(blk3)").tolist() == [[1, 1]]
    assert m.str("class(i3.Filename)") == "cell" and m.str("i3.Filename{1}").endswith("s5.jpg")
    with pytest.raises(MatlabError, match="exceeds the number of files"):
        m.run("readimage(ds, 6)")


def test_read_size_partial_last_block(pics):
    m = pics
    m.run("ds = imageDatastore('pics/square', 'ReadSize', 2); n = [];"
          "while hasdata(ds), n(end+1) = numel(read(ds)); end")
    np.testing.assert_array_equal(m.num("n"), [[2, 2, 1]])


def test_count_and_split_each_label(pics):
    m = pics
    m.run("ds = imageDatastore({'pics/round', 'pics/square'}, 'LabelSource', 'foldernames');"
          "t = countEachLabel(ds); [a, b] = splitEachLabel(ds, 0.6); ta = countEachLabel(a); tb = countEachLabel(b);"
          "[c1, c2, c3] = splitEachLabel(ds, 1, 1); n3 = countEachLabel(c3);"
          "rng(7); [r1, r2] = splitEachLabel(ds, 2, 'randomized');"
          "rng(7); [q1, q2] = splitEachLabel(ds, 2, 'randomize');"
          "inc = splitEachLabel(ds, 0.5, 'Include', 'square'); exc = splitEachLabel(ds, 1, 'Exclude', {'square'});")
    assert m.str("class(t)") == "table" and m.val("t").names == ["Label", "Count"]
    assert m.str("class(t.Label)") == "categorical"
    np.testing.assert_array_equal(m.num("t.Count"), [[3], [5]])
    np.testing.assert_array_equal(m.num("t{:, 2}"), [[3], [5]])
    np.testing.assert_array_equal(m.num("ta.Count"), [[2], [3]])
    np.testing.assert_array_equal(m.num("tb.Count"), [[1], [2]])
    assert m.num("numel(a.Files) + numel(b.Files)") == 8
    assert m.num("isempty(intersect(a.Files, b.Files))")
    assert m.str("a.Files{1}").endswith("r1.png")  # not randomized: the first files of each label, in order
    np.testing.assert_array_equal(m.num("n3.Count"), [[1], [3]])
    assert m.num("numel(c1.Files)") == 2 and m.num("numel(c2.Files)") == 2
    assert m.num("numel(r1.Files)") == 4 and m.num("numel(r2.Files)") == 4
    assert m.num("isequal(r1.Files, q1.Files)")  # the same global random state gives the same split
    assert m.num("numel(inc.Files)") == 3 and m.num("all(inc.Labels == 'square')")
    assert m.num("numel(exc.Files)") == 1 and m.str("char(exc.Labels)") == "round"
    with pytest.raises(MatlabError, match="fewer than"):
        m.run("splitEachLabel(ds, 4)")
    with pytest.raises(MatlabError, match="no labels"):
        m.run("countEachLabel(imageDatastore('pics/round'))")
    with pytest.raises(MatlabError, match="add up to 1"):
        m.run("splitEachLabel(ds, 0.7, 0.6)")
    with pytest.raises(MatlabError, match="NoSuchOption__"):
        m.run("splitEachLabel(ds, 0.5, 'NoSuchOption__', 1)")


def test_randomized_split_follows_the_global_stream(pics):
    m = pics
    m.run("ds = imageDatastore({'pics/round', 'pics/square'}, 'LabelSource', 'foldernames');"
          "rng(1); [x1, ~] = splitEachLabel(ds, 0.5, 'randomized');"
          "rng(2); [x2, ~] = splitEachLabel(ds, 0.5, 'randomized');"
          "rng(1); [x3, ~] = splitEachLabel(ds, 0.5, 'randomized');")
    assert m.num("isequal(x1.Files, x3.Files)")
    assert m.num("numel(x1.Files)") == 5  # round(0.5 * 3) + round(0.5 * 5) files, halves rounded up


def test_labels_given_directly_and_set_later(pics):
    m = pics
    m.run("ds = imageDatastore('pics/square', 'Labels', [1 1 2 2 2]); t = countEachLabel(ds);"
          "ds.Labels = categorical({'a'; 'b'; 'a'; 'b'; 'a'}); t2 = countEachLabel(ds); [~, info] = read(ds);"
          "ds.ReadFcn = @(f) size(imread(f)); s = readimage(ds, 1);")
    np.testing.assert_array_equal(m.num("t.Count"), [[2], [3]])
    np.testing.assert_array_equal(m.num("t2.Count"), [[3], [2]])
    assert m.str("char(info.Label)") == "a"
    np.testing.assert_array_equal(m.num("s"), [[5, 5, 3]])
    m.run("ds.Files = ds.Files([5 1]); kept = ds.Labels;")
    assert m.num("numel(ds.Files)") == 2 and m.str("strjoin(cellstr(kept), ',')") == "a,a"
    with pytest.raises(MatlabError, match="one element per file"):
        m.run("ds.Labels = [1 2 3]")
    with pytest.raises(MatlabError, match="function handle"):
        m.run("ds.ReadFcn = 3")
    with pytest.raises(MatlabError, match="read-only"):
        m.run("ds.Folders = {'x'}")


def test_shuffle_subset_partition(pics):
    m = pics
    m.run("ds = imageDatastore('pics/square', 'LabelSource', 'foldernames'); rng(3); sh = shuffle(ds);"
          "s = subset(ds, [5 1]); l = subset(ds, logical([0 1 0 1 0])); n = numpartitions(ds);"
          "p1 = partition(ds, 2, 1); p2 = partition(ds, 2, 2); pf = partition(ds, 'Files', 3);"
          "read(ds); fresh = hasdata(sh) && progress(sh) == 0;")
    assert m.num("numel(sh.Files)") == 5 and m.num("isequal(sort(sh.Files), ds.Files)")
    assert m.str("s.Files{1}").endswith("s5.jpg") and m.num("numel(s.Labels)") == 2
    assert m.str("l.Files{2}").endswith("s4.jpg")
    assert m.num("n") == 5
    assert m.num("numel(p1.Files) + numel(p2.Files)") == 5 and m.num("isempty(intersect(p1.Files, p2.Files))")
    assert m.str("pf.Files{1}").endswith("s3.jpg")
    assert m.num("fresh")
    with pytest.raises(MatlabError, match="from 1 to 5"):
        m.run("subset(ds, 6)")


def test_transform_and_combine(pics):
    m = pics
    m.run("ds = imageDatastore('pics/round'); t = transform(ds, @(x) double(x(1)) / 10);"
          "v1 = read(t); v2 = read(t); all = readall(t); after = read(t);"
          "ti = transform(ds, @(x, info) deal(size(x), info), 'IncludeInfo', true); [sz, inf] = read(ti);"
          "c = combine(ds, t); reset(ds); pair = read(c); everything = readall(c);"
          "cs = combine(imageDatastore('pics/square'), imageDatastore('pics/square')); rng(4); csh = shuffle(cs);"
          "q = read(csh); same = isequal(q{1}, q{2}); pv = preview(t);")
    assert m.num("v1") == 1 and m.num("v2") == 2 and m.num("after") == 3
    assert m.str("class(all)") == "cell" and m.num("size(all)").tolist() == [[3, 1]] and m.num("all{3}") == 3
    np.testing.assert_array_equal(m.num("sz"), [[4, 6]])
    assert m.str("inf.Filename").endswith("r1.png")
    assert m.str("class(c)") == "matlab.io.datastore.CombinedDatastore"
    assert m.str("class(t)") == "matlab.io.datastore.TransformedDatastore"
    assert m.num("size(pair)").tolist() == [[1, 2]] and m.num("pair{2}") == 1
    assert m.num("size(everything)").tolist() == [[3, 2]] and m.num("everything{3, 2}") == 3
    assert m.num("same") and m.num("pv") == 1
    with pytest.raises(MatlabError, match="function handle"):
        m.run("transform(ds, 3)")


def test_transform_on_tabular_text_datastore(m, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m.run("writetable(table([1; 2; 3], [4; 5; 6], 'VariableNames', {'a', 'b'}), 'p.csv');"
          "ds = tabularTextDatastore('p.csv'); t = transform(ds, @(T) T(T.a > 1, :)); r = readall(t);"
          "c = combine(ds, ds); both = read(c);")
    assert m.num("size(both)").tolist() == [[1, 2]] and m.str("class(both{1})") == "table"
    assert m.str("class(r)") == "table" and m.num("height(r)") == 2
    np.testing.assert_array_equal(m.num("r.b"), [[5], [6]])


def test_datastore_type_image_and_errors(pics):
    m = pics
    m.run("ds = datastore('pics/round', 'Type', 'image', 'LabelSource', 'foldernames');")
    assert m.str("class(ds)") == "matlab.io.datastore.ImageDatastore" and m.num("numel(ds.Labels)") == 3
    with pytest.raises(MatlabError, match="NoSuchOption__"):
        m.run("imageDatastore('pics', 'NoSuchOption__', 1)")
    with pytest.raises(MatlabError, match="Cannot find|no files"):
        m.run("imageDatastore('no_such_folder')")
    with pytest.raises(MatlabError, match="no files"):
        m.run("imageDatastore('pics/square', 'FileExtensions', '.gif')")
    with pytest.raises(MatlabError, match="'none' or 'foldernames'"):
        m.run("imageDatastore('pics/round', 'LabelSource', 'filenames')")
    with pytest.raises(MatlabError, match="not both"):
        m.run("imageDatastore('pics/round', 'LabelSource', 'foldernames', 'Labels', [1 2 3])")
    with pytest.raises(MatlabError, match="positive integer"):
        m.run("imageDatastore('pics/round', 'ReadSize', 0)")
    with pytest.raises(MatlabError, match="needs a folder"):
        m.run("imageDatastore()")


def test_methods_listed_and_callable_with_dot(pics):
    m = pics
    m.run("ds = imageDatastore('pics/round'); im = ds.readimage(2); h = ds.hasdata(); ms = methods(ds);"
          "ps = properties(ds);")
    assert m.num("im(1)") == 20 and m.num("h")
    assert m.num("any(strcmp(ms, 'splitEachLabel'))") and m.num("any(strcmp(ps, 'Labels'))")
