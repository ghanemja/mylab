"""Frames, movies, VideoWriter, animation capture and the extra graphics functions."""

import base64
import io
import time

import numpy as np
import pytest
from PIL import Image

from mylab.api import Session
from mylab.builtins import plotting
from mylab.errors import MatlabError


def gif(data) -> Image.Image:
    """An animated GIF given as bytes or as a path."""
    return Image.open(io.BytesIO(data if isinstance(data, bytes) else data.read_bytes()))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# -- getframe / frame2im / im2frame / movie -----------------------------------------------------------

def test_getframe_figure_and_axes(m):
    m.run("plot(1:10); F = getframe(gcf); G = getframe; H = getframe(gca);")
    F = m.val("F")
    cdata = F.get_field(0, "cdata")
    assert cdata.dtype == np.uint8 and cdata.shape == (480, 640, 3)
    assert F.get_field(0, "colormap").size == 0
    g = m.val("G").get_field(0, "cdata")
    assert g.shape == m.val("H").get_field(0, "cdata").shape
    assert g.shape[2] == 3 and g.shape[0] < 480 and g.shape[1] < 640  # only the axes area
    assert cdata.min() < 100 and cdata.max() == 255  # something was drawn on a white page


def test_getframe_by_number_and_rect(m):
    m.run("figure(3); plot(rand(4)); F = getframe(3); R = getframe(gcf, [10 20 100 50]);")
    assert m.val("F").get_field(0, "cdata").shape == (480, 640, 3)
    assert m.val("R").get_field(0, "cdata").shape == (50, 100, 3)
    with pytest.raises(MatlabError):
        m.run("getframe(99)")


def test_frame_array_and_conversions(m):
    m.run("""
    for k = 1:3
        plot(rand(1, 5)); F(k) = getframe(gcf);
    end
    n = numel(F); img = frame2im(F(2));
    G = im2frame(uint8(zeros(4, 6, 3))); [X, map] = frame2im(G);
    D = im2frame(ones(2, 2, 3) * 0.5);
    I = im2frame(uint8([0 1; 1 0]), [1 0 0; 0 0 1]); [XI, mi] = frame2im(I);
    movie(F); movie(F, 2, 10); movie(gcf, F);
    """)
    assert m.num("n") == 3
    assert m.val("img").shape == (480, 640, 3)
    assert m.val("X").shape == (4, 6, 3) and m.val("map").size == 0
    assert m.val("D").get_field(0, "cdata").dtype == np.uint8
    assert int(m.val("D").get_field(0, "cdata")[0, 0, 0]) == 128
    assert m.val("mi").shape == (2, 3) and m.val("XI").shape == (2, 2)
    with pytest.raises(MatlabError):
        m.run("frame2im(5)")
    with pytest.raises(MatlabError):
        m.run("movie(5)")


# -- VideoWriter ---------------------------------------------------------------------------------------

def test_videowriter_gif(m, tmp_path):
    out = tmp_path / "anim.gif"
    text = m.run(f"""
    v = VideoWriter('{out}');
    v.FrameRate = 5;
    open(v);
    for k = 1:6
        plot(1:10, (1:10) * k); title(sprintf('frame %d', k));
        writeVideo(v, getframe(gcf));
    end
    n = v.FrameCount; w = v.Width; h = v.Height; d = v.Duration; name = v.Filename;
    close(v);
    c = class(v);
    v
    """)
    assert m.num("n") == 6 and m.num("w") == 640 and m.num("h") == 480
    assert m.num("d") == pytest.approx(1.2)
    assert m.str("name") == "anim.gif" and m.str("c") == "VideoWriter"
    assert "FrameRate: 5" in text
    im = gif(out)
    assert im.format == "GIF" and im.size == (640, 480) and im.n_frames == 6
    assert im.info["duration"] == 200


def test_videowriter_images_and_4d(m, tmp_path):
    out = tmp_path / "img.gif"
    m.run(f"""
    v = VideoWriter('{out}'); open(v);
    A = zeros(20, 30, 3, 4, 'uint8');
    for k = 1:4, A(:, :, 1, k) = 60 * k; end
    writeVideo(v, A);                       % 4-D stack
    writeVideo(v, rand(20, 30));            % grayscale doubles
    writeVideo(v, uint8(255 * ones(10, 15, 3)));  % another size: resized
    n = v.FrameCount;
    v.close();
    """)
    assert m.num("n") == 6
    im = gif(out)
    assert im.size == (30, 20) and im.n_frames == 6


def test_videowriter_falls_back_to_gif_without_ffmpeg(m, tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "_find_ffmpeg", lambda: None)
    text = m.run(f"""
    v = VideoWriter('{tmp_path / "movie.avi"}'); open(v);
    for k = 1:3, writeVideo(v, uint8(80 * k * ones(16, 16, 3))); end
    close(v);
    w = VideoWriter('{tmp_path / "clip"}', 'MPEG-4'); open(w);
    for k = 1:2, writeVideo(w, uint8(100 * k * ones(16, 16, 3))); end
    close(w); fn = w.Filename;
    """)
    assert "Warning" in text and "movie.gif" in text and "ffmpeg" in text
    assert not (tmp_path / "movie.avi").exists()
    assert gif(tmp_path / "movie.gif").n_frames == 3
    assert m.str("fn") == "clip.mp4"
    assert gif(tmp_path / "clip.gif").n_frames == 2


@pytest.mark.skipif(plotting._find_ffmpeg() is None, reason="needs ffmpeg")
@pytest.mark.parametrize("name", ["movie.avi", "movie.mp4"])
def test_videowriter_with_ffmpeg(m, tmp_path, name):
    text = m.run(f"""
    v = VideoWriter('{tmp_path / name}'); v.Quality = 90; open(v);
    for k = 1:5, plot(rand(1, 8)); writeVideo(v, getframe(gcf)); end
    close(v);
    """)
    assert "Warning" not in text
    assert (tmp_path / name).stat().st_size > 1000
    assert not (tmp_path / "movie.gif").exists()


def test_videowriter_errors_and_close_dispatch(m, tmp_path):
    m.run(f"v = VideoWriter('{tmp_path / 'x.gif'}');")
    with pytest.raises(MatlabError, match="open"):
        m.run("writeVideo(v, zeros(4, 4, 3, 'uint8'))")
    with pytest.raises(MatlabError):
        m.run("v.FrameRate = -1;")
    with pytest.raises(MatlabError):
        m.run("v.FrameCount = 3;")
    with pytest.raises(MatlabError):
        m.run("VideoWriter('a.avi', 'No Such Profile')")
    with pytest.raises(MatlabError):
        m.run("writeVideo(5, 6)")
    with pytest.raises(MatlabError):
        m.run(f"q = VideoWriter('{tmp_path / 'nodir' / 'x.avi'}'); open(q);")
    # close still works for figures while a writer exists
    m.run("open(v); figure(1); figure(2); f = figure(3); close(f); n1 = numel(findobj('Type', 'figure'));")
    m.run("close(1); close all; n2 = numel(findobj('Type', 'figure')); close(v); close(v);")
    assert m.num("n1") == 2 and m.num("n2") == 0
    assert not (tmp_path / "x.gif").exists()  # no frames were written
    with pytest.raises(MatlabError, match="closed"):
        m.run("writeVideo(v, zeros(4, 4, 3, 'uint8'))")


def test_open_with_file_names(m, tmp_path):
    with pytest.raises(MatlabError, match="fig"):
        m.run("open('saved.fig')")
    with pytest.raises(MatlabError, match="not found"):
        m.run("open('no_such_file_zz.txt')")
    m.run(f"a = 5; save('{tmp_path / 'd.mat'}', 'a'); s = open('{tmp_path / 'd.mat'}'); b = s.a;")
    assert m.num("b") == 5


# -- animation capture ----------------------------------------------------------------------------------

LOOP = """
x = linspace(0, 2*pi, 50);
for k = 1:8
    plot(x, sin(x + k/3)); ylim([-1 1]);
    drawnow;
end
"""


def test_session_returns_animation():
    s = Session()
    r = s.run(LOOP)
    assert r.ok and len(r.figures) == 1 and len(r.animations) == 1
    im = gif(r.animations[0])
    assert im.format == "GIF" and im.n_frames == 8 and im.size == (640, 480)
    assert im.info["duration"] == 80  # about 12 frames per second
    d = r.to_dict()
    assert base64.b64decode(d["animations"][0]) == r.animations[0]
    assert r.animations_base64() == d["animations"]
    # the next run starts with a clean slate
    r2 = s.run("y = 1;")
    assert r2.animations == [] and r2.to_dict()["animations"] == []
    assert "animation" not in s.interp.state


def test_no_animation_without_redraws():
    s = Session()
    assert s.run("plot(1:5); title('still');").animations == []
    assert "animation" not in s.interp.state  # nothing was captured at all
    assert s.run("x = 1; drawnow; pause(0.001);").animations == []  # no figure
    assert "animation" not in s.interp.state
    assert s.run("plot(1:5); drawnow;").animations == []  # one frame is not an animation
    assert s.run("plot(1:5); drawnow; drawnow; pause(0.001); drawnow limitrate").animations == []  # identical


def test_pause_captures_and_does_not_really_wait():
    s = Session()
    t0 = time.perf_counter()
    r = s.run("for k = 1:5, bar(1:k); pause(0.5); end")
    assert time.perf_counter() - t0 < 2.0  # five half-second pauses are cut short
    assert r.ok and len(r.animations) == 1
    im = gif(r.animations[0])
    assert im.n_frames == 5 and im.info["duration"] == 500


def test_pause_without_figures_still_waits(m):
    t0 = time.perf_counter()
    m.run("pause(0.12); pause; pause on; pause(0)")
    dt = time.perf_counter() - t0
    assert 0.1 <= dt < 1.0


def test_capture_is_bounded_and_covers_the_run():
    s = Session()
    r = s.run("""
    figure('Position', [0 0 240 180]);
    h = animatedline; axis([0 400 0 400]);
    for k = 1:400
        addpoints(h, k, k); drawnow;
    end
    """)
    assert r.ok
    im = gif(r.animations[0])
    assert 50 <= im.n_frames <= plotting.ANIM_MAX_FRAMES
    assert im.size[0] <= plotting.ANIM_MAX_WIDTH
    first = np.asarray(im.convert("RGB")).astype(int)
    im.seek(im.n_frames - 1)
    last = np.asarray(im.convert("RGB")).astype(int)
    assert (last < 100).sum() > (first < 100).sum() + 100  # the final frame shows the long line


def test_wide_figures_are_scaled_down():
    s = Session()
    r = s.run("figure('Position', [0 0 1600 800]); for k = 1:3, plot(rand(1, 4)); drawnow; end")
    assert gif(r.animations[0]).size == (720, 360)


def test_one_animation_per_figure_and_opt_out():
    s = Session()
    r = s.run("""
    for k = 1:3
        figure(1); plot(1:k, 'o-'); drawnow;
        figure(2); bar(1:k); drawnow;
    end
    figure(3); plot(1:3); drawnow;
    close all
    """)
    assert len(r.animations) == 2 and r.figures == []
    assert [gif(a).n_frames for a in r.animations] == [3, 3]
    s.interp.state["animation_capture"] = False
    assert s.run(LOOP).animations == []
    assert "animation" not in s.interp.state


def test_getframe_counts_as_a_redraw():
    s = Session()
    r = s.run("for k = 1:4, plot(rand(1, 6)); F(k) = getframe(gcf); end; movie(F)")
    assert r.ok and gif(r.animations[0]).n_frames == 4


def test_frame_helpers(m):
    m.run("plot(1:3); drawnow; plot(3:-1:1); drawnow;")
    assert plotting.animation_frame_counts(m.interp) == {1: 2}
    gifs = plotting.render_animations(m.interp, clear=False)
    assert len(gifs) == 1 and gif(gifs[0]).n_frames == 2
    plotting.clear_animation(m.interp)
    assert plotting.animation_frame_counts(m.interp) == {}
    assert plotting.render_animations(m.interp) == []


# -- animatedline ---------------------------------------------------------------------------------------

def test_animatedline(m):
    m.run("""
    h = animatedline('Color', 'r', 'LineWidth', 2);
    for k = 1:10, addpoints(h, k, k^2); end
    [x, y] = getpoints(h); c = class(h); t = h.Type; xl = xlim; yl = ylim;
    addpoints(h, [11 12], [121 144]); n = numel(h.XData);
    clearpoints(h); [x0, y0] = getpoints(h);
    g = animatedline('MaximumNumPoints', 3); for k = 1:7, addpoints(g, k, -k); end
    xg = getpoints(g);
    """)
    assert m.num("x").tolist() == [list(range(1, 11))]
    assert m.num("y")[0, -1] == 100
    assert m.str("c").endswith("AnimatedLine") and m.str("t") == "animatedline"
    assert m.num("xl")[0, 1] >= 10 and m.num("yl")[0, 1] >= 100
    assert m.num("n") == 12 and m.num("x0").size == 0
    assert m.num("xg").tolist() == [[5, 6, 7]]
    with pytest.raises(MatlabError):
        m.run("addpoints(h, 1)")
    with pytest.raises(MatlabError):
        m.run("p = plot(1:3); addpoints(p, 1, 2)")


def test_animatedline_3d(m):
    m.run("h = animatedline(0, 0, 0); addpoints(h, 1, 2, 3); [x, y, z] = getpoints(h); zl = zlim;")
    assert m.num("z").tolist() == [[0, 3]] and m.num("zl")[0, 1] >= 3


# -- axes, rectangle, findobj -------------------------------------------------------------------------

def test_axes_create_and_select(m):
    m.run("""
    a1 = axes('Position', [0.1 0.1 0.8 0.8]); plot(1:10);
    a2 = axes('Position', [0.6 0.2 0.25 0.25], 'Box', 'on', 'XTick', []); plot(a2, [3 1 2]);
    p = a2.Position; isa2 = isequal(gca, a2);
    axes(a1); isa1 = isequal(gca, a1); hold on; plot([10 1]); n1 = numel(a1.Children);
    n = numel(findobj(gcf, 'Type', 'axes'));
    f2 = figure; a3 = axes(f2); a4 = axes('Parent', f2); same = isequal(ancestor(a4, 'figure'), f2);
    """)
    assert np.allclose(m.num("p"), [[0.6, 0.2, 0.25, 0.25]])
    assert m.num("isa2") and m.num("isa1") and m.num("n1") == 2 and m.num("n") == 2 and m.num("same")
    with pytest.raises(MatlabError):
        m.run("axes('Position', [0 0 0 1])")


def test_rectangle(m):
    m.run("""
    h = rectangle('Position', [1 2 3 4], 'Curvature', [1 1], 'FaceColor', 'r', 'EdgeColor', [0 0 1], ...
                  'LineWidth', 2, 'Tag', 'ball');
    p = h.Position; c = h.Curvature; fc = h.FaceColor; t = h.Type; xl = xlim; yl = ylim;
    h.Position = [0 0 10 10]; xl2 = xlim; h.Curvature = 0.2; c2 = h.Curvature;
    r = rectangle; n = numel(findobj(gca, 'Type', 'rectangle')); tagged = findobj('Tag', 'ball');
    same = isequal(tagged, h); delete(h); n2 = numel(findobj(gca, 'Type', 'rectangle'));
    """)
    assert m.num("p").tolist() == [[1, 2, 3, 4]] and m.num("c").tolist() == [[1, 1]]
    assert m.num("fc").tolist() == [[1, 0, 0]] and m.str("t") == "rectangle"
    assert m.num("xl")[0, 0] <= 1 and m.num("xl")[0, 1] >= 4 and m.num("yl")[0, 1] >= 6
    assert m.num("xl2")[0, 1] >= 10 and np.allclose(m.num("c2"), [[0.2, 0.2]])
    assert m.num("n") == 2 and m.num("same") and m.num("n2") == 1
    with pytest.raises(MatlabError):
        m.run("rectangle('Position', [1 2 3])")


def test_findobj_and_findall(m):
    m.run("""
    f = figure; subplot(1, 2, 1); p = plot(1:3, 'Tag', 'a'); hold on; q = plot(3:-1:1, 'r--');
    set(p, 'Tag', 'first'); set(q, 'Tag', 'second', 'LineWidth', 3);
    subplot(1, 2, 2); b = bar([1 2 3]); t = text(1, 1, 'hi');
    nl = numel(findobj('Type', 'line')); nall = numel(findall(f, 'type', 'LINE'));
    one = findobj(f, 'Tag', 'second'); ok1 = isequal(one, q);
    w = findobj(gcf, 'Type', 'line', 'LineWidth', 3); ok2 = isequal(w, q);
    none = findobj(f, 'Tag', 'nothing'); e = isempty(none);
    nb = numel(findobj(f, 'Type', 'bar')); nt = numel(findobj(f, 'Type', 'text'));
    either = findobj(f, 'Tag', 'first', '-or', 'Tag', 'second'); ne = numel(either);
    notl = findobj(f, 'Type', 'line', '-not', 'Tag', 'first'); ok3 = isequal(notl, q);
    rx = findobj(f, '-regexp', 'Tag', '^sec'); ok4 = isequal(rx, q);
    flat = findobj(f, 'flat', 'Type', 'line'); ef = isempty(flat);
    d1 = numel(findobj(f, '-depth', 1, 'Type', 'axes'));
    ch = allchild(f); nch = numel(ch);
    nfig = numel(findobj(0, 'Type', 'figure')); ee = isempty(findobj([], 'Type', 'line'));
    set(findobj(f, 'Type', 'line'), 'LineWidth', 5); lw = p.LineWidth;
    """)
    for name, want in [("nl", 2), ("nall", 2), ("ok1", 1), ("ok2", 1), ("e", 1), ("nb", 1), ("nt", 1), ("ne", 2),
                       ("ok3", 1), ("ok4", 1), ("ef", 1), ("d1", 2), ("nch", 2), ("nfig", 1), ("ee", 1),
                       ("lw", 5)]:
        assert m.num(name) == want, name


def test_ancestor_and_appdata(m):
    m.run("""
    f = figure(7); p = plot(1:3); ax = ancestor(p, 'axes'); ff = ancestor(p, 'figure'); n = ff.Number;
    isax = isequal(ax, gca); self = isequal(ancestor(f, 'figure'), f); none = isempty(ancestor(f, 'axes'));
    c = ancestor(p, {'figure', 'axes'}); isax2 = isequal(c, ax);
    setappdata(f, 'count', 3); setappdata(f, 'name', 'abc'); setappdata(7, 'count', 4);
    v = getappdata(f, 'count'); s = getappdata(f); has = isappdata(f, 'name'); miss = isempty(getappdata(f, 'zz'));
    rmappdata(f, 'name'); has2 = isappdata(f, 'name');
    d.items = [1 2 3]; guidata(f, d); g = guidata(p); k = numel(g.items); e0 = isempty(guidata(figure(8)));
    setappdata(0, 'shared', 11); close all; r = getappdata(0, 'shared'); r2 = getappdata(groot, 'shared');
    """)
    assert m.num("n") == 7 and m.num("isax") and m.num("self") and m.num("none") and m.num("isax2")
    assert m.num("v") == 4 and m.num("has") and m.num("miss") and not m.num("has2")
    assert m.val("s").fields == ["count", "name"]
    assert m.num("k") == 3 and m.num("e0") and m.num("r") == 11 and m.num("r2") == 11
    with pytest.raises(MatlabError):
        m.run("rmappdata(0, 'not_there')")


# -- colour limits, aspect, no-ops -----------------------------------------------------------------------

def test_caxis_clim(m):
    m.run("""
    imagesc(magic(4)); c0 = caxis; caxis([0 10]); c1 = caxis; ax = gca; c2 = ax.CLim;
    clim(ax, [2 4]); c3 = clim; caxis auto; c4 = caxis; caxis('manual');
    """)
    assert m.num("c0").tolist() == [[1, 16]] and m.num("c1").tolist() == [[0, 10]]
    assert m.num("c2").tolist() == [[0, 10]] and m.num("c3").tolist() == [[2, 4]]
    assert m.num("c4").tolist() == [[1, 16]]
    with pytest.raises(MatlabError):
        m.run("caxis([3 1])")


def test_daspect_pbaspect(m):
    m.run("""
    plot(1:10); d0 = daspect; daspect([1 2 1]); d = daspect; md = daspect('mode');
    pbaspect([2 1 1]); p = pbaspect; pbaspect auto; daspect auto; md2 = daspect('mode');
    figure; surf(peaks(10)); pbaspect([1 1 2]); daspect([1 1 1]);
    """)
    assert m.num("d0").tolist() == [[1, 1, 1]] and m.num("d").tolist() == [[1, 2, 1]]
    assert m.num("p").tolist() == [[2, 1, 1]] and m.str("md") == "manual" and m.str("md2") == "auto"
    with pytest.raises(MatlabError):
        m.run("daspect([1 0 1])")


def test_accepted_commands_do_nothing(m):
    out = m.run("""
    surf(peaks(10)); lighting gouraud; camlight; camlight('headlight'); material shiny;
    h = light('Position', [1 0 1]); e = isempty(h); zoom on; zoom(2); pan off; rotate3d on;
    datacursormode on; refreshdata; drawnow limitrate; drawnow nocallbacks; z = zoom;
    """)
    assert out == "" and m.num("e")


def test_fig_files_and_gtext(m):
    for code in ("hgsave(gcf, 'a.fig')", "hgload('a.fig')", "openfig('a.fig')", "savefig('a.fig')"):
        with pytest.raises(MatlabError, match="fig"):
            m.run(code)
    out = m.run("plot(1:3); t = gtext('here'); s = t.String;")
    assert "Warning" in out and m.str("s") == "here"


def test_linkaxes(m):
    m.run("""
    a(1) = subplot(2, 1, 1); plot(1:10); a(2) = subplot(2, 1, 2); plot(20:30, 1:11);
    linkaxes(a, 'x'); x1 = xlim(a(1)); x2 = xlim(a(2)); y1 = ylim(a(1)); y2 = ylim(a(2));
    xlim(a(1), [0 5]); x3 = xlim(a(2)); linkaxes(a, 'off'); linkaxes(a);
    """)
    assert m.num("x1").tolist() == m.num("x2").tolist()
    assert m.num("x1")[0, 0] <= 1 and m.num("x1")[0, 1] >= 30
    assert m.num("y1").tolist() != m.num("y2").tolist()
    assert m.num("x3").tolist() == [[0, 5]]
    with pytest.raises(MatlabError):
        m.run("linkaxes(gcf)")


# -- contour matrix, clabel, plotyy, polygons ----------------------------------------------------------

def test_contour_matrix_and_clabel(m):
    m.run("""
    [X, Y] = meshgrid(-2:0.25:2); Z = X.^2 + Y.^2;
    [C, h] = contour(X, Y, Z, [1 2]); t = h.Type; lev = C(1, 1); n = C(2, 1);
    clabel(C, h); clabel(C); clabel(C, h, 'FontSize', 12);
    nt = numel(findobj(gca, 'Type', 'text'));
    """)
    C = m.val("C")
    assert C.shape[0] == 2 and m.str("t") == "contour" and m.num("lev") == 1
    n = int(m.num("n").flat[0])
    seg = C[:, 1:1 + n]
    assert np.allclose(np.hypot(seg[0], seg[1]), 1.0, atol=0.05)  # the level-1 line is the unit circle
    assert C[0, 1 + n] == 2
    assert m.num("nt") > 0
    with pytest.raises(MatlabError):
        m.run("figure; clabel([])")


def test_plotyy(m):
    m.run("""
    x = 1:10; [ax, h1, h2] = plotyy(x, x.^2, x, sqrt(x)); n = numel(ax);
    y1 = ylim(ax(1)); y2 = ylim(ax(2)); c1 = h1.Color; c2 = h2.Color; t = h2.Type;
    ylabel(ax(2), 'right'); figure; plotyy(x, x, x, 10.^x, 'plot', 'semilogy'); s = get(gca, 'YScale');
    """)
    assert m.num("n") == 2 and m.num("y1")[0, 1] >= 100 and m.num("y2")[0, 1] < 5
    assert m.num("c1").tolist() != m.num("c2").tolist() and m.str("t") == "line"
    with pytest.raises(MatlabError):
        m.run("plotyy(1:3, 1:3)")


def test_fill3_and_patch_forms(m):
    m.run("""
    h = fill3([0 1 1 0], [0 0 1 1], [0 0 1 1], 'r'); t = h.Type; z = zlim;
    hold on; p = patch([0 1 0], [0 0 1], [2 2 2], 'b'); z2 = zlim;
    q = patch('Faces', [1 2 3; 1 3 4], 'Vertices', [0 0 3; 1 0 3; 1 1 4; 0 1 4], 'FaceColor', 'g', ...
              'FaceAlpha', 0.5, 'EdgeColor', 'none'); z3 = zlim;
    figure; a = patch('Faces', [1 2 3], 'Vertices', [0 0; 4 0; 2 3], 'FaceVertexCData', 5, 'FaceColor', 'flat');
    b = patch('XData', [5 6 5], 'YData', [0 0 1], 'FaceColor', [1 0 1]); x = xlim; fc = b.FaceColor;
    c = patch([0 1 0], [0 0 1], 'y'); n = numel(findobj(gca, 'Type', 'patch'));
    """)
    assert m.str("t") == "patch" and m.num("z")[0, 1] >= 1
    assert m.num("z2")[0, 1] >= 2 and m.num("z3")[0, 1] >= 4
    assert m.num("x")[0, 0] <= 0 and m.num("x")[0, 1] >= 6 and m.num("fc").tolist() == [[1, 0, 1]]
    assert m.num("n") == 3
    with pytest.raises(MatlabError):
        m.run("patch('Faces', [1 2 9], 'Vertices', [0 0; 1 0; 1 1])")
    with pytest.raises(MatlabError):
        m.run("patch('Faces', [1 2 3])")


# -- function plots --------------------------------------------------------------------------------------

def test_ezplot_forms(m):
    m.run("""
    h = ezplot('x^2 - 1'); t = get(get(gca, 'Title'), 'String'); xl = xlim; y = h.YData;
    figure; h2 = ezplot(@(x) sin(x), [0 pi]); x2 = h2.XData;
    figure; c = ezplot('x^2 + y^2 = 4'); ct = c.Type;
    figure; ezplot(@(x, y) x.^2 - y, [-1 1 -1 1]);
    figure; pc = ezplot(@(t) cos(t), @(t) sin(t)); px = pc.XData;
    figure; fi = fimplicit(@(x, y) x.^2 + y.^2 - 1, [-2 2]); ft = fi.Type;
    """)
    assert m.str("t") == "x^2 - 1" and np.allclose(m.num("xl"), [[-2 * np.pi, 2 * np.pi]])
    assert m.num("y").min() == pytest.approx(-1, abs=1e-3)
    assert m.num("x2")[0, 0] == 0 and m.num("x2")[0, -1] == pytest.approx(np.pi)
    assert m.str("ct") == "contour" and m.str("ft") == "contour"
    assert m.num("px").max() == pytest.approx(1)
    with pytest.raises(MatlabError):
        m.run("ezplot(5)")


def test_surface_function_plots(m):
    m.run("""
    s = fsurf(@(x, y) x.^2 + y.^2); z = s.ZData; x = s.XData;
    figure; s2 = fsurf(@(x, y) x + y, [0 1 0 2], 'MeshDensity', 11); x2 = s2.XData; y2 = s2.YData;
    figure; fmesh(@(x, y) sin(x) .* y); figure; c = fcontour(@(x, y) x.^2 - y.^2, [-2 2]); ct = c.Type;
    figure; e = ezsurf('x*exp(-x^2 - y^2)', [-2 2]); t = get(get(gca, 'Title'), 'String'); ez = e.ZData;
    figure; ezmesh(@(x, y) x .* y); figure; ezcontour('x^2 + y'); figure; ezcontourf(@(x, y) x - y);
    figure; fsurf(@(u, v) cos(u), @(u, v) sin(u), @(u, v) v, [0 2*pi 0 1]);
    figure; k = fsurf(@(x, y) 3); kz = k.ZData;
    """)
    assert m.num("z").shape == (50, 50) and m.num("z").max() == pytest.approx(50)
    assert m.num("x").min() == -5 and m.num("x2").shape == (11, 11)
    assert m.num("x2").max() == 1 and m.num("y2").max() == 2
    assert m.str("ct") == "contour" and m.str("t") == "x*exp(-x^2 - y^2)"
    assert m.num("ez").max() == pytest.approx(np.exp(-0.5) / np.sqrt(2), rel=1e-2)
    assert np.all(m.num("kz") == 3)


def test_comet_waterfall_ribbon_copyobj(m):
    m.run("""
    h = comet(1:5, (1:5).^2); t = h.Type; y = h.YData; comet(rand(1, 5)); comet(1:5, 1:5, 0.2);
    figure; c3 = comet3(1:4, 1:4, 1:4); zt = c3.ZData;
    figure; w = waterfall(peaks(12)); wt = w.Type; waterfall(1:12, 1:12, peaks(12));
    figure; r = ribbon(rand(6, 3)); nr = numel(r); ribbon((1:6)', rand(6, 2), 0.5);
    figure; p = plot(1:3, [4 5 6], 'r--', 'LineWidth', 2); set(p, 'Tag', 'src');
    figure; q = copyobj(p, gca); qy = q.YData; qc = q.Color; qt = q.Tag; samefig = isequal(ancestor(q, 'figure'), gcf);
    """)
    assert m.str("t") == "line" and m.num("y").tolist() == [[1, 4, 9, 16, 25]]
    assert m.num("zt").tolist() == [[1, 2, 3, 4]] and m.str("wt") == "surface" and m.num("nr") == 3
    assert m.num("qy").tolist() == [[4, 5, 6]] and m.num("qc").tolist() == [[1, 0, 0]]
    assert m.str("qt") == "src" and m.num("samefig")
    with pytest.raises(MatlabError):
        m.run("b = bar(1:3); copyobj(b, gca)")


def test_ishandle_for_other_handle_objects(m, tmp_path):
    m.run(f"v = VideoWriter('{tmp_path / 'a.gif'}'); a = ishandle(v); delete(v); b = ishandle(v); c = isvalid(v);")
    assert m.num("a") and not m.num("b") and not m.num("c")
