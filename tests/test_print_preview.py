"""print accepts -tiff, the request for a preview image inside an EPS file (written without one)."""


def test_eps_with_tiff_preview_option(m, tmp_path):
    f = (tmp_path / "fig.eps").as_posix()
    m.run(f"figure('Visible', 'off'); plot(1:3); print('-depsc', '-tiff', '-r100', '{f}'); close all")
    data = (tmp_path / "fig.eps").read_bytes()
    assert data.startswith(b"%!PS")
