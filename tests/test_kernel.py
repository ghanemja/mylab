"""The Jupyter kernel, started as a real kernel process."""

import pytest

pytest.importorskip("ipykernel")
jupyter_client = pytest.importorskip("jupyter_client")


@pytest.fixture(scope="module")
def kernel(tmp_path_factory):
    import os

    from jupyter_client.manager import start_new_kernel

    from mylab.kernel import install

    prefix = tmp_path_factory.mktemp("jupyter")
    install(prefix=str(prefix))
    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(prefix / "share" / "jupyter")
    km, kc = start_new_kernel(kernel_name="mylab", startup_timeout=60)
    yield kc
    kc.stop_channels()
    km.shutdown_kernel(now=True)
    if old is None:
        os.environ.pop("JUPYTER_PATH", None)
    else:
        os.environ["JUPYTER_PATH"] = old


def _reply(kc, msg_id):
    while True:
        msg = kc.get_shell_msg(timeout=30)
        if msg["parent_header"].get("msg_id") == msg_id:
            return msg


def _run(kc, code):
    msg_id = kc.execute(code)
    outputs = []
    while True:
        msg = kc.get_iopub_msg(timeout=30)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
            break
        outputs.append(msg)
    return _reply(kc, msg_id)["content"], outputs


def test_execute_output_and_state(kernel):
    reply, outputs = _run(kernel, "x = 6 * 7")
    assert reply["status"] == "ok"
    assert any(o["msg_type"] == "stream" and "x = 42" in o["content"]["text"] for o in outputs)
    reply, outputs = _run(kernel, "disp(x + 1)")
    assert any("43" in o["content"].get("text", "") for o in outputs)


def test_error_and_plot(kernel):
    reply, outputs = _run(kernel, "a = 1;\nnot_defined_zz")
    assert reply["status"] == "error" and "line 2" in reply["traceback"][0]
    reply, outputs = _run(kernel, "plot(1:4)")
    assert any(o["msg_type"] == "display_data" and "image/png" in o["content"]["data"] for o in outputs)


def test_complete_inspect_is_complete(kernel):
    reply = _reply(kernel, kernel.complete("linsp", 5))
    assert "linspace" in reply["content"]["matches"]
    reply = _reply(kernel, kernel.inspect("zeros", 3))
    assert reply["content"]["found"]
    reply = _reply(kernel, kernel.is_complete("for k = 1:3"))
    assert reply["content"]["status"] == "incomplete"
