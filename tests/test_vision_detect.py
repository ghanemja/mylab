"""Box utilities, cascade and HOG detectors, ONNX detectors and detection metrics."""

import numpy as np
import pytest

from mylab.errors import MatlabError

# A one-stage, one-feature Haar cascade written for this test: it fires on 12x12 windows whose left half is
# brighter than the right half.
TINY_CASCADE = """<?xml version="1.0"?>
<opencv_storage>
<cascade>
  <stageType>BOOST</stageType>
  <featureType>HAAR</featureType>
  <height>12</height>
  <width>12</width>
  <stageParams><boostType>GAB</boostType><minHitRate>0.995</minHitRate><maxFalseAlarm>0.5</maxFalseAlarm>
    <weightTrimRate>0.95</weightTrimRate><maxDepth>1</maxDepth><maxWeakCount>10</maxWeakCount></stageParams>
  <featureParams><maxCatCount>0</maxCatCount><featSize>1</featSize><mode>BASIC</mode></featureParams>
  <stageNum>1</stageNum>
  <stages>
    <_>
      <maxWeakCount>1</maxWeakCount>
      <stageThreshold>0.5</stageThreshold>
      <weakClassifiers>
        <_><internalNodes>0 -1 0 0.2</internalNodes><leafValues>-1. 1.</leafValues></_></weakClassifiers></_>
  </stages>
  <features>
    <_><rects><_>0 0 12 12 -1.</_><_>0 0 6 12 2.</_></rects></_>
  </features>
</cascade>
</opencv_storage>
"""


def test_bbox_overlap_ratio(m):
    m.run("A = [1 1 10 10; 5 5 10 10]; B = [1 1 10 10; 100 100 5 5; 6 1 10 10];"
          "r = bboxOverlapRatio(A, B); q = bboxOverlapRatio(A, B, 'Min');")
    r = m.num("r")
    assert r.shape == (2, 3)
    assert r[0, 0] == 1 and r[0, 1] == 0
    assert np.isclose(r[0, 2], 50 / 150)
    assert np.isclose(r[1, 0], 36 / 164)
    assert np.isclose(m.num("q")[1, 0], 36 / 100)


def test_select_strongest_bbox(m):
    m.run("bb = [10 10 20 20; 12 12 20 20; 50 50 10 10; 52 50 10 10]; sc = [0.9; 0.8; 0.7; 0.95];"
          "[sb, ss, idx] = selectStrongestBbox(bb, sc);"
          "[~, ~, idx2] = selectStrongestBbox(bb, sc, 'OverlapThreshold', 0.9);"
          "[~, ~, idx3] = selectStrongestBbox(bb, sc, 'NumStrongest', 1);"
          "[mb, ms, ml, midx] = selectStrongestBboxMulticlass(bb, sc, categorical({'a';'b';'a';'a'}));")
    assert m.num("idx").ravel().tolist() == [1, 4]
    assert m.num("sb").tolist() == [[10, 10, 20, 20], [52, 50, 10, 10]]
    assert m.num("idx2").ravel().tolist() == [1, 2, 3, 4]
    assert m.num("idx3").ravel().tolist() == [4]
    assert m.num("midx").ravel().tolist() == [1, 2, 4]
    assert m.str("class(ml)") == "categorical"


def test_bbox_precision_recall(m):
    m.run("[p, r] = bboxPrecisionRecall([1 1 10 10; 50 50 5 5], [2 2 10 10]);"
          "[p2, r2] = bboxPrecisionRecall({[1 1 10 10], zeros(0,4)}, {[1 1 10 10], [5 5 5 5]});")
    assert m.num("p").item() == 0.5 and m.num("r").item() == 1
    assert m.num("p2").item() == 1 and m.num("r2").item() == 0.5


def test_evaluate_detection_precision(m):
    m.run("""
        Boxes = {[1 1 10 10; 30 30 10 10]; [5 5 10 10]};
        Scores = {[0.9; 0.4]; 0.8};
        det = table(Boxes, Scores);
        car = {[1 1 10 10]; [50 50 10 10]};
        gt = table(car);
        [ap, rec, prec] = evaluateDetectionPrecision(det, gt);
    """)
    # ranked: 0.9 hit, 0.8 miss, 0.4 miss; two ground-truth boxes
    assert np.isclose(m.num("ap").item(), 0.5)
    assert np.allclose(m.num("rec").ravel(), [0, 0.5, 0.5, 0.5])
    assert np.allclose(m.num("prec").ravel(), [1, 1, 0.5, 1 / 3])


def test_cascade_detector_with_custom_xml(m, tmp_path):
    pytest.importorskip("cv2")
    xml = tmp_path / "tiny.xml"
    xml.write_text(TINY_CASCADE)
    m.interp.set_var("xmlfile", np.array([list(str(xml))]))
    m.run("""
        I = 120*ones(120, 160, 'uint8'); I(40:63, 60:71) = 220; I(40:63, 72:83) = 30;
        d = vision.CascadeObjectDetector(xmlfile, 'MergeThreshold', 2);
        b = d(I); b2 = step(d, I); ts = d.TrainingSize;
        d.MinSize = [40 40]; b3 = d(I);
    """)
    b = m.num("b")
    assert b.shape[0] >= 1 and b.shape[1] == 4
    cx, cy = b[0, 0] + b[0, 2] / 2, b[0, 1] + b[0, 3] / 2
    assert abs(cx - 72) < 3 and abs(cy - 52) < 3
    assert np.array_equal(b, m.num("b2"))
    assert m.num("ts").ravel().tolist() == [12, 12]
    b3 = m.num("b3")
    assert b3.size == 0 or b3[:, 2:].min() >= 40


def test_cascade_model_names(m, monkeypatch):
    pytest.importorskip("cv2")
    from mylab.builtins import vision_detect
    with pytest.raises(MatlabError, match="Nose"):
        m.run("vision.CascadeObjectDetector('Nose')")
    with pytest.raises(MatlabError, match="neither a known model"):
        m.run("vision.CascadeObjectDetector('NoSuchModel')")
    path = vision_detect.find_cascade("haarcascade_frontalface_alt2.xml")
    if path is None:
        with pytest.raises(MatlabError, match="MYLAB_HAARCASCADES"):
            m.run("vision.CascadeObjectDetector('FrontalFaceCART')")
    else:
        m.run("d = vision.CascadeObjectDetector('UpperBody'); b = d(zeros(100, 'uint8'));")
        assert m.num("isempty(b)").item()


def test_people_detector_runs(m):
    pytest.importorskip("cv2")
    m.run("pd = vision.PeopleDetector; [b, s] = pd(zeros(200, 100, 'uint8'));"
          "p2 = peopleDetector; [b2, s2] = detect(p2, zeros(64, 32, 'uint8'));"
          "pd.ClassificationThreshold = 2;")
    assert m.num("size(b)").ravel().tolist() == [0, 4]
    assert m.num("size(b2)").ravel().tolist() == [0, 4]
    assert "UprightPeople_128x64" in m.run("pd")


def _tiny_onnx(path, rows):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    const = np.asarray(rows, dtype=np.float32)[None]
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(const.shape))
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["s"], keepdims=0),
        helper.make_node("Constant", [], ["zero"], value=numpy_helper.from_array(np.array(0, np.float32))),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Constant", [], ["c"], value=numpy_helper.from_array(const)),
        helper.make_node("Add", ["c", "z"], ["output"]),
    ]
    model = helper.make_model(helper.make_graph(nodes, "tiny", [x], [y]),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_onnx_yolo_detector(m, tmp_path):
    pytest.importorskip("cv2")
    p = tmp_path / "tiny.onnx"
    # rows: cx cy w h objectness score(cat) score(dog), in network (32x32) pixels
    _tiny_onnx(p, [[16, 16, 8, 8, 0.9, 0.95, 0.05], [8, 24, 6, 4, 0.8, 0.1, 0.9], [17, 16, 8, 8, 0.85, 0.9, 0.1],
                   [5, 5, 2, 2, 0.1, 0.5, 0.5]])
    m.interp.set_var("f", np.array([list(str(p))]))
    m.run("""
        det = yolov4ObjectDetector(f, {'cat', 'dog'}, 'InputSize', [32 32]);
        [b, s, l] = detect(det, zeros(64, 64, 3, 'uint8'), 'Threshold', 0.3);
        [b2, s2] = detect(det, zeros(64, 64, 'uint8'), 'Threshold', 0.3, 'SelectStrongest', false);
    """)
    assert np.allclose(m.num("b"), [[25, 25, 16, 16], [11, 45, 12, 8]])
    assert np.allclose(m.num("s").ravel(), [0.855, 0.72], atol=1e-6)
    assert m.str("class(l)") == "categorical"
    assert m.num("double(l)").ravel().tolist() == [1, 2]
    assert m.num("size(b2, 1)").item() == 3


def test_onnx_detector_needs_a_model_file(m):
    pytest.importorskip("cv2")
    with pytest.raises(MatlabError, match="pretrained"):
        m.run("yolov4ObjectDetector('csp-darknet53-coco')")
    with pytest.raises(MatlabError, match="not found"):
        m.run("yoloxObjectDetector('missing.onnx', {'a'})")
