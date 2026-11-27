"""Help text for mylab/builtins/vision_annotate.py (written from scratch for mylab)."""

_OUT = ("The result is an RGB image of the input's class (a grayscale input is copied into all three channels). "
        "Colours are names (blue, green, red, cyan, magenta, yellow, black, white or their one-letter forms), an "
        "RGB triplet in the image's range, or one row per shape. Needs OpenCV.")

HELP = {
    "insertShape": (["J = insertShape(I, shape, position)",
                     "J = insertShape(..., 'Color', c, 'LineWidth', w, 'Opacity', a, 'SmoothEdges', tf)"],
                    "Draws shapes into an image. shape and the rows of position: 'rectangle' or "
                    "'filled-rectangle' [x y width height], 'circle' or 'filled-circle' [x y radius], 'line' "
                    "[x1 y1 x2 y2 ...] (a polyline) and 'polygon' or 'filled-polygon' [x1 y1 x2 y2 ...]. Several "
                    "shapes are given as several rows, or a cell array when their lengths differ. Opacity (default "
                    "0.6) applies to filled shapes; SmoothEdges (default true) anti-aliases. The default colour is "
                    "yellow. " + _OUT),
    "insertMarker": (["J = insertMarker(I, position)", "J = insertMarker(I, position, marker, 'Color', c, 'Size', s)"],
                     "Draws markers at M-by-2 [x y] positions. marker is 'plus' ('+', default), 'circle' ('o'), "
                     "'star' ('*'), 'x-mark' ('x') or 'square' ('s'); Size is the half-width in pixels (default 3). "
                     "The default colour is green. " + _OUT),
    "insertText": (["J = insertText(I, position, text)",
                    "J = insertText(..., 'FontSize', n, 'TextColor', c, 'BoxColor', c, 'BoxOpacity', a, "
                    "'AnchorPoint', anchor)"],
                   "Writes text at M-by-2 positions; text is a char vector, a cell or string array (one per "
                   "position) or numbers. The text sits in a box (BoxColor default yellow, BoxOpacity default 0.6); "
                   "AnchorPoint says which point of the box is at the position ('LeftTop' default, 'Center', "
                   "'RightBottom', ...). OpenCV's Hershey font is used, so the 'Font' option is ignored. " + _OUT),
    "insertObjectAnnotation": (["J = insertObjectAnnotation(I, 'rectangle', position, label)",
                                "J = insertObjectAnnotation(I, 'circle', position, label)",
                                "J = insertObjectAnnotation(..., 'Color', c, 'LineWidth', w, 'FontSize', n, "
                                "'TextColor', c, 'TextBoxOpacity', a)"],
                               "Draws boxes [x y w h] (or circles [x y r]) with a label above each, as used to show "
                               "detector output. label is text, a cell or string array, or numbers such as scores. "
                               + _OUT),
}
