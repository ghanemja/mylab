"""Help text for animation, movie and additional graphics functions."""

_NOOP = ("Accepted so that scripts written for a desktop plot window keep running; it has no visible effect "
         "in mylab.")
_FIG = "The .fig figure format is not supported; this raises an error. Save figures with saveas, print or " \
       "exportgraphics as .png, .pdf or .svg instead."
_DRAWNOW = (["drawnow", "drawnow limitrate"],
            "Updates figure windows so that pending drawing appears, e.g. inside an animation loop. Options "
            "such as limitrate are accepted. Where there is no live window (browser app, Jupyter, MCP, "
            "--no-plots) each call records a snapshot of the current figure, and the snapshots of a run "
            "are returned as an animated GIF. shg and refresh do the same.")

HELP = {
    # -- frames and movies ------------------------------------------------------------------------
    "getframe": (["F = getframe", "F = getframe(ax)", "F = getframe(fig)", "F = getframe(h, rect)"],
                 "Captures the current axes (or the given axes or whole figure) as a movie frame: a struct "
                 "with cdata, a height-by-width-by-3 uint8 image, and an empty colormap. rect = [left "
                 "bottom width height] in pixels keeps part of the picture. Example: F(k) = getframe(gcf) "
                 "inside a loop collects frames for movie or writeVideo."),
    "frame2im": (["img = frame2im(F)", "[X, map] = frame2im(F)"],
                 "Returns the image stored in a movie frame (its cdata), and the frame's colormap as a "
                 "second output."),
    "im2frame": (["F = im2frame(img)", "F = im2frame(X, map)"],
                 "Wraps an image in a movie frame struct with the fields cdata and colormap. RGB images given "
                 "as doubles between 0 and 1 are stored as uint8; an indexed image keeps its colormap."),
    "movie": (["movie(F)", "movie(F, n)", "movie(F, n, fps)"],
              "Plays the frames in F (from getframe or im2frame) n times at fps frames per second in a plot "
              "window. Without a live window it returns at once: the frames were already recorded as an "
              "animation while they were drawn."),
    "VideoWriter": (["v = VideoWriter(filename)", "v = VideoWriter(filename, profile)"],
                    "Creates an object that writes frames to a video file. Typical use: open(v); "
                    "writeVideo(v, getframe(gcf)) in a loop; close(v). Properties: FrameRate (default 30), "
                    "Quality (0-100), and read-only Filename, Path, FrameCount, Width, Height, Duration. "
                    "Profiles 'Motion JPEG AVI' (default, .avi) and 'MPEG-4' (.mp4) need the ffmpeg program; "
                    "when it is missing an animated GIF with the same base name is written and a warning "
                    "says so. A file name ending in .gif always writes an animated GIF. The file is "
                    "produced by close(v)."),
    "writeVideo": (["writeVideo(v, F)", "writeVideo(v, img)"],
                   "Appends to the open VideoWriter v: a frame or array of frames from getframe, an image "
                   "(height-by-width or height-by-width-by-3, uint8 or doubles from 0 to 1), or a 4-D stack "
                   "of images. Later frames are resized to the size of the first one."),
    "open": (["open(v)", "open(filename)"],
             "open(v) starts a VideoWriter file so that writeVideo can add frames. open(filename) loads a "
             ".mat file; other files are not opened because mylab has no editor windows, and .fig files "
             "are not supported."),
    "pause": (["pause(n)", "pause"],
              "Waits n seconds (at most an hour). pause with no argument, or with text such as 'on', returns "
              "immediately instead of waiting for a key press. With a figure open it also refreshes the "
              "plot; where there is no live window it records an animation frame like drawnow and waits "
              "only briefly."),
    "drawnow": _DRAWNOW, "shg": _DRAWNOW, "refresh": _DRAWNOW,
    "close": (["close", "close(n)", "close(f)", "close(name)", "close all", "close(v)", "close(h)"],
              "Closes the current figure, figure number n, the figure with handle f or with that Name, or "
              "all figures. close(v) finishes the file of a VideoWriter, and close(h) dismisses a waitbar "
              "or message box."),
    "animatedline": (["h = animatedline", "h = animatedline(x, y)", "h = animatedline(x, y, z)",
                      "h = animatedline(..., Name, Value)"],
                     "Creates a line that grows as points are added with addpoints, without clearing the "
                     "axes. Options: Color, LineWidth, LineStyle, Marker, MaximumNumPoints (older points are "
                     "dropped beyond this number). Example: h = animatedline; for k = 1:50, addpoints(h, k, "
                     "sin(k/5)); drawnow; end"),
    "addpoints": (["addpoints(h, x, y)", "addpoints(h, x, y, z)"],
                  "Appends points to the animated line h and grows the axes limits to include them."),
    "clearpoints": ("clearpoints(h)", "Removes all points from the animated line h."),
    "getpoints": (["[x, y] = getpoints(h)", "[x, y, z] = getpoints(h)"],
                  "Returns the points currently stored in the animated line h as row vectors."),
    # -- axes and shapes ---------------------------------------------------------------------------
    "axes": (["axes", "axes('Position', [left bottom width height])", "axes(ax)", "ax = axes(...)"],
             "Creates new axes in the current figure and makes them current; Position is given as "
             "fractions of the figure (use it for insets), and other axes properties can follow as "
             "name/value pairs. axes(ax) makes existing axes current. A figure handle as first input or "
             "'Parent' chooses the figure."),
    "rectangle": (["rectangle('Position', [x y w h])", "rectangle(..., 'Curvature', c)", "h = rectangle(...)"],
                  "Adds a rectangle with lower-left corner (x, y), width w and height h to the current axes. "
                  "Curvature rounds the corners: 0 is square, [1 1] gives an ellipse (a circle when w equals "
                  "h). Options: FaceColor, EdgeColor, LineWidth, LineStyle. Example: rectangle('Position', "
                  "[0 0 2 2], 'Curvature', [1 1], 'FaceColor', 'r'); axis equal"),
    "fill3": (["fill3(x, y, z, c)", "h = fill3(...)"],
              "Draws filled polygons in 3-D with corners (x(k), y(k), z(k)); matrix inputs draw one polygon "
              "per column. c is a colour name, an RGB triplet, or numbers mapped through the colormap. "
              "Options: FaceColor, EdgeColor, FaceAlpha, LineWidth."),
    "patch": (["patch(x, y, c)", "patch(x, y, z, c)", "patch('Faces', F, 'Vertices', V, ...)",
               "patch('XData', x, 'YData', y, ...)", "h = patch(...)"],
              "Adds filled polygons to the current axes without clearing them. Polygons are given by corner "
              "coordinates, or by a vertex list V (one row per point, 2 or 3 columns) and a face list F "
              "whose rows hold vertex numbers. Options: FaceColor, EdgeColor, FaceAlpha, LineWidth, "
              "FaceVertexCData (one number or RGB row per face)."),
    "plotyy": (["plotyy(x1, y1, x2, y2)", "plotyy(x1, y1, x2, y2, fun)", "[ax, h1, h2] = plotyy(...)"],
               "Plots y1 against the left y-axis and y2 against a second y-axis on the right. fun names the "
               "plotting function ('plot', 'semilogy', 'stem', ...). Returns the two axes and the two "
               "plotted objects. yyaxis is the newer way to do this."),
    "comet": (["comet(y)", "comet(x, y)"],
              "Draws the curve as an ordinary line plot; the moving-head animation is not reproduced."),
    "comet3": ("comet3(x, y, z)", "Draws the 3-D curve as an ordinary plot3 line, without the animation."),
    "waterfall": (["waterfall(Z)", "waterfall(X, Y, Z)"],
                  "Draws each row of Z as a separate line in 3-D, like a mesh without the lines across rows."),
    "ribbon": (["ribbon(Y)", "ribbon(x, Y)", "ribbon(x, Y, width)"],
               "Draws each column of Y as a flat strip in 3-D, side by side, against x (default 1, 2, ...). "
               "width (default 0.75) is the strip width."),
    "clabel": (["clabel(C, h)", "clabel(C)", "clabel(C, h, 'FontSize', n)"],
               "Writes the level values on the lines of a contour plot, where [C, h] came from contour or "
               "contourf. With C alone the most recent contour plot of the current axes is labelled."),
    "contour": (["contour(Z)", "contour(X, Y, Z)", "contour(X, Y, Z, n)", "contour(X, Y, Z, levels)",
                 "h = contour(...)", "[C, h] = contour(...)"],
                "Draws contour lines (lines of equal height) of Z, optionally over the grid X, Y, with n "
                "levels or at the given level values (also 'LevelList'). 'ShowText', 'on' labels the lines; "
                "'LineColor' and 'LineWidth' style them. Replaces the current axes unless hold is on. With "
                "two outputs, C is the contour matrix (for each line a column [level; count] followed by "
                "the x and y coordinates) and h the Contour handle."),
    # -- plots of functions ------------------------------------------------------------------------
    "ezplot": (["ezplot(f)", "ezplot(f, [a b])", "ezplot(fx, fy)", "ezplot('x^2 + y^2 = 4')"],
               "Plots a function given as a handle or as an expression in quotes over [-2*pi, 2*pi] or the "
               "given interval. A function of two variables, or an equation, is drawn as the curve where it "
               "is zero; two functions give the parametric curve (fx(t), fy(t)). Expressions in quotes need "
               "no dots before *, / and ^."),
    "fimplicit": (["fimplicit(f)", "fimplicit(f, [xmin xmax ymin ymax])"],
                  "Draws the curve where f(x, y) equals zero, over [-5, 5] in both directions by default. "
                  "Example: fimplicit(@(x, y) x.^2 + y.^2 - 4)"),
    "fsurf": (["fsurf(f)", "fsurf(f, [xmin xmax ymin ymax])", "fsurf(fx, fy, fz, [umin umax vmin vmax])"],
              "Draws the surface z = f(x, y) over [-5, 5] in x and y, or over the given ranges; three "
              "functions give a parametric surface. 'MeshDensity', n sets the number of grid points per "
              "direction."),
    "fmesh": (["fmesh(f)", "fmesh(f, [xmin xmax ymin ymax])"],
              "Like fsurf, but draws a wireframe mesh of z = f(x, y)."),
    "fcontour": (["fcontour(f)", "fcontour(f, [xmin xmax ymin ymax])"],
                 "Draws contour lines of f(x, y) over [-5, 5] in x and y, or over the given ranges."),
    "ezsurf": (["ezsurf(f)", "ezsurf(f, [a b])", "ezsurf('x*exp(-x^2 - y^2)')"],
               "Draws the surface z = f(x, y) for a function handle or an expression in quotes, over "
               "[-2*pi, 2*pi] by default, and titles the plot with the expression. ezsurfc draws the same "
               "plot."),
    "ezmesh": (["ezmesh(f)", "ezmesh(f, [a b])"], "Like ezsurf, but draws a wireframe mesh."),
    "ezcontour": (["ezcontour(f)", "ezcontour(f, [a b])"],
                  "Draws contour lines of f(x, y), a handle or an expression in quotes, over [-2*pi, 2*pi] "
                  "by default. ezcontourf fills the areas between the lines."),
    # -- finding and organising objects ---------------------------------------------------------
    "findobj": (["h = findobj", "h = findobj('Type', 'line')", "h = findobj(parent, Name, Value, ...)",
                 "h = findobj(parent, '-depth', d, ...)", "h = findobj(parent, 'flat', ...)"],
                "Returns the graphics objects whose properties have the given values, searching all figures "
                "or only parent and the objects inside it. Several name/value pairs must all match; "
                "'-or' and '-not' combine conditions, '-regexp' matches text with a regular expression, "
                "'-property' tests that a property exists, and '-depth' limits how deep the search goes "
                "('flat' is depth 0). The result is empty when nothing matches. findall does the same. "
                "Example: set(findobj(gca, 'Type', 'line'), 'LineWidth', 2)"),
    "allchild": ("c = allchild(h)", "The objects directly inside the figure, axes or other graphics object h."),
    "ancestor": (["p = ancestor(h, type)", "p = ancestor(h, {type1, type2})", "p = ancestor(h, type, 'toplevel')"],
                 "Walks up from h through its parents and returns the first object of the given type, such "
                 "as 'axes' or 'figure' (h itself when it has that type), or [] when there is none. "
                 "'toplevel' returns the highest match instead."),
    "copyobj": ("new = copyobj(h, ax)",
                "Copies 2-D line objects into the axes ax and returns the copies. Other kinds of object "
                "cannot be copied in mylab."),
    "linkaxes": (["linkaxes(axs)", "linkaxes(axs, 'x')", "linkaxes(axs, 'y')", "linkaxes(axs, 'off')"],
                 "Gives the listed axes the same x and/or y limits (the range covering all of them) and "
                 "keeps them in step when one of them changes. 'off' is accepted but does not unlink."),
    "setappdata": ("setappdata(h, name, value)",
                   "Stores a value under a name on the graphics object h (a figure, axes, ... or 0 for the "
                   "root). Read it back with getappdata."),
    "getappdata": (["value = getappdata(h, name)", "s = getappdata(h)"],
                   "Returns the value stored on h under name by setappdata, or [] when there is none. With "
                   "one input, returns everything stored on h as a struct."),
    "isappdata": ("tf = isappdata(h, name)", "True when setappdata stored something under name on h."),
    "rmappdata": ("rmappdata(h, name)", "Removes the value stored under name on h; an error if it is missing."),
    "guidata": (["guidata(h, data)", "data = guidata(h)"],
                "Stores or returns one value kept with the figure that contains h. mylab keeps the data but "
                "cannot run GUI callbacks."),
    # -- colour limits and proportions ---------------------------------------------------------------
    "caxis": (["caxis([cmin cmax])", "caxis auto", "lims = caxis", "caxis(ax, ...)"],
              "Sets the data values mapped to the first and last colormap colours of the current axes "
              "(surfaces, images, filled contours and scatter colours); 'auto' returns to the data range, "
              "and with no input the current limits are returned. clim is the newer name. Call it after "
              "plotting."),
    "daspect": (["daspect([dx dy dz])", "d = daspect", "daspect auto"],
                "Sets the relative length of one data unit along x, y and z; daspect([1 1 1]) draws circles "
                "as circles, like axis equal. In 3-D only equal ratios have an effect."),
    "pbaspect": (["pbaspect([px py pz])", "p = pbaspect", "pbaspect auto"],
                 "Sets the relative lengths of the sides of the plot box; pbaspect([1 1 1]) makes it square "
                 "and pbaspect([2 1 1]) twice as wide as high."),
    "lighting": ("lighting mode", _NOOP + " Surfaces keep their flat colormap shading."),
    "camlight": ("camlight", _NOOP + " An output, if requested, is []."),
    "light": ("light('Position', p)", _NOOP + " An output, if requested, is []."),
    "material": ("material name", _NOOP),
    "zoom": (["zoom on", "zoom off", "zoom(factor)"], _NOOP),
    "pan": (["pan on", "pan off"], _NOOP),
    "rotate3d": (["rotate3d on", "rotate3d off"], _NOOP + " Use view(az, el) to turn a 3-D plot."),
    "datacursormode": (["datacursormode on", "datacursormode off"], _NOOP),
    "refreshdata": ("refreshdata", _NOOP + " Update a plot by setting XData and YData on its handle."),
    "gtext": ("gtext(str)",
              "Places text where the mouse is clicked in a plot window. Without a live window the text is "
              "put in the middle of the axes and a warning suggests text(x, y, str)."),
    "hgsave": ("hgsave(h, filename)", _FIG),
    "hgload": ("h = hgload(filename)", _FIG),
    "openfig": ("openfig(filename)", _FIG),
    "savefig": ("savefig(filename)", _FIG),
}
HELP["findall"] = HELP["findobj"]
HELP["clim"] = HELP["caxis"]
for _n in ("ezsurfc",):
    HELP[_n] = HELP["ezsurf"]
HELP["ezcontourf"] = HELP["ezcontour"]
