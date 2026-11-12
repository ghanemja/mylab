"""Help text for JSON, audio and Excel functions, categorical arrays and additional plots."""

_WAV = "Only WAV files are supported."
_EXCEL = ("Works with .xlsx workbooks. With the openpyxl package installed (pip install mylab[excel]) it is "
          "used for reading and writing; otherwise a built-in reader and writer handles numbers, text and "
          "logical values without formatting. Dates are read as Excel serial numbers. Old .xls files are "
          "not supported.")
_CAT = "c = categorical({'red', 'blue', 'red'})"

HELP = {
    # -- JSON -----------------------------------------------------------------------------------------
    "jsondecode": ("value = jsondecode(text)",
                   "Converts JSON text to a value. Objects become structs, arrays of numbers become "
                   "column vectors or matrices (null becomes NaN), arrays of objects with the same "
                   "fields become struct arrays, other arrays become cell arrays, true/false become "
                   "logical values and a lone null becomes []. Field names that are not valid "
                   "identifiers are changed (other characters become _). To read a file: "
                   "jsondecode(fileread('data.json'))."),
    "jsonencode": (["text = jsonencode(value)", "text = jsonencode(value, 'PrettyPrint', true)",
                    "text = jsonencode(value, 'ConvertInfAndNaN', false)"],
                   "Converts a value to JSON text. Structs and containers.Map become objects, vectors "
                   "and cell arrays become arrays, matrices become arrays of rows, tables become an array "
                   "with one object per row. NaN and Inf become null unless 'ConvertInfAndNaN' is false. "
                   "'PrettyPrint' adds line breaks and two-space indentation."),

    # -- audio ----------------------------------------------------------------------------------------
    "audioread": (["[y, Fs] = audioread(file)", "[y, Fs] = audioread(file, [first last])",
                   "[y, Fs] = audioread(file, 'native')"],
                  "Reads a sound file: y has one row per sample and one column per channel, scaled to "
                  "the range -1 to 1, and Fs is the sample rate in Hz. [first last] reads part of the "
                  "file; 'native' returns the stored integer type. " + _WAV),
    "audiowrite": (["audiowrite(file, y, Fs)", "audiowrite(file, y, Fs, 'BitsPerSample', n)"],
                   "Writes the samples y (one column per channel, values from -1 to 1) to a WAV file "
                   "with sample rate Fs. BitsPerSample is 8, 16 (default), 24, 32 or 64. Values outside "
                   "-1 to 1 are clipped with a warning. " + _WAV),
    "audioinfo": ("info = audioinfo(file)",
                  "Returns a struct describing a sound file: Filename, NumChannels, SampleRate, "
                  "TotalSamples, Duration (seconds) and BitsPerSample. " + _WAV),
    "wavread": ("[y, Fs, nbits] = wavread(file)", "Older form of audioread for WAV files."),
    "wavwrite": (["wavwrite(y, file)", "wavwrite(y, Fs, file)", "wavwrite(y, Fs, nbits, file)"],
                 "Older form of audiowrite for WAV files (default 8000 Hz, 16 bits)."),
    "sound": (["sound(y)", "sound(y, Fs)"],
              "Checks its arguments and plays nothing: mylab has no audio device. A note is printed the "
              "first time. Use audiowrite to save the sound as a WAV file."),
    "soundsc": (["soundsc(y)", "soundsc(y, Fs)"],
                "Like sound: checks its arguments and plays nothing, because mylab has no audio device."),
    "audioplayer": ("p = audioplayer(y, Fs)",
                    "Creates a player object for the samples y. play(p), playblocking(p), pause(p), "
                    "resume(p) and stop(p) are accepted but play nothing: mylab has no audio device. "
                    "Properties: SampleRate, TotalSamples, NumChannels, BitsPerSample."),

    # -- Excel ----------------------------------------------------------------------------------------
    "xlsread": (["num = xlsread(file)", "num = xlsread(file, sheet)", "num = xlsread(file, sheet, range)",
                 "[num, txt, raw] = xlsread(...)"],
                "Reads a worksheet of an Excel file. num is the block of numbers (NaN where a cell holds "
                "text or nothing; header rows and columns are dropped), txt a cell array with the text "
                "cells and raw a cell array with everything. sheet is a name or number, range is like "
                "'B2:D10'. " + _EXCEL),
    "xlswrite": (["xlswrite(file, A)", "xlswrite(file, A, sheet)", "xlswrite(file, A, sheet, range)"],
                 "Writes a numeric matrix or a cell array of numbers and text to a worksheet of an Excel "
                 "file, starting at cell A1 or at the first cell of range. Other sheets of an existing "
                 "file are kept. " + _EXCEL),
    "sheetnames": ("names = sheetnames(file)", "Returns the worksheet names of an Excel file as a string array."),
    "xlsfinfo": ("[status, sheets] = xlsfinfo(file)",
                 "Returns a description and the worksheet names (cell array) of an Excel file."),

    # -- categorical ------------------------------------------------------------------------------------
    "categorical": (["c = categorical(values)", "c = categorical(values, valueset)",
                     "c = categorical(values, valueset, names)", "c = categorical(..., 'Ordinal', true)"],
                    "Creates a categorical array: each element is one of a fixed list of named "
                    "categories. values can be numbers, logical values, a cell array of text or a string "
                    "array. Without valueset the categories are the sorted distinct values; empty text "
                    "and NaN become <undefined>. names gives the category name for each allowed value. "
                    "Compare with text (c == 'red'), count with countcats, list with categories. Ordinal "
                    "arrays also support < and >. Assigning a new name adds a category. Works as a table "
                    "variable and as a grouping variable for groupsummary, findgroups and grp2idx. "
                    "Example: " + _CAT),
    "iscategorical": ("tf = iscategorical(x)", "True when x is a categorical array."),
    "categories": ("names = categories(c)",
                   "Returns the category names of a categorical array as a cell array (a column), in "
                   "category order. Example: " + _CAT + "; categories(c) gives {'blue'; 'red'}."),
    "countcats": (["n = countcats(c)", "n = countcats(c, dim)"],
                  "Number of elements in each category of a categorical array, in category order."),
    "isundefined": ("tf = isundefined(c)", "True for the elements of a categorical array that have no category."),
    "iscategory": ("tf = iscategory(c, names)", "True for each name that is a category of c."),
    "isordinal": ("tf = isordinal(c)", "True when the categories of c have an order (created with 'Ordinal', true)."),
    "addcats": (["c = addcats(c, names)", "c = addcats(c, names, 'After', existing)"],
                "Adds categories to a categorical array, at the end or next to an existing category. No "
                "element changes."),
    "removecats": (["c = removecats(c)", "c = removecats(c, names)"],
                   "Removes categories: the unused ones, or the named ones (their elements become "
                   "<undefined>)."),
    "renamecats": (["c = renamecats(c, newnames)", "c = renamecats(c, oldnames, newnames)"],
                   "Renames categories of a categorical array."),
    "reordercats": (["c = reordercats(c)", "c = reordercats(c, neworder)"],
                    "Changes the order of the categories (alphabetical by default). The elements keep "
                    "their names."),
    "mergecats": (["c = mergecats(c, oldnames)", "c = mergecats(c, oldnames, newname)"],
                  "Combines several categories into one, named newname (or the first of oldnames)."),
    "setcats": ("c = setcats(c, names)",
                "Sets the categories of c to the given list. Elements whose category is not in the list "
                "become <undefined>."),

    # -- plots ------------------------------------------------------------------------------------------
    "quiver3": (["quiver3(X, Y, Z, U, V, W)", "quiver3(Z, U, V, W)", "quiver3(..., scale)",
                 "h = quiver3(..., 'Color', c)"],
                "Draws 3-D arrows with components (U, V, W) at the points (X, Y, Z). Arrow lengths are "
                "scaled to fit; scale stretches them, and 0 or 'AutoScale', 'off' draws them at true "
                "length."),
    "surface": (["surface(X, Y, Z)", "surface(Z)", "h = surface(..., 'FaceAlpha', a)"],
                "Adds a surface to the current axes without clearing what is already drawn (surf starts "
                "a new plot unless hold is on). Accepts 'XData', 'YData', 'ZData' name-value pairs."),
    "pcolor": (["pcolor(C)", "pcolor(X, Y, C)", "h = pcolor(...)"],
               "Draws the matrix C as a flat grid of coloured cells. Each cell takes the colour of its "
               "lower-left corner, so the last row and column of C are not shown; shading interp blends "
               "the colours. The y-axis points up (unlike imagesc)."),
    "surfl": (["surfl(Z)", "surfl(X, Y, Z)"],
              "Surface plot shaded as if lit from the side. Light direction arguments are ignored."),
    "trisurf": (["trisurf(T, x, y, z)", "h = trisurf(T, x, y, z, 'FaceAlpha', a)"],
                "Draws the triangles listed in T (from delaunay) as a coloured surface through the points "
                "(x, y, z)."),
    "trimesh": ("trimesh(T, x, y, z)",
                "Draws the triangles listed in T (from delaunay) as a mesh through the points (x, y, z)."),
    "triplot": (["triplot(T, x, y)", "triplot(T, x, y, linespec)"],
                "Draws the edges of the 2-D triangulation T (from delaunay)."),
    "stem3": (["stem3(Z)", "stem3(X, Y, Z)", "stem3(..., 'filled')"],
              "Draws 3-D stems: a vertical line from the xy-plane to each point, with a marker on top."),
    "xline": (["xline(x)", "xline(x, linespec)", "xline(x, linespec, label)", "h = xline(..., 'Color', c)"],
              "Draws a vertical line across the current axes at x (several lines for a vector). Options: "
              "a line style such as '--r', a label, 'Color', 'LineWidth', 'LineStyle', 'Alpha'."),
    "yline": (["yline(y)", "yline(y, linespec)", "yline(y, linespec, label)", "h = yline(..., 'Color', c)"],
              "Draws a horizontal line across the current axes at y (several lines for a vector). Options: "
              "a line style such as '--r', a label, 'Color', 'LineWidth', 'LineStyle', 'Alpha'."),
    "alpha": (["alpha(value)", "alpha(h, value)"],
              "Sets the transparency of the surfaces, patches and images in the current axes, or of the "
              "object h: 0 is clear, 1 is opaque. Per-point modes ('flat', 'interp') are ignored."),
    "boxplot": (["boxplot(x)", "boxplot(x, groups)", "boxplot(..., 'Labels', names)"],
                "Box-and-whisker plot. Each column of x is one box, or the values of a vector x are "
                "split by the grouping variable. Options: 'Labels', 'Notch', 'Whisker', 'Widths', "
                "'Positions', 'Orientation' ('horizontal')."),
    "boxchart": ("boxchart(x)", "Box-and-whisker plot; drawn like boxplot in mylab."),
    "gobjects": (["h = gobjects(n)", "h = gobjects(m, n)", "h = gobjects(0)"],
                 "Creates an array that holds graphics handles, to fill in a loop: h = gobjects(1, 3); "
                 "h(k) = plot(...)."),
}
