"""No built-in may accept a name-value option and then ignore it.

Every built-in that parses name-value pairs (found by scanning the built-in modules for the
option helpers, below) must be listed in one of the tables:

* ``CALLS``: a minimal valid call. ``'NoSuchOption__', 1`` is appended and the call must raise
  an error whose message names ``NoSuchOption__``.
* ``LOUD``: the same, but the error does not name the option yet (for example, the unknown name
  is taken for a positional argument and rejected as such). The call must still fail.
* ``KNOWN_IGNORED``: known bugs, the unknown option is still silently accepted. These are
  strict xfails, so fixing one makes the test fail until the entry moves to ``CALLS``.
* ``EXEMPT``: built-ins not checked here, with a one-line reason each.
"""

import inspect
import os
import sys
import types

import pytest

from mylab import Interpreter
from mylab.builtins._registry import BUILTINS
from mylab.errors import MatlabError


# Module -> the functions in it that parse name-value pairs.
OPTION_HELPERS = {
    "_util": {"parse_options", "split_options"},
    "_ode_extra": {"_pairs"},
    "_optim_local": {"_find_options"},
    "_quad": {"_options"},
    "_stats_util": {"get_opts"},
    "blockdiagram": {"_pairs"},
    "control_analysis": {"_fill_options", "_info_options"},
    "data": {"_split_options"},
    "datetimes": {"_split_options"},
    "excel": {"_file_options"},
    "fileformats": {"_parse_options", "_struct_options"},
    "fileops": {"_split_web_options"},
    "graphs": {"_options", "_method_opt"},
    "images": {"_options"},
    "images_more": {"_space_opts"},
    "io": {"_name_value"},
    "ode_more": {"_set_options"},
    "patterns": {"_options"},
    "plot_charts": {"_options"},
    "plotting": {"_split_name_values", "_name_values"},
    "signal": {"_options"},
    "signal_more": {"_name_values"},
    "sparse": {"_iter_options"},
    "spatial": {"_options"},
    "stats_dist": {"_check_options"},
    "sysmore": {"_options"},
    "table": {"_split_opts"},
    "tables_more": {"_options"},
    "uicomponents": {"_pairs"},
    "unittestfw": {"_split_options", "_fixture_options"},
}

CALLS = {
    'adapthisteq': '1:5',
    'adaptthresh': '1:5',
    'adtest': '1:5',
    'animatedline': '',
    'annotation': "'line', [0.1 0.2], [0.1 0.2]",
    'anovan': 'randn(12,1), {[1;1;1;1;2;2;2;2;3;3;3;3]}',
    'array2table': '1:5',
    'axes': '',
    'bandpass': 'rand(100,1), [0.2 0.4]',
    'bandstop': 'rand(100,1), [0.2 0.4]',
    'betafit': 'rand(20,3)',
    'bootci': "50, @mean, (1:10)'",
    'bootstrp': "20, @mean, (1:10)'",
    'bvpset': '',
    'bwskel': '1:5',
    'cell2table': "{'a','b'}",
    'contains': "'abc', 'b'",
    'chi2gof': '1:5',
    'clabel': 'contour(magic(4))',
    'clusterdata': 'magic(4), 2',
    'confusionmat': '[1 2 3], [4 5 6]',
    'contour': 'magic(4)',
    'contour3': 'magic(4)',
    'contourf': 'magic(4)',
    'corr': '1:5',
    'createns': '1:5',
    'crossval': "'mse', (1:10)', (1:10)', 'Predfun', @(xt,yt,xs) xs",
    'cvpartition': "10, 'KFold', 2",
    'dbscan': 'rand(20,2), 0.3, 3',
    'ddeset': '',
    'dendrogram': 'linkage(rand(5,2))',
    'designfilt': "'lowpassfir', 'FilterOrder', 10, 'CutoffFrequency', 0.3",
    'ecdf': '1:5',
    'eigs': 'magic(4)',
    'endsWith': "'abc', 'c'",
    'evalclusters': "rand(20,2), 'kmeans', 'CalinskiHarabasz', 'KList', 2:3",
    'evfit': '1:5',
    'ExhaustiveSearcher': '1:5',
    'expfit': '1:5',
    'factoran': 'randn(50,6), 1',
    'fillmissing': "[1 NaN 3], 'linear'",
    'findchangepts': '1:5',
    'fishertest': '[3 1; 1 3]',
    'fitcdiscr': "(1:10)', (1:10)'",
    'fitcecoc': "(1:10)', (1:10)'",
    'fitcensemble': "(1:10)', (1:10)'",
    'fitcknn': "(1:10)', (1:10)'",
    'fitclinear': '[1 2; 3 4], [1 2]',
    'fitcnb': 'randn(30,2), [ones(15,1); 2*ones(15,1)]',
    'fitcnet': "(1:10)', (1:10)'",
    'fitcsvm': '[1 2; 3 4], [1 2]',
    'fitctree': "(1:10)', (1:10)'",
    'fitdist': "randn(50,1), 'Normal'",
    'fitgmdist': '[randn(30,1); randn(30,1)+5], 2',
    'cfit': "fittype('poly1'), 1, 2",
    'fit': "(1:5)', (2:6)', 'poly1'",
    'fitoptions': "'poly1'",
    'fittype': "'a*x+b'",
    'round': '2.5',
    'sfit': "fittype('poly11'), 1, 2, 3",
    'fitnlm': "(1:10)', 2*(1:10)'+1, @(b,x) b(1)*x+b(2), [1 1]",
    'fitrensemble': "(1:10)', (1:10)'",
    'fitrgp': "(1:10)', (1:10)'",
    'fitrlinear': "(1:10)', (1:10)'",
    'fitrnet': "(1:10)', (1:10)'",
    'fitrsvm': "(1:10)', (1:10)'",
    'fitrtree': "(1:10)', (1:10)'",
    'fplot': '@sin',
    'gamfit': '1:5',
    'geoaxes': '',
    'gevfit': '1:5',
    'glmval': "[1; 2], (1:3)', 'identity'",
    'gpfit': '1:5',
    'groupfilter': "table([1;1;2],[1;2;3]), 'Var1', @(x) true(size(x))",
    'groupsummary': '[1 2 3], [4 5 6]',
    'grouptransform': "table([1;1;2],[1;2;3]), 'Var1', 'meancenter'",
    'hggroup': '',
    'hgtransform': '',
    'highpass': 'rand(100,1), 0.3',
    'histogram2': '[1 2 3], [4 5 6]',
    'hough': '1:5',
    'houghlines': 'eye(5), -90:89, -7:7, [3 3]',
    'houghpeaks': '1:5',
    'image': '',
    'imagesc': '',
    'imbilatfilt': '1:5',
    'imboxfilt': '1:5',
    'imguidedfilter': '1:5',
    'imhistmatch': '[1 2 3], [4 5 6]',
    'imageDatastore': "'img.png'",
    'imread': "'img.png'",
    'imsharpen': '1:5',
    'imshow': '1:5',
    'imtile': '1:5',
    'imwrite': "uint8(magic(4)), 'img2.png'",
    'integral': '@(x) x.^2, 0, 1',
    'integral2': '@(x,y) x.*y, 0, 1, 0, 1',
    'integral3': '@(x,y,z) x.*y.*z, 0, 1, 0, 1, 0, 1',
    'integralBoxFilter': 'magic(4)',
    'istft': '1:5',
    'jackknife': "@mean, (1:10)'",
    'KDTreeSearcher': '1:5',
    'kmeans': 'magic(4), 2',
    'kmedoids': 'magic(4), 2',
    'knnsearch': '[1 2 3], [4 5 6]',
    'ksdensity': '1:5',
    'kstest': '1:5',
    'kstest2': '[1 2 3], [4 5 6]',
    'lab2rgb': 'rand(20,3)',
    'lab2xyz': 'rand(20,3)',
    'lasso': "(1:10)', (1:10)'",
    'light': '',
    'lillietest': '1:5',
    'lognfit': '1:5',
    'lowpass': 'rand(100,1), 0.3',
    'lsqlin': "(1:10)', (1:10)'",
    'matlab.lang.makeValidName': "'abc'",
    'matlab.unittest.constraints.ContainsSubstring': "'abc'",
    'matlab.unittest.constraints.EndsWithSubstring': "'abc'",
    'matlab.unittest.constraints.IsEqualTo': '1:5',
    'matlab.unittest.constraints.IsSubstringOf': "'abc'",
    'matlab.unittest.constraints.IssuesNoWarnings': '',
    'matlab.unittest.constraints.StartsWithSubstring': "'abc'",
    'matlab.unittest.fixtures.PathFixture': '1:5',
    'matlab.unittest.fixtures.TemporaryFolderFixture': '',
    'matlab.unittest.fixtures.WorkingFolderFixture': '',
    'matlab.unittest.selectors.HasParameter': '',
    'matlab.unittest.TestRunner.withDefaultPlugins': '',
    'matlab.unittest.TestRunner.withTextOutput': '',
    'mdscale': "'x', 1",
    'mergevars': 'table([1;2],[3;4]), [1 2]',
    'mesh': '1:5',
    'meshc': 'magic(4)',
    'meshz': '1:5',
    'mle': '1:5',
    'mnrfit': '[1 2 3], [4 5 6]',
    'mnrval': "[1; 2], (1:3)'",
    'montage': '1:5',
    'NaT': '',
    'nbinfit': 'magic(4)',
    'nlinfit': "(1:10)', 2*(1:10)'+1, @(b,x) b(1)*x+b(2), [1 1]",
    'normfit': '1:5',
    'ode': '',
    'odeEvent': '',
    'odeJacobian': '',
    'odeMassMatrix': '',
    'odeset': '',
    'optimalleaforder': '[1 2 3], [4 5 6]',
    'parallelcoords': '1:5',
    'partialcorr': '1:5',
    'pca': '1:5',
    'pdist': '1:5',
    'pdist2': '[1 2 3], [4 5 6]',
    'peaks': '',
    'perfcurve': '[1;1;0;0], [0.9;0.6;0.4;0.1], 1',
    'plsregress': "(1:10)', (1:10)'",
    'poissfit': '1:5',
    'polaraxes': '',
    'polarbubblechart': '1:10, 1:10, 1:10',
    'polarhistogram': '1:5',
    'polyconf': '[1 2 3], [4 5 6]',
    'polyshape': '[0 1 1], [0 0 1]',
    'nsidedpoly': '5',
    'polybuffer': 'polyshape([0 1 1], [0 0 1]), 0.1',
    'addboundary': 'polyshape([0 1 1], [0 0 1]), [2 3 3], [0 0 1]',
    'rmboundary': 'polyshape({[0 1 1], [2 3 3]}, {[0 0 1], [0 0 1]}), 1',
    'sortboundaries': "polyshape([0 1 1], [0 0 1]), 'area', 'ascend'",
    'subtract': 'polyshape([0 1 1], [0 0 1]), polyshape([0 1 1] + 0.5, [0 0 1])',
    'procrustes': "(1:10)', (1:10)'",
    'pspectrum': '1:5',
    'quad2d': '@(x,y) x.*y, 0, 1, 0, 1',
    'quadgk': '@(x) x.^2, 0, 1',
    'quiver': '[1 2 3], [4 5 6]',
    'quiver3': '1, 1, 1, 1, 1, 1',
    'rangesearch': '[0 1 2], [0 1 4], 0.5',
    'ranksum': '[1 2 3], [4 5 6]',
    'raylfit': '1:5',
    'readcell': "'m.csv'",
    'readmatrix': "'m.csv'",
    'readstruct': "'s.xml'",
    'readtable': "'m.csv'",
    'rectangle': '',
    'relieff': 'rand(20,3), [ones(10,1); 2*ones(10,1)], 3',
    'rescale': '1:5',
    'RespConfig': '',
    'rgb2lab': 'rand(20,3)',
    'rgb2xyz': 'rand(20,3)',
    'rmmissing': '1:5',
    'rowfun': '@(a,b) a+b, table([1;2],[3;4])',
    'rows2vars': "table((1:3)', (4:6)')",
    'scatterhistogram': '[1 2 3], [4 5 6]',
    'sequentialfs': '@(a,b,c,d) sum((d - c).^2), rand(20,3), rand(20,1)',
    'set': "gca, 'XLim', [0 1]",
    'sgtitle': '1:5',
    'signrank': '1:5',
    'signtest': '1:5',
    'simset': '',
    'spectralcluster': 'magic(4), 2',
    'ssim': '[1 2 3], [4 5 6]',
    'stack': 'table([1;2],[3;4]), [1 2]',
    'startsWith': "'abc', 'a'",
    'stepDataOptions': '',
    'stepinfo': '1:5',
    'stepwisefit': "(1:10)', (1:10)'",
    'stft': 'rand(256,1)',
    'streamline': "{'a','b'}",
    'struct2table': "struct('a',1)",
    'structfun': "@(x) x, struct('a', 1)",
    'subplot': '2, 1, 1',
    'suptitle': '1:5',
    'surf': '1:5',
    'surfc': 'magic(4)',
    'svds': '1:5',
    'synchronize': "timetable(seconds(1:3)', (1:3)'), timetable(seconds(1:3)', (4:6)')",
    'table2struct': "table((1:3)', (4:6)')",
    'table2timetable': "table(seconds(1:3)', (1:3)')",
    'tetramesh': '[1 2 3 4], [0 0 0; 1 0 0; 0 1 0; 0 0 1]',
    'timetable2table': "timetable(seconds(1:3)', (1:3)')",
    'triplot': '[1 2 3], [0 1 0], [0 0 1]',
    'trisurf': '[1 2 3], [0 1 0], [0 0 1], [0 0 0]',
    'tsne': "(1:5)'",
    'ttest': '1:5',
    'ttest2': '[1 2 3], [4 5 6]',
    'uialert': "uifigure, 'msg', 'title'",
    'uiaxes': '',
    'uibutton': '',
    'uibuttongroup': '',
    'uicheckbox': '',
    'uiconfirm': "uifigure, 'msg', 'title'",
    'uicontrol': '',
    'uidatepicker': '',
    'uidropdown': '',
    'uieditfield': '',
    'uifigure': '',
    'uigauge': '',
    'uigridlayout': '',
    'uihyperlink': '',
    'uiimage': '',
    'uiknob': '',
    'uilabel': '',
    'uilamp': '',
    'uilistbox': '',
    'uimenu': '',
    'uipanel': '',
    'uiprogressdlg': 'uifigure',
    'uiradiobutton': '',
    'uislider': '',
    'uispinner': '',
    'uiswitch': '',
    'uitab': '',
    'uitabgroup': '',
    'uitable': '',
    'uitextarea': '',
    'uitogglebutton': '',
    'unstack': "table([1;1;2;2],[1;2;1;2],[5;6;7;8]), 'Var3', 'Var2'",
    'varfun': '@sum, table([1;2],[3;4])',
    'vartest': 'magic(4), 2',
    'vartest2': '[1 2 3], [4 5 6]',
    'waitbar': '1:5',
    'wblfit': '1:5',
    'writelines': "'abc', 'w.txt'",
    'writestruct': "struct('a', 1), 's2.xml'",
    'xlabel': '1:5',
    'xyz2lab': 'rand(20,3)',
    'xyz2rgb': 'rand(20,3)',
    'ylabel': '1:5',
    'zlabel': '1:5',
    'ztest': 'randn(20,1), 0, 1',
}

LOUD = {
    'addvars': "table((1:3)', (4:6)')",
    'array2timetable': "(1:3)', 'RowTimes', seconds(1:3)'",
    'arrayfun': '@(x) (x-1).^2, 0',
    'blockproc': 'magic(4), [2 2], @(b) b.data',
    'boxplot': '1:5',
    'cellfun': '@numel, {1, [1 2]}',
    'classify': '[0 1 2], [0 1 4], 0.5',
    'cluster': '[1 2 3], [4 5 6]',
    'confusionchart': 'magic(4)',
    'corner': '1:5',
    'count': "'abc', 'b'",
    'datasample': 'magic(4), 2',
    'datetime': '',
    'dlmwrite': "'x', 1",
    'donutchart': '1:5',
    'duration': '',
    'eraseBetween': "'abcde', 2, 3",
    'exportgraphics': "gca, 'out.png'",
    'extractBetween': "'abcde', 2, 3",
    'ezcontour': '@(x,y) x.*y',
    'ezcontourf': '@(x,y) x.*y',
    'ezmesh': '@(x,y) x.*y',
    'ezplot': "'abc'",
    'ezsurf': '@(x,y) x.*y',
    'ezsurfc': '@(x,y) x.*y',
    'fcontour': '@(x,y) x.*y',
    'fimplicit': '@(x,y) x.^2+y.^2-1',
    'fimplicit3': '@(x,y,z) x.^2+y.^2+z.^2-1',
    'find_system': '',
    'fitglm': "(1:10)', (1:10)'",
    'fitlm': "(1:10)', (1:10)'",
    'fmesh': '@(x,y) x.*y',
    'fplot3': '@sin, @cos, @(t) t',
    'fsurf': '@(x,y) x.*y',
    'geobubble': '[1 2 3], [4 5 6]',
    'glmfit': "(1:10)', (1:10)'",
    'grpstats': '1:5',
    'heatmap': '1:5',
    'imbinarize': '1:5',
    'imfuse': '[1 2 3], [4 5 6]',
    'imgaussfilt': '1:5',
    'imresize': '[1 2; 3 4], [1 2]',
    'imresize3': 'rand(4,4,4), 2',
    'imshowpair': '[1 2 3], [4 5 6]',
    'imtransform': "magic(4), maketform('affine', eye(3))",
    'imtranslate': '[1 2 3], [4 5 6]',
    'imwarp': 'magic(4), affine2d(eye(3))',
    'lassoglm': "(1:10)', (1:10)'",
    'linkage': "(1:5)'",
    'load': "'v.mat'",
    'matches': "'abc', 'abc'",
    'matlab.unittest.constraints.Matches': "'abc'",
    'matlab.unittest.constraints.Throws': '1:5',
    'parallelplot': '1:5',
    'pcolor': 'magic(4)',
    'piechart': '1:5',
    'pivot': "table([1;1;2],[1;2;3]), 'Rows', 'Var1'",
    'replaceBetween': "'abcde', 2, 3, 'x'",
    'retime': "timetable(seconds(1:3)', (1:3)'), 'regular', 'linear', 'TimeStep', seconds(0.5)",
    'runstest': '1:5',
    'save': '',
    'splitvars': "table((1:3)', (4:6)')",
    'stackedplot': "(1:5)'",
    'stepwiselm': "(1:10)', (1:10)'",
    'subtitle': '1:5',
    'tiledlayout': '',
    'timetable': '',
    'title': '1:5',
    'vartestn': 'magic(4)',
    'violinplot': '1:5',
    'wordcloud': '1:5',
    'writecell': "{'a','b'}",
    'writematrix': '1:5',
    'writetable': "table((1:3)', (4:6)')",
}

KNOWN_IGNORED = {
    'set_param': "'optsmdl', 'StopTime', '5'",
    'table': '',
}

_TEXT_SLOT = "the unknown name lands in a positional slot that takes text ({}) and is rejected there"
_TODO = "not fixed yet: {}"
LOUD_REASONS = {
    **{n: "owned by the parallel merge of the text and table functions"
       for n in ("addvars",)},
    **{n: _TODO.format("the unknown name is counted as an extra positional argument")
       for n in ("array2timetable", "blockproc", "count", "datasample", "donutchart", "eraseBetween",
                 "extractBetween", "imfuse", "imresize3", "imshowpair", "imtransform", "matches", "piechart",
                 "replaceBetween", "splitvars", "violinplot", "fplot3")},
    **{n: _TODO.format("the plotting helper takes the unknown name for the plotting interval")
       for n in ("ezcontour", "ezcontourf", "ezmesh", "ezplot", "ezsurf", "ezsurfc", "fcontour", "fimplicit",
                 "fimplicit3", "fmesh", "fsurf")},
    **{n: _TODO.format("the call's own name-value parser rejects the pair without naming it")
       for n in ("cluster", "exportgraphics", "find_system", "geobubble", "heatmap", "imgaussfilt", "imresize",
                 "pcolor", "subtitle", "title", "parallelplot", "stackedplot", "wordcloud")},
    "arrayfun": "a text argument is a data input (a char array), as in MATLAB",
    "cellfun": "a text argument is a data input, as in MATLAB",
    "classify": _TEXT_SLOT.format("the discriminant type"),
    "corner": _TEXT_SLOT.format("the method"),
    "imbinarize": _TEXT_SLOT.format("the method"),
    "imtranslate": _TEXT_SLOT.format("the interpolation method"),
    "imwarp": _TEXT_SLOT.format("the interpolation method"),
    "linkage": _TEXT_SLOT.format("the linkage method"),
    "tiledlayout": _TEXT_SLOT.format("the arrangement"),
    "datetime": _TEXT_SLOT.format("the date text"),
    "duration": _TEXT_SLOT.format("the duration text"),
    "timetable": _TEXT_SLOT.format("a variable"),
    "fitlm": _TEXT_SLOT.format("the model specification"),
    "fitglm": _TEXT_SLOT.format("the model specification"),
    "stepwiselm": _TEXT_SLOT.format("the model specification"),
    "glmfit": _TEXT_SLOT.format("the distribution name"),
    "lassoglm": _TEXT_SLOT.format("the distribution name"),
    "runstest": _TEXT_SLOT.format("the reference value, which may be 'ud'"),
    "confusionchart": _TEXT_SLOT.format("the predicted labels"),
    "grpstats": _TEXT_SLOT.format("the grouping variable"),
    "vartestn": _TEXT_SLOT.format("the grouping variable"),
    "boxplot": _TEXT_SLOT.format("the grouping variable"),
    "load": _TEXT_SLOT.format("the variable names; load has no name-value options"),
    "save": _TEXT_SLOT.format("the file name; save has no name-value options"),
    "dlmwrite": _TEXT_SLOT.format("the delimiter"),
    "writecell": _TEXT_SLOT.format("the file name, which this minimal call leaves out"),
    "writematrix": _TEXT_SLOT.format("the file name, which this minimal call leaves out"),
    "writetable": _TEXT_SLOT.format("the file name, which this minimal call leaves out"),
    "retime": _TEXT_SLOT.format("the time-step name"),
    "pivot": _TEXT_SLOT.format("a variable name"),
    "matlab.unittest.constraints.Matches": _TEXT_SLOT.format("the expression"),
    "matlab.unittest.constraints.Throws": _TEXT_SLOT.format("the error identifier"),
}

NOCALL = ['listdlg', 'readlines', 'textread', 'uigetfile', 'uiputfile']

DUPS = ['boxchart',
 'clock',
 'dbstop',
 'discretize',
 'filter2',
 'fminbnd',
 'fmincon',
 'fminsearch',
 'fminunc',
 'fsolve',
 'fzero',
 'getfield',
 'groupcounts',
 'gscatter',
 'histcounts2',
 'intlinprog',
 'isoutlier',
 'keyboard',
 'lasterr',
 'linkprop',
 'linprog',
 'lsqcurvefit',
 'lsqnonlin',
 'lsqnonneg',
 'mislocked',
 'mlock',
 'munlock',
 'mustBeBetween',
 'mustBeColumn',
 'mustBeFloat',
 'mustBeMatrix',
 'mustBeNonmissing',
 'mustBeNonsparse',
 'mustBeRow',
 'mustBeScalarOrEmpty',
 'mustBeSparse',
 'mustBeUnderlyingType',
 'mustBeValidVariableName',
 'namelengthmax',
 'now',
 'num2cell',
 'optimoptions',
 'optimset',
 'pause',
 'quadprog',
 'rats',
 'setfield',
 'standardizeMissing',
 'strfind',
 'swapbytes',
 'uistack',
 'uniquetol']

CALLS["readlines"] = "'m.csv'"
CALLS["textread"] = "'m.csv', '%s'"
CALLS.update({  # these need the model 'optsmdl' made by the session fixture, or check options before any I/O
    "add_block": "'simulink/Sources/Constant', 'optsmdl/C9'",
    "add_line": "'optsmdl', 'C1/1', 'G1/1'",
    "sim": "'optsmdl'",
    "bubblelegend": "ancestor(bubblechart(1:3, 1:3, 1:3), 'axes')",
    "gmdistribution.fit": "[randn(30,1); randn(30,1)+5], 2",
    "h5create": "'f1.h5', '/d', [2 2]",
    "h5writeatt": "'f2.h5', '/', 'a', 1",
    "nccreate": "'f1.nc', 'v'",
    "multcompare": "nthargout(3, @anova1, [1 2 3 4 5 6]', [1 1 2 2 3 3]', 'off')",
    "nlparci": "[1 1], [0.1; -0.1; 0.2; -0.2], 'jacobian', [1 1; 2 1; 3 1; 4 1]",
    "nlpredci": "@(b,x) b(1)*x+b(2), (1:4)', [1 1], [0.1; -0.1; 0.2; -0.2], 'jacobian', [1 1; 2 1; 3 1; 4 1]",
    "odextend": "ode45(@(t,y) -y, [0 1], 1), [], 2",
    "sampsizepwr": "'z', [0 1], 0.5, 0.9",
    "urlread": "'http://127.0.0.1:9/x'",
    "urlwrite": "'http://127.0.0.1:9/x', 'u.txt'",
})

_NOCALL_REASON = "no minimal call written yet (needs a model, a block diagram, a network or a dialog)"
EXEMPT = {name: "being merged" for name in DUPS}
EXEMPT.update({name: _NOCALL_REASON for name in NOCALL if name not in CALLS})
EXEMPT.update({
    "set": "sets graphics properties by name; an unknown property is the object's own check",
    "table": "a text argument that is not an option name is a table variable, as in MATLAB",
    "uigetfile": "opens a dialog",
    "uiputfile": "opens a dialog",
    "listdlg": "opens a dialog",
    "webread": "name-value pairs after the URL are query parameters, as in MATLAB",
    "websave": "name-value pairs after the URL are query parameters, as in MATLAB",
})
for _name in ("set", "table"):
    CALLS.pop(_name, None)
    LOUD.pop(_name, None)
    KNOWN_IGNORED.pop(_name, None)


def _codes(co):
    yield co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from _codes(c)


def _helpers() -> dict:
    out = {}
    for mod, names in OPTION_HELPERS.items():
        m = sys.modules["mylab.builtins." + mod]
        for n in names:
            out[id(getattr(m, n))] = f"{mod}.{n}"
    return out


def _uses_helper(fn, helpers: dict, depth: int = 3, seen: set | None = None) -> bool:
    """True when ``fn`` (or a private function it calls, a few levels deep) calls an option helper."""
    seen = set() if seen is None else seen
    fn = inspect.unwrap(fn)
    if id(fn) in seen or not hasattr(fn, "__code__"):
        return False
    seen.add(id(fn))
    names = set()
    for co in _codes(fn.__code__):
        names |= set(co.co_names)
    cands = []
    for n in names:
        v = fn.__globals__.get(n)
        if isinstance(v, types.ModuleType) and v.__name__.startswith("mylab.builtins"):
            cands += [getattr(v, n2) for n2 in names if callable(getattr(v, n2, None))]
        elif v is not None:
            cands.append(v)
    for cell in fn.__closure__ or ():
        try:
            cands.append(cell.cell_contents)
        except ValueError:
            pass
    for v in cands:
        if id(v) in helpers:
            return True
        if depth > 0 and isinstance(v, types.FunctionType) and v.__module__.startswith("mylab.builtins") \
                and not getattr(v, "matlab_names", None) and _uses_helper(v, helpers, depth - 1, seen):
            return True
    return False


def _option_builtins() -> list[str]:
    Interpreter()
    helpers = _helpers()
    return sorted(name for name, fn in BUILTINS.items() if _uses_helper(fn, helpers))


def test_every_option_parser_is_listed():
    listed = set(CALLS) | set(LOUD) | set(KNOWN_IGNORED) | set(EXEMPT)
    missing = [n for n in _option_builtins() if n not in listed]
    assert not missing, "add these built-ins to a table in tests/test_options_unknown.py: " + ", ".join(missing)


def test_loud_entries_have_reasons():
    assert sorted(LOUD) == sorted(LOUD_REASONS)


def test_tables_do_not_overlap():
    groups = [set(CALLS), set(LOUD), set(KNOWN_IGNORED), set(EXEMPT)]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b), sorted(a & b)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    old = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("opts"))
    interp = Interpreter(out=lambda s: None)
    interp.execute("writematrix([0 1 2; 3 4 5], 'm.csv'); a = 1; save('v.mat', 'a'); "
                   "imwrite(uint8(magic(4)), 'img.png'); writestruct(struct('a', 1), 's.xml'); "
                   "new_system('optsmdl'); add_block('simulink/Sources/Constant', 'optsmdl/C1'); "
                   "add_block('simulink/Math Operations/Gain', 'optsmdl/G1');")
    yield interp
    interp.execute("close all force; close_system('optsmdl', 0);")
    os.chdir(old)


def _call(name: str, call: str) -> str:
    return f"{name}({call + ', ' if call else ''}'NoSuchOption__', 1);"


def _run(interp, code: str):
    try:
        interp.execute(code)
    finally:
        try:
            interp.execute("close all force;")
        except MatlabError:
            pass


def test_helper_rejects_bad_input():
    from mylab.builtins._util import parse_options, split_options
    from mylab.values import char
    assert parse_options([char("maxiter"), 3], ["MaxIter", "Display"], "f") == {"MaxIter": 3}
    assert list(parse_options([char("Disp"), 5], ["MaxIter", "Display"], "f")) == ["Display"]
    with pytest.raises(MatlabError, match="'Bogus' is not supported"):
        parse_options([char("Bogus"), 1], ["MaxIter"], "f")
    with pytest.raises(MatlabError, match="needs a value"):
        parse_options([char("MaxIter")], ["MaxIter"], "f")
    with pytest.raises(MatlabError, match="expected an option name"):
        parse_options([1, 2], ["MaxIter"], "f")
    with pytest.raises(MatlabError, match="ambiguous"):
        parse_options([char("Max"), 2], ["MaxIter", "MaxFunEvals"], "f")
    pos, opts = split_options([1, char("omitnan"), char("MaxIter"), 2], ["MaxIter"], "f", flags=("omitnan",))
    assert len(pos) == 2 and opts == {"MaxIter": 2}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_unknown_option_is_named(session, name):
    with pytest.raises(MatlabError) as e:
        _run(session, _call(name, CALLS[name]))
    assert "NoSuchOption__" in str(e.value)


@pytest.mark.parametrize("name", sorted(LOUD))
def test_unknown_option_is_rejected(session, name):
    _run(session, f"{name}({LOUD[name]});")  # the call itself is valid
    with pytest.raises(MatlabError):
        _run(session, _call(name, LOUD[name]))


@pytest.mark.parametrize("name", sorted(KNOWN_IGNORED))
@pytest.mark.xfail(strict=True, reason="known bug: the unknown option is still ignored")
def test_known_ignored(session, name):
    with pytest.raises(MatlabError):
        _run(session, _call(name, KNOWN_IGNORED[name]))
