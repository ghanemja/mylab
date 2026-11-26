"""Help text for the data-file functions of mylab/builtins/fileformats.py: MAT-files, import
options and datastores, memory maps, XML, struct files, NetCDF, HDF5 and video.

Written from scratch for mylab.
"""

_H5 = (" HDF5 files are read and written with the Python package h5py, which is installed with mylab; the "
       "browser app loads it the first time a program uses HDF5.")
_H5_ORDER = (" Sizes and positions are given the MATLAB way, first dimension first; in the file the "
             "dimensions are stored in the opposite order, as MATLAB also does.")
_NC = (" Classic files ('classic' and '64bit') are read and written directly; NetCDF-4 files ('netcdf4' and "
       "'netcdf4_classic', which are HDF5 files) through the Python package h5py, which is installed with "
       "mylab. The format of a file is detected when it is read. In a NetCDF-4 file a variable or group is "
       "named by its path, such as '/grp/var'.")

HELP = {
    # -- MAT-files -----------------------------------------------------------------------------------
    "matfile": (["m = matfile(filename)", "m = matfile(filename, 'Writable', true)", "x = m.name",
                 "part = m.name(rows, cols)", "m.name = value", "who(m)", "whos(m)", "sz = size(m, 'name')"],
                "Gives access to the variables of a MAT-file as if they were properties of m. Reading "
                "m.name loads that variable only; indexing such as m.x(2, :) returns part of it. "
                "Assigning m.name = value, or to part of a variable, writes the file at once (the file "
                "must have been opened with 'Writable', true, which is the default only for a file that "
                "does not exist yet). who(m), whos(m) and size(m, 'name') describe what is stored, and "
                "m.Properties holds Source and Writable. In a version 7.3 (HDF5) file, indexing a numeric, "
                "char or logical variable reads only the selected elements, and assigning to part of one "
                "writes only that part, as long as the variable keeps its size and class; size(m, 'name') "
                "and whos(m) read no data. Indices can be scalars, ranges such as a:b and a:step:b, colons "
                "and 'end'. Other variables, and every variable of an older version of file, are read as a "
                "whole. All MAT-file versions that load reads are accepted; like MATLAB, matfile creates a "
                "version 7.3 file when the file does not exist yet."),
    # -- import options and datastores -----------------------------------------------------------------
    "detectImportOptions": (
        ["opts = detectImportOptions(filename)", "opts = detectImportOptions(filename, Name, Value)",
         "T = readtable(filename, opts)", "opts = setvartype(opts, names, type)",
         "opts = setvaropts(opts, names, Name, Value)", "v = getvaropts(opts, names)",
         "T = preview(filename, opts)"],
        "Looks at a delimited text file and returns an options object that says how readtable, readmatrix "
        "or readcell will read it: Delimiter, VariableNames, VariableTypes ('double', 'char', 'string', "
        "'datetime', 'duration', 'categorical', 'logical' and the integer and single classes), "
        "SelectedVariableNames, DataLines, VariableNamesLine, RowNamesColumn and the rules MissingRule, "
        "ImportErrorRule ('fill', 'error', 'omitrow', 'omitvar'), EmptyLineRule, ExtraColumnsRule, "
        "ConsecutiveDelimitersRule and LeadingDelimitersRule. Change a property with opts.Name = value. "
        "setvartype changes the type of some or all variables; setvaropts sets per-variable options "
        "(FillValue, TreatAsMissing, DecimalSeparator, ThousandsSeparator, TrimNonNumeric, InputFormat, "
        "DatetimeFormat, TimeZone, Categories, Ordinal, TrueSymbols, FalseSymbols, Prefixes, Suffixes); "
        "getvaropts returns them as a struct; preview reads the first eight rows. The name-value "
        "arguments are those of readtable, for example 'Delimiter', 'NumHeaderLines', "
        "'DecimalSeparator', 'TextType' and 'VariableNamingRule'. Spreadsheet files are not supported "
        "here: pass their Sheet and Range to readtable directly."),
    "delimitedTextImportOptions": (
        ["opts = delimitedTextImportOptions", "opts = delimitedTextImportOptions('NumVariables', n)",
         "opts = delimitedTextImportOptions(Name, Value)"],
        "Creates import options for delimited text by hand instead of detecting them from a file. "
        "Give 'NumVariables' or 'VariableNames' (and 'VariableTypes', which default to 'char'), then "
        "'Delimiter', 'DataLines', 'VariableNamesLine' and the other properties listed under "
        "detectImportOptions. Use the result with readtable(filename, opts)."),
    "tabularTextDatastore": (
        ["ds = tabularTextDatastore(location)", "ds = tabularTextDatastore(location, Name, Value)",
         "T = read(ds)", "[T, info] = read(ds)", "T = readall(ds)", "tf = hasdata(ds)", "reset(ds)",
         "T = preview(ds)"],
        "A collection of delimited text files that have the same columns, read a block of rows at a "
        "time. location is a file, a folder, a pattern with * or a cell array of these. read returns "
        "the next ReadSize rows (20000 by default, or 'file' for one file at a time) as a table, "
        "hasdata tells whether rows are left, reset starts again, readall returns everything and "
        "preview the first eight rows. Properties: Files, ReadSize, VariableNames, "
        "SelectedVariableNames, SelectedFormats, Delimiter, NumHeaderLines, ReadVariableNames, TextType "
        "and DatetimeType. Options: those of readtable plus 'FileExtensions', 'IncludeSubfolders', "
        "'ReadSize', 'SelectedVariableNames' and 'SelectedFormats'. Also available: partition(ds, n, k), "
        "subset(ds, indices), shuffle(ds), numpartitions(ds), progress(ds), transform(ds, @fcn) and "
        "combine(ds1, ds2)."),
    "datastore": (
        ["ds = datastore(location)", "ds = datastore(location, 'Type', 'tabulartext', Name, Value)",
         "ds = datastore(location, 'Type', 'image', Name, Value)"],
        "Creates a datastore for a collection of files. The default kind is the tabular text datastore "
        "(see tabularTextDatastore); 'Type', 'image' makes an image datastore (see imageDatastore) and "
        "takes its options. Another 'Type' is an error."),
    "memmapfile": (
        ["m = memmapfile(filename)", "m = memmapfile(filename, 'Format', type)",
         "m = memmapfile(filename, 'Format', {type, size, fieldname; ...})",
         "m = memmapfile(..., 'Offset', bytes, 'Repeat', n, 'Writable', true)", "x = m.Data",
         "m.Data(k) = value"],
        "Maps a file into memory, so that its bytes are read and changed like an array. With a type name "
        "('uint8' by default, any integer class, 'single' or 'double') m.Data is a column vector of the "
        "whole file. With a cell array, each row {type, [rows cols], name} describes one field of a "
        "record and m.Data is a struct array with one element per record. Offset skips bytes at the "
        "start, Repeat limits the number of elements or records (Inf: as many as fit), and Writable "
        "true lets assignments to m.Data, or to parts of it, change the file. The properties Filename, "
        "Format, Offset, Repeat and Writable can be changed later. Values are stored in the byte order "
        "of the machine."),
    # -- XML -------------------------------------------------------------------------------------------
    "xmlread": (
        ["doc = xmlread(filename)", "root = doc.getDocumentElement()", "list = doc.getElementsByTagName(tag)",
         "n = list.getLength()", "node = list.item(k)", "value = char(node.getAttribute(name))",
         "text = char(node.getTextContent())"],
        "Reads an XML file and returns its document node. Nodes answer to the usual DOM method names: "
        "getNodeName, getNodeValue, getNodeType, getParentNode, getChildNodes, getFirstChild, "
        "getLastChild, getNextSibling, getPreviousSibling, hasChildNodes, getAttributes, hasAttributes, "
        "getTextContent, setTextContent, appendChild, removeChild, insertBefore, replaceChild and "
        "cloneNode; elements add getTagName, getAttribute, setAttribute, hasAttribute, removeAttribute "
        "and getElementsByTagName; the document adds getDocumentElement, createElement, createTextNode, "
        "createComment and createCDATASection; text nodes add getData, setData and getLength. Node "
        "lists and attribute lists have getLength and item(k), where k counts from 0 and an index past "
        "the end gives []. A method without arguments may be written without parentheses. Text comes "
        "back as character vectors (in MATLAB as Java strings, so char(...) around it works in both), "
        "counts as numbers and tests as logicals. As in MATLAB, white space between elements is kept "
        "as text nodes. The constants ELEMENT_NODE, TEXT_NODE and so on are properties of every node."),
    "xmlwrite": (
        ["xmlwrite(filename, node)", "text = xmlwrite(node)"],
        "Writes a document or element node (from xmlread or built with createElement) as indented XML "
        "text, to a file or, with one argument, as the returned character vector."),
    "readstruct": (
        ["S = readstruct(filename)", "S = readstruct(filename, 'FileType', 'xml')",
         "S = readstruct(filename, 'StructNodeName', name, 'AttributeSuffix', suffix, 'ImportAttributes', tf)"],
        "Reads an XML or JSON file into a structure. In XML, child elements become fields, repeated "
        "elements become arrays (struct arrays, numeric rows or string arrays), attributes become fields "
        "whose name ends in 'Attribute' (or the given suffix), and the text of an element that also has "
        "attributes or children is its field Text. Text that is a number or true/false is converted; "
        "other text is returned as strings. A JSON file must hold an object; its text also becomes "
        "strings. The file type follows the extension unless 'FileType' is given."),
    "writestruct": (
        ["writestruct(S, filename)", "writestruct(S, filename, 'FileType', 'json')",
         "writestruct(S, filename, 'StructNodeName', name, 'AttributeSuffix', suffix, 'PrettyPrint', tf)"],
        "Writes a structure to an XML or JSON file, the reverse of readstruct. In XML the root element "
        "is named 'struct' unless StructNodeName says otherwise, each field becomes an element, the "
        "elements of an array become repeated elements, fields ending in 'Attribute' become "
        "attributes, and nested structures become nested elements."),
    # -- NetCDF ----------------------------------------------------------------------------------------
    "nccreate": (
        ["nccreate(filename, varname)", "nccreate(filename, varname, 'Dimensions', {name1, len1, name2, len2})",
         "nccreate(..., 'Datatype', type, 'Format', fmt, 'FillValue', value)",
         "nccreate(..., 'ChunkSize', chunk, 'DeflateLevel', level, 'Shuffle', tf)"],
        "Creates a variable in a NetCDF file, and the file, the groups on its path and the named "
        "dimensions if they do not exist yet. A dimension that exists already is given by its name alone; "
        "a length of Inf makes the dimension unlimited. Without Dimensions the variable holds one value. "
        "Format applies when the file is created: 'classic' (the default), '64bit', 'netcdf4' or "
        "'netcdf4_classic'. Datatype is 'double' (the default), 'single', 'int32', 'int16', 'int8' or "
        "'char'; NetCDF-4 files also take 'uint8', 'uint16', 'uint32', 'int64', 'uint64' and, in the "
        "'netcdf4' format, 'string'. Classic and netcdf4_classic files allow one unlimited dimension, "
        "which must be the variable's last; 'netcdf4' files allow several, anywhere, and groups. "
        "ChunkSize, DeflateLevel (0 to 9) and Shuffle set the storage of NetCDF-4 variables. A new "
        "variable is filled with the fill value ('disable': no filling), which ncread returns as NaN for "
        "double and single variables." + _NC),
    "ncwrite": (
        ["ncwrite(filename, varname, data)", "ncwrite(filename, varname, data, start)",
         "ncwrite(filename, varname, data, start, stride)"],
        "Writes data to a variable of a NetCDF file, starting at index 1 of each dimension or at start, "
        "taking every stride-th position. Writing past the end of an unlimited dimension makes it longer. "
        "The data is converted to the variable's type; if the variable has scale_factor or add_offset "
        "attributes they are applied in reverse first, and NaN is stored as the fill value." + _NC),
    "ncread": (
        ["data = ncread(filename, varname)", "data = ncread(filename, varname, start, count)",
         "data = ncread(filename, varname, start, count, stride)"],
        "Reads a variable of a NetCDF file, or the part that starts at start with count elements per "
        "dimension (Inf: to the end) taking every stride-th. The result has the variable's class; a "
        "variable with one dimension comes back as a column. For double and single variables the fill "
        "value becomes NaN, and scale_factor and add_offset attributes are applied (giving double)."
        + _NC),
    "ncinfo": (
        ["info = ncinfo(filename)", "vinfo = ncinfo(filename, varname)"],
        "Describes a NetCDF file as a struct with Filename, Name, Dimensions (Name, Length, Unlimited), "
        "Variables (Name, Dimensions, Size, Datatype, Attributes, ChunkSize, FillValue, DeflateLevel, "
        "Shuffle), Attributes (Name, Value), Groups (the same fields for each group of a NetCDF-4 file) "
        "and Format, or one group or variable of it." + _NC),
    "ncdisp": (
        ["ncdisp(filename)", "ncdisp(filename, varname)"],
        "Prints the dimensions, variables, attributes and groups of a NetCDF file, or of one group or "
        "variable." + _NC),
    "ncreadatt": (
        ["value = ncreadatt(filename, location, attname)"],
        "Reads an attribute of a NetCDF file. location is '/' for the file's own attributes, the path of "
        "a group, or the name of a variable. Text comes back as a character vector, NC_STRING values "
        "as strings and numbers as a row vector of their class." + _NC),
    "ncwriteatt": (
        ["ncwriteatt(filename, location, attname, value)"],
        "Creates or replaces an attribute of a NetCDF file ('/'), of a group or of a variable. The value "
        "is text or a numeric vector of class double, single, int32, int16 or int8; a 'netcdf4' file also "
        "takes the other integer classes and string arrays." + _NC),
    # -- HDF5 ------------------------------------------------------------------------------------------
    "h5create": (
        ["h5create(filename, dataset, size)", "h5create(filename, dataset, size, 'Datatype', type)",
         "h5create(..., 'ChunkSize', chunk, 'Deflate', level, 'Shuffle', tf, 'Fletcher32', tf, 'FillValue', v)"],
        "Creates a dataset in an HDF5 file (and the file and the groups on the way, if needed). dataset "
        "is a full name such as '/group/data'. size may contain Inf for a dimension that can grow, "
        "which needs a ChunkSize; compression with Deflate (0 to 9) and the Shuffle and Fletcher32 "
        "filters need one too. Datatype is 'double' (the default), 'single', an integer class or "
        "'string'." + _H5_ORDER + _H5),
    "h5write": (
        ["h5write(filename, dataset, data)", "h5write(filename, dataset, data, start, count)",
         "h5write(filename, dataset, data, start, count, stride)"],
        "Writes data to an existing dataset of an HDF5 file: all of it, or the block that starts at "
        "start and has count elements per dimension. Writing beyond the current size of a dimension "
        "created with Inf makes the dataset larger." + _H5_ORDER + _H5),
    "h5read": (
        ["data = h5read(filename, dataset)", "data = h5read(filename, dataset, start, count)",
         "data = h5read(filename, dataset, start, count, stride)"],
        "Reads a dataset of an HDF5 file, or part of it. Numbers keep their class, strings come back as "
        "a string array, and a compound dataset as a struct with one field per member. A dataset with "
        "one dimension is returned as a column." + _H5_ORDER + _H5),
    "h5info": (
        ["info = h5info(filename)", "info = h5info(filename, location)"],
        "Describes an HDF5 file, group or dataset. A group gives Filename, Name, Groups, Datasets and "
        "Attributes; a dataset gives Name, Datatype (Class, Type, Size), Dataspace (Size, MaxSize), "
        "ChunkSize, FillValue, Filters and Attributes." + _H5),
    "h5disp": (
        ["h5disp(filename)", "h5disp(filename, location)"],
        "Prints the groups, datasets and attributes of an HDF5 file, or of one group or dataset." + _H5),
    "h5readatt": (
        ["value = h5readatt(filename, location, attname)"],
        "Reads an attribute of a group or dataset of an HDF5 file ('/' is the file itself). Text is "
        "returned as a character vector." + _H5),
    "h5writeatt": (
        ["h5writeatt(filename, location, attname, value)"],
        "Creates or replaces an attribute of an existing group or dataset of an HDF5 file. The value is "
        "numeric or text." + _H5),
    # -- video -----------------------------------------------------------------------------------------
    "VideoReader": (
        ["v = VideoReader(filename)", "v = VideoReader(filename, 'CurrentTime', t)", "frame = readFrame(v)",
         "tf = hasFrame(v)", "frames = read(v)", "frame = read(v, k)", "frames = read(v, [first last])"],
        "Opens a video file for reading. readFrame returns the next frame as a height-by-width-by-3 "
        "uint8 array (height-by-width for grayscale video) and hasFrame tells whether one is left; "
        "read returns frame k, a range of frames (last may be Inf) or all of them, stacked along the "
        "fourth dimension. Properties: Name, Path, Duration, CurrentTime (can be set, to jump), "
        "NumFrames, FrameRate, Height, Width, BitsPerPixel, VideoFormat, Tag and UserData. AVI files "
        "with Motion JPEG or uncompressed frames, which is what VideoWriter writes, are decoded by "
        "mylab itself; other formats (MP4 and so on) are decoded with the ffmpeg program when it is "
        "installed, otherwise an error says so."),
    "readFrame": ("frame = readFrame(v)",
                  "The next frame of a VideoReader object as a uint8 image; see VideoReader. It is an error "
                  "when no frame is left, so test hasFrame(v) first."),
    "hasFrame": ("tf = hasFrame(v)",
                 "True while a VideoReader object has frames that readFrame has not returned yet."),
    "mmfileinfo": ("info = mmfileinfo(filename)",
                   "Describes a video file as a struct with Filename, Path, Duration, Video (Format, Height, "
                   "Width) and Audio. The audio part is always empty: mylab does not read the sound track "
                   "of a video file."),
}
