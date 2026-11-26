"""Help text for imageDatastore and the methods shared by mylab's datastores.

Written from scratch for mylab.
"""

_DS = (" Works on image datastores, tabular text datastores and the datastores that transform and combine "
       "return. Also callable as ds.{name}(...).")

HELP = {
    "imageDatastore": (
        ["ds = imageDatastore(location)", "ds = imageDatastore(location, 'IncludeSubfolders', true)",
         "ds = imageDatastore(location, 'LabelSource', 'foldernames')",
         "ds = imageDatastore(location, 'Labels', labels)",
         "ds = imageDatastore(location, 'FileExtensions', {'.png', '.jpg'})",
         "ds = imageDatastore(location, 'ReadFcn', @fcn, 'ReadSize', n)"],
        "A list of image files that is read one image at a time. location is a folder, a file, a pattern "
        "with * or a cell or string array of these; 'IncludeSubfolders' also looks inside subfolders. Only "
        "files whose extension is in 'FileExtensions' are kept; by default those are the image formats "
        "imread reads. With 'LabelSource', 'foldernames' each file is labelled with the name of the folder "
        "that holds it; 'Labels' gives one label per file instead. Properties: Files (a column cell array "
        "of full paths), Folders, Labels (categorical for folder names; a column vector), ReadSize, "
        "ReadFcn (imread unless you give a function that takes a file name), AlternateFileSystemRoots, "
        "SupportedOutputFormats and DefaultOutputFormat. Labels, ReadSize, ReadFcn and Files can be "
        "assigned. The datastore is a handle: read moves a position shared by every copy of the variable. "
        "Methods: read, readimage, readall, hasdata, reset, preview, progress, countEachLabel, "
        "splitEachLabel, shuffle, subset, partition, numpartitions, transform, combine and copy. "
        "datastore(location, 'Type', 'image', ...) makes the same object."),
    "readimage": (
        ["img = readimage(ds, k)", "[img, info] = readimage(ds, k)"],
        "Reads the k-th file of an image datastore with its ReadFcn, without moving the read position. "
        "info is a struct with Filename, FileSize (bytes) and Label (empty when the datastore has no "
        "labels). Also callable as ds.readimage(k)."),
    "countEachLabel": (
        "T = countEachLabel(ds)",
        "Table with one row per label of an image datastore: the variable Label (categorical) and the "
        "variable Count, the number of files that have that label. For categorical labels every category "
        "is listed, in category order. Also callable as ds.countEachLabel()."),
    "splitEachLabel": (
        ["[ds1, ds2] = splitEachLabel(ds, p)", "[ds1, ..., dsN] = splitEachLabel(ds, p1, ..., pN-1)",
         "[ds1, ds2] = splitEachLabel(ds, n)", "[...] = splitEachLabel(..., 'randomized')",
         "[...] = splitEachLabel(..., 'Include', labels, 'Exclude', labels)"],
        "Splits the files of a labelled image datastore label by label. Proportions (each below 1, adding "
        "up to at most 1) give the share of each label's files that goes into each output, rounded to the "
        "nearest whole file; whole numbers give a count of files per label instead. The last output gets "
        "the files that are left. Without 'randomized' the first files of each label are taken and the "
        "outputs keep the original order; 'randomized' picks the files at random with the global random "
        "number generator (rng). 'Include' keeps only the listed labels and 'Exclude' leaves out the "
        "listed ones. Also callable as ds.splitEachLabel(...)."),
    "read": (
        ["data = read(ds)", "[data, info] = read(ds)"],
        "Reads the next block of a datastore and moves its read position on: one image (or a column cell "
        "array of ReadSize images) for an image datastore, the next rows as a table for a tabular text "
        "datastore. info describes what was read (Filename, FileSize and, for images, Label). Reading past "
        "the end is an error; check hasdata first or call reset." + _DS.format(name="read")),
    "readall": (
        ["data = readall(ds)", "data = readall(ds, 'UseParallel', false)"],
        "Reads all the data of a datastore without moving its read position: a column cell array of "
        "images for an image datastore, one table for a tabular text datastore. For transformed and "
        "combined datastores, tables and cell arrays from each read are stacked vertically and other "
        "values are collected in a column cell array." + _DS.format(name="readall")),
    "hasdata": (
        "tf = hasdata(ds)",
        "True while a datastore has data left to read." + _DS.format(name="hasdata")),
    "reset": (
        "reset(ds)",
        "Moves the read position of a datastore back to the start." + _DS.format(name="reset")),
    "preview": (
        ["data = preview(ds)", "T = preview(filename, opts)"],
        "The first block of a datastore (the first image of an image datastore, the first eight rows of "
        "a tabular text datastore), whatever the read position. With a file name and import options it "
        "reads the first eight rows of the file." + _DS.format(name="preview")),
    "progress": (
        "p = progress(ds)",
        "The fraction of a datastore that has been read, from 0 to 1." + _DS.format(name="progress")),
    "shuffle": (
        "ds2 = shuffle(ds)",
        "A copy of a datastore with its files in random order (the global random number generator, rng, "
        "decides it), starting from the beginning. A combined datastore can be shuffled when it is made of "
        "image datastores with the same number of files; they keep their files paired."
        + _DS.format(name="shuffle")),
    "subset": (
        "ds2 = subset(ds, indices)",
        "A datastore with only the files at the given indices (a vector of file numbers or a logical "
        "vector with one element per file), in that order, with their labels." + _DS.format(name="subset")),
    "partition": (
        ["part = partition(ds, n, k)", "part = partition(ds, 'Files', k)", "part = partition(ds, 'Files', name)"],
        "Divides the files of a datastore into n parts of nearly equal size and returns the k-th one, or "
        "returns a datastore with the single file k (or the file with that name)."
        + _DS.format(name="partition")),
    "numpartitions": (
        ["n = numpartitions(ds)", "n = numpartitions(ds, pool)"],
        "The largest number of parts partition can divide a datastore into: its number of files."
        + _DS.format(name="numpartitions")),
    "transform": (
        ["tds = transform(ds, @fcn)", "tds = transform(ds, @fcn, 'IncludeInfo', true)"],
        "A datastore that passes every read of ds through fcn: read(tds) returns fcn(read(ds)). With "
        "'IncludeInfo', true the function takes the data and the info struct and returns both. The "
        "datastore is copied and starts from the beginning; readall applies the function to every block. "
        "The result can be transformed or combined again." + _DS.format(name="transform")),
}
