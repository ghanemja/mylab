"""Help text for the low-level HDF5 functions of mylab/builtins/hdf5_lowlevel.py.

Written from scratch for mylab.
"""

_LL = (" Identifiers are H5ML.id handles; constants are given by name, and 'H5P_DEFAULT' is the only "
       "property list accepted. Runs on the Python package h5py, which is installed with mylab.")
_ORDER = (" Dimensions are in the C library's order, slowest first, and offsets start at 0; the MATLAB array "
          "that goes with a dataspace of dimensions [a b] is b-by-a.")

HELP = {
    "H5F.open": (["fid = H5F.open(filename)", "fid = H5F.open(filename, flags, fapl)"],
                 "Opens an existing HDF5 file and returns its identifier. flags is 'H5F_ACC_RDONLY' (the "
                 "default) or 'H5F_ACC_RDWR' for writing." + _LL),
    "H5F.create": (["fid = H5F.create(filename)", "fid = H5F.create(filename, flags, fcpl, fapl)"],
                   "Creates an HDF5 file and returns its identifier. With 'H5F_ACC_TRUNC' an existing file "
                   "is overwritten; with 'H5F_ACC_EXCL' (the default) creating it fails if it exists." + _LL),
    "H5F.close": (["H5F.close(fid)"],
                  "Closes a file identifier. The file itself is closed once the datasets, groups and "
                  "attributes opened in it are closed too." + _LL),
    "H5G.open": (["gid = H5G.open(loc_id, name)", "gid = H5G.open(loc_id, name, gapl)"],
                 "Opens a group, given by a path relative to a file or group identifier." + _LL),
    "H5G.create": (["gid = H5G.create(loc_id, name, lcpl, gcpl, gapl)", "gid = H5G.create(loc_id, name)"],
                   "Creates a group in a file or group and returns its identifier." + _LL),
    "H5G.close": (["H5G.close(gid)"], "Closes a group identifier." + _LL),
    "H5D.open": (["dset_id = H5D.open(loc_id, name)", "dset_id = H5D.open(loc_id, name, dapl)"],
                 "Opens a dataset, given by a path relative to a file or group identifier." + _LL),
    "H5D.create": (["dset_id = H5D.create(loc_id, name, type_id, space_id, dcpl)",
                    "dset_id = H5D.create(loc_id, name, type_id, space_id, lcpl, dcpl, dapl)"],
                   "Creates a dataset with the given datatype (an identifier or a predefined name such as "
                   "'H5T_NATIVE_DOUBLE') and dataspace. Without creation property lists the layout is "
                   "contiguous, so the dataspace cannot have unlimited dimensions." + _LL),
    "H5D.read": (["data = H5D.read(dset_id)",
                  "data = H5D.read(dset_id, mem_type, mem_space, file_space, dxpl)"],
                 "Reads a dataset. mem_type 'H5ML_DEFAULT' keeps the stored type; a predefined type name "
                 "converts to it. 'H5S_ALL' for both spaces reads everything; a file dataspace with a "
                 "hyperslab selection (H5S.select_hyperslab) reads that block. Numbers come back in their "
                 "class, fixed-length strings as columns of a char array and variable-length strings as a "
                 "cell array." + _ORDER + _LL),
    "H5D.write": (["H5D.write(dset_id, mem_type, mem_space, file_space, dxpl, data)"],
                  "Writes data to a dataset, all of it with 'H5S_ALL', 'H5S_ALL', or the block selected in "
                  "the file dataspace. The data is converted to mem_type ('H5ML_DEFAULT': the dataset's "
                  "type) and must have as many elements as the selection." + _ORDER + _LL),
    "H5D.get_space": (["space_id = H5D.get_space(dset_id)"],
                      "Returns a copy of a dataset's dataspace, which can be inspected or given a hyperslab "
                      "selection for H5D.read and H5D.write." + _LL),
    "H5D.get_type": (["type_id = H5D.get_type(dset_id)"], "Returns a copy of a dataset's datatype." + _LL),
    "H5D.close": (["H5D.close(dset_id)"], "Closes a dataset identifier." + _LL),
    "H5S.create_simple": (["space_id = H5S.create_simple(rank, dims, maxdims)"],
                          "Creates a simple dataspace of rank dimensions. maxdims is [] (the same as dims) or "
                          "has one value per dimension, where -1 or Inf means unlimited." + _ORDER + _LL),
    "H5S.create": (["space_id = H5S.create(type)"],
                   "Creates a dataspace of type 'H5S_SCALAR' (one element, for example for an attribute), "
                   "'H5S_SIMPLE' or 'H5S_NULL'." + _LL),
    "H5S.get_simple_extent_dims": (["[ndims, dims, maxdims] = H5S.get_simple_extent_dims(space_id)"],
                                   "Returns the number of dimensions, the current dimensions and the maximum "
                                   "dimensions of a dataspace; an unlimited maximum is returned as -1."
                                   + _ORDER + _LL),
    "H5S.select_hyperslab": (["H5S.select_hyperslab(space_id, op, start, stride, count, block)"],
                             "Selects a regular block pattern in a dataspace: count blocks of size block, "
                             "spaced stride apart, from start (zero-based). [] for stride or block means "
                             "ones. op is 'H5S_SELECT_SET' to replace the selection, or 'H5S_SELECT_OR', "
                             "'H5S_SELECT_AND', 'H5S_SELECT_XOR', 'H5S_SELECT_NOTB' or 'H5S_SELECT_NOTA' "
                             "to combine it with the current one." + _ORDER + _LL),
    "H5S.close": (["H5S.close(space_id)"], "Closes a dataspace identifier." + _LL),
    "H5A.open": (["attr_id = H5A.open(obj_id, attr_name)", "attr_id = H5A.open(obj_id, attr_name, aapl)"],
                 "Opens an attribute of a file (its root group), group or dataset." + _LL),
    "H5A.create": (["attr_id = H5A.create(loc_id, name, type_id, space_id, acpl)",
                    "attr_id = H5A.create(loc_id, name, type_id, space_id, acpl, aapl)"],
                   "Creates an attribute on a file, group or dataset, with a datatype and a dataspace "
                   "(H5S.create('H5S_SCALAR') for a single value). Write its value with H5A.write." + _LL),
    "H5A.read": (["data = H5A.read(attr_id)", "data = H5A.read(attr_id, mem_type)"],
                 "Reads the value of an attribute, converted as H5D.read converts data." + _ORDER + _LL),
    "H5A.write": (["H5A.write(attr_id, mem_type, data)"],
                  "Writes the value of an attribute; mem_type 'H5ML_DEFAULT' uses the attribute's type."
                  + _ORDER + _LL),
    "H5A.close": (["H5A.close(attr_id)"], "Closes an attribute identifier." + _LL),
    "H5T.copy": (["type_id = H5T.copy(type)"],
                 "Copies a datatype, given as an identifier, a dataset identifier or a predefined name: "
                 "'H5T_NATIVE_DOUBLE', 'H5T_NATIVE_FLOAT', 'H5T_NATIVE_INT', 'H5T_NATIVE_UINT8' and the "
                 "other native integer types, 'H5T_IEEE_F64LE' and the other standard types, or "
                 "'H5T_C_S1' for text. The copy can be changed, for example with H5T.set_size." + _LL),
    "H5T.set_size": (["H5T.set_size(type_id, size)", "H5T.set_size(type_id, 'H5T_VARIABLE')"],
                     "Sets the size in bytes of a datatype; for a string type copied from 'H5T_C_S1' this "
                     "is the length of the strings, or 'H5T_VARIABLE' for variable-length strings." + _LL),
    "H5T.close": (["H5T.close(type_id)"], "Closes a datatype identifier." + _LL),
    "H5ML.get_constant_value": (["value = H5ML.get_constant_value(name)"],
                                "Returns the number that stands for a named HDF5 constant such as "
                                "'H5F_ACC_RDWR' or 'H5S_UNLIMITED'. The low-level functions also accept "
                                "the names themselves." + _LL),
}
