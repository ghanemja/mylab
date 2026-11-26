"""Help text for the dictionary type."""

_D = "d = dictionary([\"a\" \"b\"], [1 2]); d(\"c\") = 3; d(\"b\")"

HELP = {
    "dictionary": (["d = dictionary", "d = dictionary(keys, values)", "d = dictionary(k1, v1, ..., kN, vN)"],
                   "A map from keys to values that remembers the order in which entries were added. All keys "
                   "have one type and all values have one type; text is stored as strings. keys and values "
                   "are arrays of the same size (or one value for all the keys). Look up with d(keys), set with "
                   "d(keys) = values, delete with d(keys) = [] or remove. For values of any kind use cells: "
                   "d(key) = {value} and d{key}. A dictionary is a value, so d2 = d makes an independent "
                   "copy. Functions: keys, values, entries, numEntries, isKey, insert, remove, lookup, types, "
                   "isConfigured. Example: " + _D),
    "configureDictionary": ("d = configureDictionary(keyType, valueType)",
                            "An empty dictionary whose key and value types are fixed before any entry is "
                            "added, e.g. configureDictionary(\"string\", \"double\")."),
    "isConfigured": ("tf = isConfigured(d)", "True when the dictionary d has key and value types, which it "
                     "gets from configureDictionary or from its first entry."),
    "numEntries": ("n = numEntries(d)", "Number of key-value pairs in the dictionary d."),
    "types": (["[keyType, valueType] = types(d)", "t = types(d)"],
              "The class names of the keys and of the values of the dictionary d, as strings."),
    "entries": (["E = entries(d)", "E = entries(d, format)"],
                "The key-value pairs of the dictionary d: a table with the variables Key and Value, or with "
                "format \"struct\" a struct array with those fields, or with \"cell\" an N-by-2 cell array."),
    "insert": (["d = insert(d, keys, values)", "d = insert(d, keys, values, 'Overwrite', false)"],
               "Adds entries to the dictionary d and returns it. With 'Overwrite', false it is an error if a "
               "key is already there."),
    "lookup": (["v = lookup(d, keys)", "v = lookup(d, keys, 'FallbackValue', value)"],
               "The values of the given keys in the dictionary d. With 'FallbackValue' a key that is not in "
               "the dictionary gives that value instead of an error."),
    "isdictionary": ("tf = isdictionary(x)", "True when x is a dictionary."),
}
