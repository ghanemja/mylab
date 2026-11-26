"""Help text for the functions in builtins/tables_more.py."""

_VARS = ("Variables are selected by name, by number, with a logical vector, with vartype(...) or with a "
         "function handle such as @isnumeric.")
_GROUP = (" groupvars names the grouping variables. An optional groupbins argument puts a grouping variable "
          "into bins first: bin edges, a number of bins, or for datetimes a unit such as 'month', 'day', "
          "'hourofday', 'dayname'.")

HELP = {
    "vartype": ("S = vartype(type)",
                "A subscript that selects the variables of a table by type, e.g. T(:, vartype('numeric')). "
                "type is a class name or 'numeric', 'float', 'integer', 'cellstr'."),
    "discretize": (["Y = discretize(X, edges)", "Y = discretize(X, N)", "Y = discretize(X, edges, values)",
                    "Y = discretize(X, edges, 'categorical'[, names])",
                    "[Y, E] = discretize(..., 'IncludedEdge', side)"],
                   "Puts the values of X into bins and returns the bin number of each value (NaN outside the "
                   "edges). A bin includes its left edge, and the last bin also its right edge; with "
                   "'IncludedEdge', 'right' it is the other way round. With values the result holds the value "
                   "of each bin instead of its number; with 'categorical' it is an ordinal categorical array "
                   "whose categories are named after the bins, e.g. '[0, 4)'. N asks for N bins of equal "
                   "width. X and edges can be datetime or duration arrays."),
    "stack": (["S = stack(U, vars)", "S = stack(U, vars, Name, Value)", "[S, iu] = stack(...)"],
              "Turns several variables of a wide table into one variable of a tall table. Every row of U "
              "gives one row of S per stacked variable; the other variables are repeated. S gets an "
              "indicator variable (categorical, named after the data variable with _Indicator) that tells "
              "which variable each value came from. Options: 'NewDataVariableName', 'IndexVariableName', "
              "'ConstantVariables'. iu gives the row of U for each row of S."),
    "unstack": (["U = unstack(S, vars, ivar)", "U = unstack(S, vars, ivar, Name, Value)", "[U, is] = unstack(...)"],
                "The opposite of stack: the values of the variable vars are spread over one new variable per "
                "value of the indicator variable ivar. Rows that agree in the remaining (grouping) variables "
                "become one row. Several values in one place are summed, or combined with "
                "'AggregationFunction'; empty places are filled with missing values. Other options: "
                "'GroupingVariables', 'ConstantVariables', 'NewDataVariableNames', 'VariableNamingRule'."),
    "rowfun": (["B = rowfun(func, A)", "B = rowfun(func, A, Name, Value)"],
               "Calls func once for every row of the table A, with the variables as separate inputs, and "
               "collects the results in a table. Options: 'InputVariables', 'GroupingVariables' (func is then "
               "called once per group, with the rows of the group), 'OutputVariableNames', 'NumOutputs', "
               "'SeparateInputs' (false passes one row vector), 'ExtractCellContents', 'OutputFormat' "
               "('table', 'uniform', 'cell') and 'ErrorHandler'. " + _VARS),
    "groupcounts": (["G = groupcounts(T, groupvars)", "G = groupcounts(T, groupvars, groupbins)",
                     "[B, BG, P] = groupcounts(A)"],
                    "Counts the rows of each group. For a table the result has the grouping variables, "
                    "GroupCount and Percent. For an array the counts, the groups and the percentages are "
                    "returned separately." + _GROUP + " 'IncludeMissingGroups', false leaves out the groups "
                    "of missing values."),
    "groupfilter": (["G = groupfilter(T, groupvars, method)", "G = groupfilter(T, groupvars, method, datavars)"],
                    "Keeps the rows of a table that pass a test made group by group. method is a function "
                    "handle that gets the data of one variable for one group and returns true or false for "
                    "the whole group, or one logical value per row." + _GROUP),
    "grouptransform": (["G = grouptransform(T, groupvars, method)",
                        "G = grouptransform(T, groupvars, method, datavars)",
                        "G = grouptransform(..., 'ReplaceValues', false)"],
                       "Transforms the data of each group separately: 'zscore', 'norm', 'meancenter', "
                       "'rescale', 'meanfill', 'linearfill' or a function handle that returns one row per row "
                       "of the group (or one row for all). The transformed values replace the data variables, "
                       "or with 'ReplaceValues', false are added as new variables named method_variable."
                       + _GROUP),
    "pivot": ("P = pivot(T, Columns=colvars, Rows=rowvars, DataVariable=var, Method=method)",
              "A pivot table: one row for every value of the Rows variables and one variable for every "
              "value of the Columns variables. Each cell counts the matching rows of T, or with a "
              "DataVariable summarizes it with Method ('sum' by default; 'count', 'mean', 'median', 'min', "
              "'max', 'mode', 'std', 'var', 'range', 'numunique', 'nummissing', 'percentage', 'none' or a "
              "function handle). Other options: IncludeTotals, IncludeMissingGroups, RowsBinMethod, "
              "ColumnsBinMethod."),
    "mergevars": (["T2 = mergevars(T, vars)", "T2 = mergevars(T, vars, 'NewVariableName', name)",
                   "T2 = mergevars(T, vars, 'MergeAsTable', true)"],
                  "Combines several variables of a table into one variable with several columns (or into a "
                  "nested table). The new variable takes the place of the first merged one and is called "
                  "VarN (N is that position) unless a name is given."),
    "splitvars": (["T2 = splitvars(T)", "T2 = splitvars(T, vars)",
                   "T2 = splitvars(T, vars, 'NewVariableNames', names)"],
                  "Splits variables that have several columns, or that are nested tables, into separate "
                  "variables. The columns of X are called X_1, X_2, ... unless names are given; the variables "
                  "of a nested table keep their names."),
    "convertvars": ("T2 = convertvars(T, vars, dataType)",
                    "Converts the selected variables of a table with a conversion function given by name "
                    "('string', 'double', 'categorical', ...) or as a function handle. " + _VARS),
    "inner2outer": ("T2 = inner2outer(T)",
                    "Reorganizes a table whose variables are nested tables: the inner variable names become "
                    "the outer variables, each one a nested table with the former outer names."),
}
