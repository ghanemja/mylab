names = {'b01_scalar_loop', 'b02_array_fill', 'b03_while_if', 'b04_nested_matrix', 'b05_recursion_fib22', ...
    'b06_call_user_function', 'b07_call_anonymous', 'b08_struct_fields', 'b09_cell_and_string_build', ...
    'b10_small_array_ops', 'b11_object_methods', 'b12_map_inserts', 'b13_ode45_vdp', 'b14_fminsearch_rosen', ...
    'b15_bubble_sort', 'b16_text_parsing', 'b20_vec_elementwise', 'b21_matmul_1500', 'b22_solve_1500', ...
    'b23_eig_500', 'b24_svd_600', 'b25_fft_4m', 'b26_sort_unique_5m', 'b27_logical_index_1e7', ...
    'b28_sparse_tridiag_solve', 'b29_arrayfun_cellfun', 'b30_accumarray_interp', 'b31_csv_write_read', ...
    'b32_plot_to_png', 'b33_regexprep_cellstr'};
for i = 1:numel(names)
    best = Inf; val = NaN; err = '';
    for rep = 1:3
        try
            t0 = tic; val = feval(names{i}); el = toc(t0);
            best = min(best, el);
        catch e
            err = e.message; break
        end
        if el > 20, break; end
    end
    fprintf('BENCH\t%s\t%.4f\t%.10g\t%s\n', names{i}, best, val, strrep(err, sprintf('\n'), ' '));
end
