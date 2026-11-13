function todo_list
% To-do list: a table you can edit, with buttons that add and remove rows.
%
% The app is one function. The callbacks are nested functions, so they all see
% the same variables (tbl, status, ...). Tick the "Done" box of a task to finish it.

fig = uifigure('Name', 'To-do list', 'Position', [100 100 440 360]);
grid = uigridlayout(fig, [3 3]);
grid.RowHeight = {'fit', '1x', 'fit'};
grid.ColumnWidth = {'1x', 'fit', 'fit'};

task = uieditfield(grid, 'Placeholder', 'What needs doing?', 'Tag', 'newtask');
uibutton(grid, 'Text', 'Add', 'Tag', 'add', 'ButtonPushedFcn', @addTask);
uibutton(grid, 'Text', 'Clear done', 'Tag', 'clear', 'ButtonPushedFcn', @clearDone);

tbl = uitable(grid, 'ColumnName', {'Task', 'Done'}, 'ColumnEditable', [true true], ...
    'Data', {'Try the app builder', false; 'Read docs/APPS.md', false}, ...
    'CellEditCallback', @(src, event) refresh());
tbl.Layout.Row = 2;
tbl.Layout.Column = [1 3];

status = uilabel(grid, 'Text', '', 'Tag', 'status');
status.Layout.Row = 3;
status.Layout.Column = [1 3];

refresh();

    function addTask(~, ~)
        name = strtrim(task.Value);
        if isempty(name)
            uialert(fig, 'Type a task first.', 'Nothing to add', 'Icon', 'info');
            return
        end
        tbl.Data(end + 1, :) = {name, false};
        task.Value = '';
        refresh();
    end

    function clearDone(~, ~)
        if isempty(tbl.Data)
            return
        end
        done = cell2mat(tbl.Data(:, 2));
        tbl.Data = tbl.Data(~done, :);
        refresh();
    end

    function refresh()
        n = size(tbl.Data, 1);
        if n == 0
            status.Text = 'Nothing to do.';
        else
            status.Text = sprintf('%d of %d done', sum(cell2mat(tbl.Data(:, 2))), n);
        end
    end
end
