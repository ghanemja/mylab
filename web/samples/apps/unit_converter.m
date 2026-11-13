% Unit converter: an app with a grid layout, an edit field, two drop-downs and a button.
%
% Run it in the mylab web app (Run), or from a terminal with:  mylab unit_converter.m

fig = uifigure('Name', 'Unit converter', 'Position', [100 100 420 190]);
grid = uigridlayout(fig, [4 3]);
grid.RowHeight = {'fit', 'fit', 'fit', '1x'};
grid.ColumnWidth = {'1x', 'fit', '1x'};

uilabel(grid, 'Text', 'Value');
uilabel(grid, 'Text', '');
uilabel(grid, 'Text', 'Kind');
value = uieditfield(grid, 'numeric', 'Value', 1, 'Tag', 'input');
uilabel(grid, 'Text', '');
kind = uidropdown(grid, 'Items', {'Length', 'Mass', 'Temperature'}, 'Tag', 'kind');

from = uidropdown(grid, 'Tag', 'from');
swap = uibutton(grid, 'Text', 'Swap', 'Tag', 'swap', 'Tooltip', 'Swap the two units');
to = uidropdown(grid, 'Tag', 'to');

result = uilabel(grid, 'Text', '', 'Tag', 'result', 'FontSize', 18, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
result.Layout.Row = 4;
result.Layout.Column = [1 3];

% Everything the callbacks need travels in one struct.
ui = struct('value', value, 'kind', kind, 'from', from, 'to', to, 'result', result);

kind.ValueChangedFcn = @(src, event) pickKind(ui);
value.ValueChangedFcn = @(src, event) convert(ui);
from.ValueChangedFcn = @(src, event) convert(ui);
to.ValueChangedFcn = @(src, event) convert(ui);
swap.ButtonPushedFcn = @(src, event) swapUnits(ui);

pickKind(ui);

function pickKind(ui)
    % Fill the two unit lists for the chosen kind of quantity.
    switch ui.kind.Value
        case 'Length'
            units = {'km', 'm', 'cm', 'mi', 'ft', 'in'};
        case 'Mass'
            units = {'kg', 'g', 'lb', 'oz'};
        otherwise
            units = {'C', 'F', 'K'};
    end
    ui.from.Items = units;
    ui.to.Items = units;
    ui.from.Value = units{1};
    ui.to.Value = units{min(4, numel(units))};
    convert(ui);
end

function swapUnits(ui)
    a = ui.from.Value;
    ui.from.Value = ui.to.Value;
    ui.to.Value = a;
    convert(ui);
end

function convert(ui)
    x = ui.value.Value;
    if strcmp(ui.kind.Value, 'Temperature')
        y = fromKelvin(toKelvin(x, ui.from.Value), ui.to.Value);
    else
        y = x * factor(ui.from.Value) / factor(ui.to.Value);
    end
    ui.result.Text = sprintf('%g %s = %.5g %s', x, ui.from.Value, y, ui.to.Value);
end

function f = factor(unit)
    % Size of one unit in metres (lengths) or kilograms (masses).
    names = {'km', 'm', 'cm', 'mi', 'ft', 'in', 'kg', 'g', 'lb', 'oz'};
    sizes = [1000, 1, 0.01, 1609.344, 0.3048, 0.0254, 1, 0.001, 0.45359237, 0.028349523125];
    f = sizes(strcmp(names, unit));
end

function k = toKelvin(x, unit)
    switch unit
        case 'C', k = x + 273.15;
        case 'F', k = (x - 32) * 5 / 9 + 273.15;
        otherwise, k = x;
    end
end

function x = fromKelvin(k, unit)
    switch unit
        case 'C', x = k - 273.15;
        case 'F', x = (k - 273.15) * 9 / 5 + 32;
        otherwise, x = k;
    end
end
