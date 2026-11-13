% Plot explorer: move the slider and the plot follows.
%
% Shows uiaxes, uislider (ValueChangingFcn fires while you drag), a drop-down and a check box.

fig = uifigure('Name', 'Plot explorer', 'Position', [100 100 560 420]);
grid = uigridlayout(fig, [3 4]);
grid.RowHeight = {'1x', 'fit', 'fit'};
grid.ColumnWidth = {'fit', '1x', 'fit', 'fit'};

ax = uiaxes(grid);
ax.Layout.Row = 1;
ax.Layout.Column = [1 4];

uilabel(grid, 'Text', 'Frequency');
freq = uislider(grid, 'Limits', [1 10], 'Value', 2, 'Tag', 'frequency');
shape = uidropdown(grid, 'Items', {'sine', 'square', 'sawtooth'}, 'Tag', 'shape');
damped = uicheckbox(grid, 'Text', 'Damped', 'Tag', 'damped');

readout = uilabel(grid, 'Text', '', 'Tag', 'readout');
readout.Layout.Row = 3;
readout.Layout.Column = [1 4];

freq.ValueChangingFcn = @(src, event) draw(ax, readout, event.Value, shape.Value, damped.Value);
freq.ValueChangedFcn = @(src, event) draw(ax, readout, src.Value, shape.Value, damped.Value);
shape.ValueChangedFcn = @(src, event) draw(ax, readout, freq.Value, src.Value, damped.Value);
damped.ValueChangedFcn = @(src, event) draw(ax, readout, freq.Value, shape.Value, src.Value);

draw(ax, readout, freq.Value, shape.Value, damped.Value);

function draw(ax, readout, f, shape, damped)
    t = linspace(0, 2, 600);
    switch shape
        case 'square'
            y = sign(sin(2 * pi * f * t));
        case 'sawtooth'
            y = 2 * (f * t - floor(f * t + 0.5));
        otherwise
            y = sin(2 * pi * f * t);
    end
    if damped
        y = y .* exp(-1.5 * t);
    end
    plot(ax, t, y, 'LineWidth', 1.5);
    ylim(ax, [-1.2 1.2]);
    grid(ax, 'on');
    xlabel(ax, 'time (s)');
    title(ax, sprintf('%s wave, %.1f Hz', shape, f));
    readout.Text = sprintf('Frequency %.1f Hz, period %.3f s', f, 1 / f);
end
