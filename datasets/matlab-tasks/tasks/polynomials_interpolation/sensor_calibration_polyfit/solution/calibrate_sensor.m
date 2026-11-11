function [c, rmse] = calibrate_sensor(raw, ref, deg)
%CALIBRATE_SENSOR Least-squares polynomial calibration curve.
raw = raw(:);
ref = ref(:);
if numel(raw) < deg + 1
    error('calibrate_sensor:points', 'Need at least deg + 1 points.');
end
c = polyfit(raw, ref, deg);
c = reshape(c, 1, []);
rmse = sqrt(mean((polyval(c, raw) - ref).^2));
end
