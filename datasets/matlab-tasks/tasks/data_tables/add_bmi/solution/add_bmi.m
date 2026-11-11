function T = add_bmi(T)
%ADD_BMI Append rounded BMI and a band label.
bmi = round(T.MassKg ./ T.HeightM .^ 2, 1);
band = repmat({'normal'}, height(T), 1);
band(bmi < 18.5) = {'low'};
band(bmi >= 25) = {'high'};
T.BMI = bmi;
T.Band = band;
end
