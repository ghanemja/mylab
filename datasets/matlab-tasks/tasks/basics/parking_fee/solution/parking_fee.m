function fee = parking_fee(minutes)
%PARKING_FEE Fee for each stay: free up to 30 min, 2.50 per started hour, cap 18.
fee = 2.5 * ceil(minutes / 60);
fee(minutes <= 30) = 0;
fee = min(fee, 18);
fee = double(fee);
end
