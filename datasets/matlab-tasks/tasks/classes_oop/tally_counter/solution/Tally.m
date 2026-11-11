classdef Tally < handle
    %TALLY Hand-held tally counter.
    properties (SetAccess = private)
        Count = 0
    end
    properties (Access = private)
        Moved = 0
    end
    methods
        function obj = Tally(start)
            if nargin > 0
                obj.Count = start;
            end
        end
        function increment(obj, k)
            if nargin < 2
                k = 1;
            end
            obj.Count = obj.Count + k;
            obj.Moved = obj.Moved + abs(k);
        end
        function reset(obj)
            obj.Count = 0;
        end
        function n = total(obj)
            n = obj.Moved;
        end
    end
end
