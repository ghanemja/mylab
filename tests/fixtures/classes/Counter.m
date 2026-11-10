classdef Counter < handle
    properties
        count = 0
    end
    methods
        function increment(obj, n)
            if nargin < 2, n = 1; end
            obj.count = obj.count + n;
        end
        function disp(obj)
            fprintf('  Counter at %d\n', obj.count);
        end
    end
end
