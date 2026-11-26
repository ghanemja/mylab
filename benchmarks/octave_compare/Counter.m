classdef Counter < handle
    properties
        n = 0
    end
    methods
        function inc(obj, d)
            obj.n = obj.n + d;
        end
    end
end
