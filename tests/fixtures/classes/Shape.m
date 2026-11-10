classdef (Abstract) Shape < handle
    properties (SetAccess = protected)
        name = 'shape'
    end
    properties (Access = private)
        secret = 42
    end
    methods (Abstract)
        a = area(obj)
    end
    methods
        function describe(obj)
            fprintf('%s with area %.2f\n', obj.name, obj.area());
        end
        function s = peek(obj)
            s = obj.secret;
        end
    end
end
