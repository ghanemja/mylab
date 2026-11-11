classdef Circle < Shape
    %CIRCLE Circle given by its radius.
    properties (SetAccess = private)
        Radius
    end
    methods
        function obj = Circle(r)
            obj@Shape('circle');
            obj.Radius = r;
        end
        function a = area(obj)
            a = pi * obj.Radius ^ 2;
        end
        function p = perimeter(obj)
            p = 2 * pi * obj.Radius;
        end
    end
end
