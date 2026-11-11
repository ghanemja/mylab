classdef Square < Shape
    %SQUARE Square given by its side length.
    properties (SetAccess = private)
        Side
    end
    methods
        function obj = Square(s)
            obj@Shape('square');
            obj.Side = s;
        end
        function a = area(obj)
            a = obj.Side ^ 2;
        end
        function p = perimeter(obj)
            p = 4 * obj.Side;
        end
    end
end
