classdef Triangle < Shape
    %TRIANGLE Triangle given by its three side lengths.
    properties (SetAccess = private)
        Sides
    end
    methods
        function obj = Triangle(a, b, c)
            obj@Shape('triangle');
            v = sort([a b c]);
            if v(3) >= v(1) + v(2)
                error('Triangle:invalid', 'Sides do not form a triangle.');
            end
            obj.Sides = [a b c];
        end
        function A = area(obj)
            s = sum(obj.Sides) / 2;
            A = sqrt(s * prod(s - obj.Sides));
        end
        function p = perimeter(obj)
            p = sum(obj.Sides);
        end
    end
end
