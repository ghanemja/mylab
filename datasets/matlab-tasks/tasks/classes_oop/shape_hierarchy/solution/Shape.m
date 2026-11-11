classdef (Abstract) Shape
    %SHAPE Abstract base class for plane shapes.
    properties
        Name = ''
    end
    methods (Abstract)
        a = area(obj)
        p = perimeter(obj)
    end
    methods
        function obj = Shape(name)
            if nargin > 0
                obj.Name = name;
            end
        end
        function c = compactness(obj)
            c = 4 * pi * area(obj) / perimeter(obj) ^ 2;
        end
        function tf = isLargerThan(obj, other)
            tf = area(obj) > area(other);
        end
    end
    methods (Static)
        function t = totalArea(shapes)
            t = 0;
            for k = 1:numel(shapes)
                t = t + area(shapes{k});
            end
        end
    end
end
