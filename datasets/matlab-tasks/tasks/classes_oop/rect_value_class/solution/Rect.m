classdef Rect
    %RECT Axis-aligned rectangle (value class).
    properties
        Width
        Height
    end
    properties (Dependent)
        Area
        Perimeter
    end
    methods
        function obj = Rect(w, h)
            if w < 0 || h < 0
                error('Rect:negative', 'Sides must be nonnegative.');
            end
            obj.Width = w;
            obj.Height = h;
        end
        function a = get.Area(obj)
            a = obj.Width * obj.Height;
        end
        function p = get.Perimeter(obj)
            p = 2 * (obj.Width + obj.Height);
        end
        function r2 = scale(obj, f)
            r2 = Rect(obj.Width * f, obj.Height * f);
        end
        function tf = isSquare(obj)
            tf = obj.Width == obj.Height;
        end
    end
end
