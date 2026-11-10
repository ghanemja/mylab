classdef Vec2
    properties
        x = 0
        y = 0
    end
    properties (Constant)
        ZERO_NAME = 'origin'
    end
    properties (Dependent)
        len
    end
    methods
        function obj = Vec2(x, y)
            if nargin > 0
                obj.x = x;
                obj.y = y;
            end
        end
        function r = plus(a, b)
            r = Vec2(a.x + b.x, a.y + b.y);
        end
        function r = mtimes(a, k)
            r = Vec2(a.x * k, a.y * k);
        end
        function v = get.len(obj)
            v = hypot(obj.x, obj.y);
        end
        function obj = scale(obj, k)
            obj.x = obj.x * k;
            obj.y = obj.y * k;
        end
    end
    methods (Static)
        function v = unitX()
            v = Vec2(1, 0);
        end
    end
end
