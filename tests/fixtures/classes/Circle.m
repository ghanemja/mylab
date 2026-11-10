classdef Circle < Shape
    properties
        r (1,1) double {mustBePositive} = 1
        tag string = "none"
    end
    methods
        function obj = Circle(r)
            obj.name = 'circle';
            if nargin > 0
                obj.r = r;
            end
        end
        function a = area(obj)
            a = pi * obj.r^2;
        end
        function describe(obj)
            describe@Shape(obj);
            fprintf('(radius %g)\n', obj.r);
        end
        function delete(obj)
            fprintf('bye\n');
        end
    end
    methods (Access = private)
        function hidden(obj)
        end
    end
end
