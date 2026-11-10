classdef Animal
    properties
        name
    end
    methods
        function obj = Animal(name)
            obj.name = name;
        end
        function s = speak(obj)
            s = [obj.name ' makes a sound'];
        end
        function s = describe(obj)
            s = ['I am ' obj.name];
        end
    end
end
