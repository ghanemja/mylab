classdef Dog < Animal
    properties
        tricks = {}
    end
    methods
        function obj = Dog(name)
            obj = obj@Animal(name);
            obj.tricks = {'sit'};
        end
        function s = speak(obj)
            s = [obj.name ' says woof'];
        end
        function s = both(obj)
            s = [speak@Animal(obj) ' / ' obj.speak()];
        end
    end
end
