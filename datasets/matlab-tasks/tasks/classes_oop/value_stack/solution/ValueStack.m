classdef ValueStack < handle
    %VALUESTACK LIFO stack of arbitrary values.
    properties (Access = private)
        Items = cell(1, 0)
    end
    properties (Dependent)
        Size
    end
    methods
        function obj = ValueStack()
        end
        function push(obj, varargin)
            for k = 1:numel(varargin)
                obj.Items{end + 1} = varargin{k};
            end
        end
        function x = pop(obj)
            if isempty(obj.Items)
                error('ValueStack:empty', 'Stack is empty.');
            end
            x = obj.Items{end};
            obj.Items(end) = [];
        end
        function x = peek(obj)
            if isempty(obj.Items)
                error('ValueStack:empty', 'Stack is empty.');
            end
            x = obj.Items{end};
        end
        function n = get.Size(obj)
            n = numel(obj.Items);
        end
        function tf = isEmpty(obj)
            tf = isempty(obj.Items);
        end
        function c = toCell(obj)
            c = obj.Items;
        end
    end
end
