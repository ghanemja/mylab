classdef TaskQueue < handle
    %TASKQUEUE Min-heap priority queue with FIFO tie-breaking.
    properties (Access = private)
        Keys = zeros(0, 2)   % [priority, sequence]
        Names = cell(0, 1)
        N = 0
        Seq = 0
    end
    properties (Dependent)
        Count
    end
    methods
        function obj = TaskQueue()
        end
        function c = get.Count(obj)
            c = obj.N;
        end
        function add(obj, priority, name)
            obj.Seq = obj.Seq + 1;
            obj.N = obj.N + 1;
            i = obj.N;
            obj.Keys(i, :) = [priority, obj.Seq];
            obj.Names{i, 1} = name;
            while i > 1
                parent = floor(i / 2);
                if TaskQueue.less(obj.Keys(i, :), obj.Keys(parent, :))
                    obj.swap(i, parent);
                    i = parent;
                else
                    break
                end
            end
        end
        function [name, priority] = next(obj)
            if obj.N == 0
                error('TaskQueue:empty', 'Queue is empty.');
            end
            name = obj.Names{1};
            priority = obj.Keys(1, 1);
            obj.swap(1, obj.N);
            obj.N = obj.N - 1;
            obj.Keys(end, :) = [];
            obj.Names(end) = [];
            i = 1;
            while true
                l = 2 * i;
                r = l + 1;
                smallest = i;
                if l <= obj.N && TaskQueue.less(obj.Keys(l, :), obj.Keys(smallest, :))
                    smallest = l;
                end
                if r <= obj.N && TaskQueue.less(obj.Keys(r, :), obj.Keys(smallest, :))
                    smallest = r;
                end
                if smallest == i
                    break
                end
                obj.swap(i, smallest);
                i = smallest;
            end
        end
        function p = peekPriority(obj)
            if obj.N == 0
                error('TaskQueue:empty', 'Queue is empty.');
            end
            p = obj.Keys(1, 1);
        end
    end
    methods (Access = private)
        function swap(obj, i, j)
            obj.Keys([i j], :) = obj.Keys([j i], :);
            obj.Names([i j]) = obj.Names([j i]);
        end
    end
    methods (Static, Access = private)
        function tf = less(a, b)
            tf = a(1) < b(1) || (a(1) == b(1) && a(2) < b(2));
        end
    end
end
