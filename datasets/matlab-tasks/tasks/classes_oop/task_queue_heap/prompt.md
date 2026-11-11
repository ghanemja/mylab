Write a MATLAB **handle** class `TaskQueue` (file `TaskQueue.m`): a priority
queue of named tasks, implemented as a binary min-heap so that adding and
removing take O(log n) time.

API:

- `q = TaskQueue()` creates an empty queue.
- `add(q, priority, name)` inserts a task with a numeric `priority` (smaller
  is more urgent) and a character-vector `name`.
- `[name, priority] = next(q)` removes and returns the most urgent task. Among
  tasks with equal priority, the one that was added **first** comes out first.
- `p = peekPriority(q)` returns the priority of the task `next` would return,
  without removing it.
- `next` and `peekPriority` on an empty queue throw an error with identifier
  `'TaskQueue:empty'`.
- Dependent property `Count` — the number of tasks currently queued.

Example:

```matlab
q = TaskQueue();
add(q, 5, 'write'); add(q, 1, 'plan'); add(q, 5, 'test');
next(q)     % 'plan'
next(q)     % 'write'   (added before 'test')
q.Count     % 1
```
