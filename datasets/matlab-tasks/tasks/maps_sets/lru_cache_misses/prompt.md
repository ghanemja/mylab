Write a MATLAB function
`[misses, contents] = lru_cache_misses(requests, capacity)`.

A cache can hold at most `capacity` distinct items (a nonnegative integer).
Items are requested one after another in the order given by the numeric
vector `requests`. For each request:

- if the item is in the cache, it is a **hit** and the item becomes the most
  recently used;
- otherwise it is a **miss**: if the cache is full, the least recently used
  item is evicted, and then the requested item is inserted as the most
  recently used. (With `capacity` 0 nothing is ever stored, so every request
  misses.)

Return the number of misses (a double) and `contents`, a row vector of the
items in the cache at the end, ordered from most recently used to least
recently used (`zeros(1,0)` if the cache is empty).

Example:

```matlab
[m, c] = lru_cache_misses([1 2 3 1 4 2], 3)
% m = 5, c = [2 4 1]
```

(Requests 1, 2, 3 miss; 1 hits; 4 misses and evicts 2; 2 misses and evicts 3.)
