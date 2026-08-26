"""Top-K selection in O(N log K) — slides 8-Scoring, s13-14.

ponytail: heapq.nlargest is the stdlib one-liner and is what you would reach
for in ordinary code. Written out here because selecting top-K without sorting
all N is the point being demonstrated, and the course specifies the min-heap.
"""


class TopKMinHeap:
    """Fixed-capacity binary min-heap keeping the K highest-scoring items.

    The root is always the weakest of the K kept so far, so a new candidate
    can be rejected in O(1) and admitted in O(log K).
    """

    def __init__(self, k: int):
        self.k = k
        self.heap: list[tuple[float, str]] = []

    def _sift_up(self, i: int):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[i][0] < self.heap[parent][0]:
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int):
        n = len(self.heap)
        while True:
            smallest, left, right = i, 2 * i + 1, 2 * i + 2
            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right
            if smallest == i:
                break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def push(self, score: float, item: str):
        if len(self.heap) < self.k:
            self.heap.append((score, item))
            self._sift_up(len(self.heap) - 1)
        elif score > self.heap[0][0]:  # beats the weakest kept -> replace root
            self.heap[0] = (score, item)
            self._sift_down(0)

    def ranked(self) -> list[tuple[float, str]]:
        """The kept items, best first."""
        return sorted(self.heap, key=lambda pair: pair[0], reverse=True)
