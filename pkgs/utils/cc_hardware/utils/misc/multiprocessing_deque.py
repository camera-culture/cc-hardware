from multiprocessing import Queue


class MultiprocessingDeque:
    def __init__(self, maxsize: int):
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def push(self, item):
        # Remove the oldest item if the queue is full
        if self.queue.full():
            self.queue.get()
        self.queue.put(item)

    def pop(self):
        # Wait for an item if the queue is empty
        return self.queue.get()

    def size(self):
        return self.queue.qsize()
