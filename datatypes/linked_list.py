class SinglyLinkedNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next

class SinglyLinkedList:
    def __init__(self, items: list):
        self.head: SinglyLinkedNode = self.list_to_ll(items)
        self.length = len(items)
    
    def ll_to_list(self):
        def recurse(head: SinglyLinkedNode, out=[]):
            if head:
                out.append(head.val)
                recurse(head.next, out)
        out = []
        recurse(self.head, out)
        return out
    

    def _append(self, head, val):
        if head:
            if head.next is not None:
                self._append(head.next, val)
            else:
                head.next = SinglyLinkedNode(val, None)
        return head


    def list_to_ll(self, items: list):
        if len(items) == 0:
            return SinglyLinkedNode(None, None)
        head = SinglyLinkedNode(items[0], None)
        for i in items[1:]:
            head = self._append(head, i)
        return head
    
    def index(self, val):
        ind = 0
        scope = 'self.head'
        while len(scope.split('.')) <= self.length + 1:
            if eval(scope + '.val') == val:
                return ind
            else:
                ind += 1
                scope += '.next'
        raise IndexError()

    def append(self, val):
        if self.head:
            if self.head.next is not None:
                self._append(self.head.next, val)
            else:
                self.head.next = SinglyLinkedNode(val, None)
        self.length += 1
    def pop(self, val=None):
        '''remove item based on INDEX!'''
        if val == None or self.index(val) == self.length - 1:
            def _pop(node):
                if node.next.next is None:
                    node.next = None
                    return
                _pop(node.next)
            if self.head is None or self.head.next is None:
                self.head = None
                return

            _pop(self.head)
        elif self.index(val) == 0:
            self.head = self.head.next
        else:
            def b(node):
                if node.next.val == val:
                    node.next = node.next.next
                    return
                b(node.next)
            b(self.head)
        self.length -= 1
    def __str__(self):
        return f'{self.ll_to_list()}'

            


# a = SinglyLinkedList([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# a.pop()
# print(a.ll_to_list())
# a.pop(6)
# print(a.ll_to_list())
# a.pop(1)
# print(a.ll_to_list())
# a.pop(9)
# print(a.ll_to_list())
# a.pop(3)
# print(a.ll_to_list())
# a.append(11)
# print(a.ll_to_list())

class DoublyLinkedNode:
    def __init__(self, val, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedHead:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class DoublyLinkedTail:
    def __init__(self, val, prev=None):
        self.val = val
        self.prev = prev

class DoublyLinkedList:
    def __init__(self, items):
        self.head = self.list_to_dll(items)
    
    def list_to_dll(self, items: list):
        def _append(head, val):
            if head:
                if head.next is None:
                    head.next = DoublyLinkedNode(val, head, None)
                    return head
                else:
                    _append(head.next, val)
        start = DoublyLinkedHead(items[0], None)
        for i in items[1:-1]:
            _append(start, i)
        def _append_tail(head, val):
            if head:
                if head.next is None:
                    head.next = DoublyLinkedTail(val, head)
                    return head
                else:
                    _append_tail(head.next, val)
        _append_tail(start, items[-1])
        return start
    
    def dll_to_list(self):
        def recurse(head, out=[]):
            if head:
                out.append(head.val)
                if isinstance(head, DoublyLinkedTail):
                    return 
                recurse(head.next, out)
        out = []
        recurse(self.head, out)
        return out
    
    def append(self, val):
        def _append(head, val):
            if head:
                if isinstance(head.next, DoublyLinkedTail):
                    head.next = DoublyLinkedNode(head.next.val, head, None)
                    head.next.next = DoublyLinkedTail(val, head.next)
                    return head
                else:
                    _append(head.next, val)
        _append(self.head, val)
    
    def __str__(self):
        return f'{self.dll_to_list()}'
    
# b = DoublyLinkedList([1, 2, 3])
# print(b)
# b.append(4)
# print(b, b.head.next.next.next.val, b.head.next.next.next.prev.val)
