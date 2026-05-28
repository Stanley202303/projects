class SinglyLinkedNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next

class SinglyLinkedList:
    def __init__(self, items: list):
        self.head: SinglyLinkedNode = self.list_to_ll(items)
    
    def ll_to_list(self):
        def recurse(head: SinglyLinkedNode, out=[]):
            if head:
                out.append(head.val)
                recurse(head.next)
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

a = SinglyLinkedList([1, 2, 3])
print(a.ll_to_list())