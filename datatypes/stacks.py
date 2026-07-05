from abc import abstractmethod, ABC
from linked_list import SinglyLinkedList, SinglyLinkedNode, DoublyLinkedHead, DoublyLinkedNode, DoublyLinkedTail, DoublyLinkedList

class Class(ABC):
    @abstractmethod
    def pop(self):
        pass

    @abstractmethod
    def push(self, item):
        pass

    @abstractmethod
    def peek(self):
        pass


class LinkedListStack(Class):
    def __init__(self, items):
        self.nums = SinglyLinkedList(items)
        self.len = self.nums.length

    def push(self, item):
        self.nums.head = SinglyLinkedNode(val=item, next=self.nums.head)
    def pop(self):
        temp = self.nums.head.val
        self.nums.head = self.nums.head.next
        return temp
    def peek(self):
        return self.nums.head.val
    def is_empty(self):
        return self.len == 0
    
    def __str__(self):
        return f'{self.nums.ll_to_list()[:-1]}'




# a = Stack([])
# print(a)
# a.push(1)
# print(a)
# a.push(2)
# print(a)
# print(a.pop())
# print(a)
# print(a.peek())
# print(a)

        
