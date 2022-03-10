
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.top = None
        self.end = None
    def add(self,node):
        node = Node(node)
        if self.top == None:
            self.top = node
            self.end = node
        else:
            self.end.next = node
            self.end = node
            
    def reversebygroup(self,group):
        self.top = self.reverse(group,self.top)
        return self.display()
    
    def reverse(self,k, head):
        if head == None:
            return None
        current = head
        next = None
        prev = None
        count = 0
 
        while(current is not None and count < k):
            next = current.next
            current.next = prev
            prev = current
            current = next
            count += 1
 
        if next is not None:
            head.next = self.reverse(k,next)
 
        return prev
    
    def display(self):
        if(self.top == None):
            print('List is empty')
            return
        temp = self.top
        print(temp.data,end= '->')
        while temp.next != None:
            temp = temp.next
            print(temp.data,end= '->')
    
    def mergealternatively(self,LL):
        self.merge(self.top,LL)
        return self.display()
    
        if self.top == None or LL.top == None:
            self.top = LL.top
            return
        
        temp = self.top
        temp2 = LL.top
        while temp != None:
            mainnext = temp.next
            temp.next = temp2
            LL.next = mainnext
            temp = LL.next
            
    def merge(self, p, q):
        p_curr = p
        q_curr = q.top
 
        while p_curr != None and q_curr != None:

            p_next = p_curr.next
            q_next = q_curr.next
 
            q_curr.next = p_next  
            p_curr.next = q_curr  
 
            p_curr = p_next
            q_curr = q_next
            q.head = q_curr     

    
    def removeZeroSum(self):
        if self.top == None:
            return "stack is empty"
        else:
            prev = self.top
            current = self.top
            while current != None:
                sum = current.data
                next = current.next
                while next != None:
                    sum += next.data
                    next = next.next
                    if sum == 0:
                        if current == self.top:
                            self.top = next
                        else:
                            prev.next = next
                prev = current
                current = current.next

        return self.display()

#1. Delete the elements in an linked list whose sum is equal to zero

ll = LinkedList()
ll.add(5)
ll.add(-2)
ll.add(3)
ll.add(-2)
ll.add(-1)
ll.add(6)
ll.removeZeroSum()
print()

#2. Reverse a linked list in groups of given size
ll = LinkedList()
ll.add(10)
ll.add(20)
ll.add(30)
ll.add(40)
ll.add(50)
ll.add(60)
ll.reversebygroup(3)  

print()


#3. Merge a linked list into another linked list at alternate positions.
ll = LinkedList()
ll.add(10)
ll.add(20)
ll.add(30)
ll2 = LinkedList()
ll2.add(40)
ll2.add(50)
ll2.add(60)
ll.mergealternatively(ll2)

#4. In an array, Count Pairs with given sum
listvalues = list(map(int,input('enter space seperated values').split()))
sum = int(input('enter sum value'))

for i in range(len(listvalues)):
    for j in range(i,len(listvalues)):
        if listvalues[i] + listvalues[j] == sum:
            print('({0},{1})'.format(listvalues[i],listvalues[j]))
            
            
#5. Find duplicates in an array
from collections import Counter
listvalues = list(map(int,input('enter space seperated values').split()))
dic = Counter(listvalues)
for k in dic:
    if dic[k]>1:
        print('duplicate values found for',k)
        
        
#6. Find the Kth largest and Kth smallest number in an array
listvalues = list(map(int,input('enter space seperated values').split()))
k = int(input('enter k value'))
listvalues.sort()
print('kth largest value is',listvalues[len(listvalues)-k])
print('kth smallest value is',listvalues[k-1])

#7. Move all the negative elements to one side of the array
listvalues = list(map(int,input('enter space seperated values').split()))
j = 0
for i in range(0, len(listvalues)):
    if (listvalues[i] < 0) :
        temp = listvalues[i]
        listvalues[i] = listvalues[j]
        listvalues[j]= temp
        j = j + 1
print(listvalues)


class Stack:
    def __init__(self):
        self.stack = []
    def push(self,val):
        self.stack.append(val)
    def pop(self):
        return self.stack.pop()
    def stringval(self,string):
        for i in string:
            self.push(i)
    def revstring(self):
        s = ''
        while self.stack!= []:
            s+=self.pop()
        return s
    def evaluatepostfix(self,string):
        self.stack = []
        for i in string:
            if i.isdigit():
                self.push(int(i))
 
            else:
                v1 = self.pop()
                v2 = self.pop()
                if i =='+':
                    self.push(v1+v2)
                if i =='-':
                    self.push(v2-v1)
                if i =='*':
                    self.push(v1*v2)
                if i =='/':
                    self.push(v2/v1)
 
        return int(self.pop())
    def dequeue(self):
        if self.stack == []:
            return 'stack is empty'
        temp = []
        value = None
        while self.stack != []:
            temp.append(self.pop())
        value = temp.pop()
        self.stack = temp[::-1]
        
        return value
    
#8. Reverse a string using a stack data structure
s = Stack()
s.stringval('stringreverse')
print(s.revstring())

#9. Evaluate a postfix expression using stack

print(s.evaluatepostfix('231*+9-'))

#10. Implement a queue using the stack data structure
s.push(10)
s.push(20)
s.push(30)
s.push(40)
s.push(60)
s.dequeue()
