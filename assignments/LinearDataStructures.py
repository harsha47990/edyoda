#question 1

lst = list(map(int,input("enter space seperated integers").split()))
sumvalue = int(input('enter sum value'))
for i in range(len(lst)):
    for j in range(i,len(lst)):
        if lst[i] + lst[j] == sumvalue:
            print('({0},{1})'.format(lst[i],lst[j]))
            
            
#question 2
ary = [1,2,3,4,5,6]
n = len(ary)
for i in range(n//2):
    temp = ary[i]
    ary[i] = ary[n-1-i]
    ary[n-1-i] = temp
print(ary)


#question 3
s1 = "check"
s2 = 'kcehc'
print(s1[::-1] == s2)

#question 4
from collections import Counter
a = 'aabbbcccddd'
b = Counter(a)
min = 0
for k in b:
    if min == 0:
        min = b[k]
    if b[k] < min:
        min = b[k]
for i in a:
    if b[i] == min:
        print(i)
        break

#question 5
def hanoi(n , s, d, a):
    if n==1:
        print ("Move disk 1 from source",s,"to destination",d)
        return
    hanoi(n-1, s, a, d)
    print ("Move disk",n,"from source",s,"to destination",d)
    hanoi(n-1, a, d, s)

n = 4
hanoi(n,'A','B','C')

#question 6
def PostfixToPrefix(postfix):
    postfix = postfix.replace(" ","")
    s = []  
    for i in range(len(postfix)): 
        if postfix[i] not in '+*/-':
            s.append(postfix[i])  
        else:  
            op1 = s.pop()  
            op2 = s.pop()  
            expression = postfix[i] + op2 + op1  
            s.append(expression)  
    return s.pop()  
print(PostfixToPrefix('AB+CD-*'))

#question 7
def prefixToInfix(prefix):
    stack = []
    i = len(prefix) - 1
    while i >= 0:
        if prefix[i] not in "*+-/^()":
            stack.append(prefix[i])
            i -= 1
        else:
            str = "(" + stack.pop() + prefix[i] + stack.pop() + ")"
            stack.append(str)
            i -= 1
    return stack.pop()


strn = "*-A/BC-/AKL"
print(prefixToInfix(strn))

#question 8
from collections import Counter
s = '((A-(B/C))*((A/K)-L))'
l = []
for i in s:
    if i in '(){}[]':
        if i in '({[':
            l.append(i)
        else:
            try:
                popv = l.pop()
                if popv == '(' and i == ')':
                    pass
                elif popv == '[' and i == ']':
                    pass
                elif popv == '{' and i == '}':
                    pass
                else:
                    print('bracket missing')
            except:
                print('bracket missing')
if l != []:
    print('bracket missing')
else:
    print('expression is valid')
  
#question 9
stack = list(map(int,input("enter space seperated integeres").split()))

def Reverse(s): 
    if s == []:
        pass
    else:
        popped = s.pop()
        Reverse(s)
        s.insert(0,popped)
Reverse(stack)
print(stack)

#question 10

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        
    def __str__(self):
        return "Node({})".format(self.value)


class Stack:
    def __init__(self):
        self.top = None
        self.count = 0
        self.minimum = None
        
    def __str__(self):
        temp = self.top
        out = []
        while temp:
            out.append(str(temp.value))
            temp = temp.next
        out = '\n'.join(out)
    

    def getMin(self):
        if self.top is None:
            return "Stack is empty"
        else:
            print("Minimum Element in the stack is: {}" .format(self.minimum))


    def isEmpty(self):
        if self.top == None:
            return True
        else:
            return False


    def push(self,value):
        if self.top is None:
            self.top = Node(value)
            self.minimum = value

        elif value < self.minimum:
            temp = (2 * value) - self.minimum
            new_node = Node(temp)
            new_node.next = self.top
            self.top = new_node
            self.minimum = value
        else:
            new_node = Node(value)
            new_node.next = self.top
            self.top = new_node


    def pop(self):
        if self.top is None:
            print( "Stack is empty")
        else:
            removedNode = self.top.value
            self.top = self.top.next
            if removedNode < self.minimum:
                print ("Top Most Element Removed :{} " .format(self.minimum))
                self.minimum = ( ( 2 * self.minimum ) - removedNode )
            else:
                print ("Top Most Element Removed : {}" .format(removedNode))


stack = Stack()

stack.push(3)
stack.push(5)
stack.push(1)
stack.push(2)
stack.getMin()
