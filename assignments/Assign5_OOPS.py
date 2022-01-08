class Power():
    def __init__(self,x,n):
        self.x = x
        self.n = n
    def pow(self):
        return self.x**self.n

x, n = list(map(int,input("enter space seperated X and N value : ").split(' ')))
p = Power(x,n)
print(p.pow())
