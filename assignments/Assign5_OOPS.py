class Power():
    def pow(self,x,n):
        return x**n
p = Power()
x, n = list(map(int,input("enter space seperated X and N value : ").split(' ')))
print(p.pow(x,n))
