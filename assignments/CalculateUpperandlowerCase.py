def calUpandLow(s):
    l = 0
    u = 0
    for c in s:
        if(c.isupper()):
            u+=1
        if(c.islower()):
            l+=1
    return l,u

l,u = calUpandLow(input("enter the string : "))
print("No. of Upper case characters :",u)
print("No. of Lower case Characters :",l)