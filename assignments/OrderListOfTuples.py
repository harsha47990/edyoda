lst = []
for i in range(int(input('enter no of items in list : '))):
    t = tuple(map(int,input('space seperated numbers for tuples: ').split()))
    lst.append(t)
print('INPUT:- ',lst) 

for i in range(len(lst)-1):
    for j in range(i,len(lst)):
        if(lst[i][len(lst[i])-1] > lst[j][len(lst[j])-1]):
            temp = lst[i]
            lst[i] = lst[j]
            lst[j] = temp

print('OUTPUT:- ', lst)