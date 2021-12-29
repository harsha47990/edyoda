def sumvalues(lst):
    sum = 0
    for i in lst:
        sum+= i
    return sum
        
listvalues = list(map(int, input("enter comma seperated integers only ").split(',')))
print(sumvalues(listvalues))