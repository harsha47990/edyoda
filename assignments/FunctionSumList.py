def sumvalues(lst):
    return sum(lst)
        
listvalues = list(map(int, input("enter comma seperated integers only ").split(',')))
print(sumvalues(listvalues))
