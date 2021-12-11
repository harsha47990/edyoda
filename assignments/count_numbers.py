inputvalue = input() #enter comma seperated intergers
lst = inputvalue.split(',')
even = 0
odd = 0
for i in lst:
    val = int(i)
    if(val%2==0):
        even+=1
    else:
        odd+=1
print("Number of even numbers :", even)
print("Number of odd numbers :", odd)