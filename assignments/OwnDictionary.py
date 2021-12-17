i = 0
while True:
    if('a'==chr(i)):
        break
    i+=1

dic = dict()
while True:
    dic[chr(i)] = i
    if(chr(i)=='z'):
        break
    i+=1
print(dic)
        