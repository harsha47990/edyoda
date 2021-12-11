a = 0
b = 1
nxt = 1
limit = 50
while(nxt < limit):
    print(nxt , end=' ')
    nxt = a+b
    a = b
    b = nxt
    