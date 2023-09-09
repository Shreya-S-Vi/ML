import csv
a = []

with open('play_tennis.csv','r') as dataset:
    for i in csv.reader(dataset):
        a.append(i)

a.pop(0)

print(a)

case = []
no_attributes = len(a[0]) - 2

for i in range(0,no_attributes):
    x = input("Attribute " + str(i+1))
    case.append(x)

print("The given case is : " + str(case))
positive = 0
negative = 0

# finding positive and negative instances

for i in range(0,len(a)):
    if(a[i][len(a[i]) - 1] == "Yes"):
        positive = positive + 1 
    if(a[i][len(a[i]) - 1] == "No"):
        negative = negative + 1
        
print(positive)
print(negative)
#finding positive and negative probabilities
prob_pos = positive/len(a)
prob_neg = negative/len(a)

print(prob_pos)
print(prob_neg)
NB_pos = prob_pos
NB_neg = prob_neg

j = 1
count_pos = 0
count_neg = 0

for i in range(1,no_attributes+1):
    count_pos = 0
    count_neg = 0
    for j in range(0,len(a)):
        if case[i-1] in a[j]:
            if(a[j][len(a[0])-1] == "Yes"):
                count_pos = count_pos + 1
            if(a[j][len(a[0])-1] == "No"):
                count_neg = count_neg + 1
    # print(count_pos,count_neg)
    x = count_pos/positive
    y = count_neg/negative
    
    NB_pos = NB_pos * x
    NB_neg = NB_neg * y

print(NB_pos)
print(NB_neg)
if(NB_pos > NB_neg) : 
    print(str(case) + " corresponds to YES")
else:
    print(str(case) + "corresponds to NO")
