import csv
a = []
with open('enjoysport.csv','r') as csvfile:
    for i in csv.reader(csvfile):
        a.append(i)
num_attribute = len(a[0]) - 1
hypothesis = ['0']*num_attribute
print("Initial Hypothesis, H0 : ")
print(hypothesis)

for i in range(0,len(a)):
    if (a[i][num_attribute] == 'yes'):
        print("\nInstance ", i+1, "is", a[i], "is +ve")

        for j in range(0,num_attribute):
            if (hypothesis[j] == '0' or hypothesis[j] == a[i][j]):
                hypothesis[j] = a[i][j]
            else :
                hypothesis[j] = '?'
        
        print("Hypothesis " , i+1 , " : ", hypothesis)
        print()
    elif(a[i][num_attribute] == 'no'):
        print("\nInstance ", i+1, "is", a[i], "is -ve ")

print()
print("The maximally specific hypothesis is : ")
print(hypothesis)
