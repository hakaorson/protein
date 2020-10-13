'''
10 30
100 200
150 50
60 20
50 70
120 20
#
'''

datas = []
while(True):
    tempdata = input()
    if tempdata == '#':
        break
    datas.append(list(map(int, tempdata.split(' '))))
target = len(datas)//2
datas.insert(0, [0, 0])
matrix = [[0 for j in range(target+1)]for i in range(len(datas))]
for i in range(1, len(datas)):
    for j in range(target+1):
        if j > i or i-j > target:
            continue
        if j > 0:
            matrix[i][j] = max(datas[i][0]+matrix[i-1][j-1], matrix[i][j])
        if i > j:
            matrix[i][j] = max(datas[i][1] + matrix[i-1][j], matrix[i][j])
print(matrix[-1][-1])
