'''
对Python中的序列进行复习
序列 包括列表 元组  集合 字典 和字符串
'''

# 列表

# test_a = ["hello",'world']
# str_a = str(test_a)
# print(str_a)
#
# for i, j in enumerate(test_a):
###############################################  一维列表转二维列表 #####################################
# 	print(f"in {i} value is {j}")
import math
mylist = [1,2,3,4,5,6,7,8,9]
temp = []
if len(mylist)%int(math.sqrt(len(mylist)))==0:

	for idx in range (0,len(mylist),int(math.sqrt(len(mylist)))):
		temp.append([mylist[i] for i in range(idx,idx+int(math.sqrt(len(mylist))))])


print(temp)

