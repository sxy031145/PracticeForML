data = 'Hello, World!'
print(data[1:5])
print(len(data))

a, b, c, d = 8, False, 'He', None
print(b, a, d)

val = 2;
if val == 5:
	print("yingyingying")
elif val > 3:
	print("shutup")
else:
	print("Nice!")

for i in range(10):
	print("Nice Buddy %d" % i)
i = 0
while i < 10:
	print("SakurajimaMai %d" % i)
	i = i + 1

#元组 只读数据类型，初始化后不能重新幅值
now = (9, 9, 6)
print(now)
print(now[2])
#列表 【】定义，可以赋值
now2 = [3, 6, 8]
print(now)
now2.append(4)
print(now2)
print(now2[2])
now2[1] = 7
print(now2)
for i in now2:
	print(i)
#字典 键值对(key, value)可以存储任意类型对象
dic = {'a': 9.96, 'b' : 'Miku', 'c' : False}
print('like... %.2f' % dic['a'])
dic['e'] = 7.78
print('Keys... %s \nValues... %s' % (dic.keys(), dic.values()))
for key in dic:
	print(dic[key])
dic.pop('b')
print(dic)
dic.clear()
print(dic)

def getmux(x, y) :
	return x*y
res = getmux(5, 6)
print(res)
