### numpy 基础

* 引入: imprt numpy

* numpy 的array 与python的list异同


  * ```
    python list

    1.内容可变的list
    L = [i for i in range[10]]
    L #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    下面两个操作都正常执行
    L[5] = "Machine Learning"
    L[5] = 100

    以上Python的List不要求存储同样的类型，带来效率问题

    2.内容不可变的list
    import array
    arr = array.array('i', [i for i in range(10)])
    arr #array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    下面两个操作都无法正常执行
    arr[5] = "Machine Learning"
    arr[5] = 5.0

    array的缺点是没有将数据当做向量或者矩阵，不支持基本运算。
    ```

  * ```
    numpy.array  

    np = npmpy
    nparr = np.array([i for i in range(10)])
    nparr #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    npmpy array 可以改变赋值给某个元素相同类型的数据,如
    可以执行 nparr[5] = 100
    不允许执行 nparr[5] = "Machine Learning"

    //打印当前数组的元素类型
    nparr.dtype

    此外数值类型的数据,numpy会进行一次兼容 像int64 float64之间
    ```



#### 创建numpy数组的方法

```
1. 建立普通数组
nparr = np.array([i for i in range(10)])
2. 建立全0数组或者矩阵
   np.zeros(10)
   np.zeros(4,dtype=float)
   np.zeros((3,4))
   np.zeros(shape=(3,4), dtype=int)
3. 建立全1数组或者矩阵
   np.ones(10)
   np.ones((3,5))
   np.ones(shape=(3,5), dtype=int)
4. 建立指定数据的数组或者矩阵
   np.full((3,5), 66)
   np.full(shape=(3,5), fill_value=626)
  
5. 在一定范围内, 指定步长建立数组
   [i for i in range(2,10, 3)]
   np.arange(2,10,3)
   np.arange(2,10,0.3)
   np.arange(0, 10)
   np.arange(10)
   [i for i in range(2,10, 0.3)]  #原始的python建立的方式不允许有非整数的步长,但是numpy可以
6.建立等差数列的数组
	np.linspace(1,10,10)
	# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
	
7.随机数
	7.1 randint
		np.random.randint(0,10) #5 [a, b) 后面是一个闭区间
        np.random.randint(0,10,2) #array([7, 3])
        np.random.randint(0, 10, size=10) #array([7, 9, 9, 0, 1, 9, 5, 6, 2, 7])
        np.random.randint(0, 10, size=(4,6))#创建随机矩阵

	7.2 seed
		指定随机数种子,这样出来的随机数组每次都是一样的
		np.random.seed(666)
		np.random.randint(0, 10, size=(3,5))
		
	7.3 random 
		返回0 -1.0的随机浮点数
		np.random.random() 
		np.random.random((3,5))
		
	7.4 normal
        均值μ和标准差σ,标准差能反映一个数据集的离散程度。平均数相同的，标准差未必相同。
        方差 s=[(x1-x)^2 +(x2-x)^2 +.(xn-x)^2]/n 　　(x为平均数) 
        标准差=方差的算术平方根
        标准正态分布又称为u分布，是以0为均数、以1为标准差的正态分布，记为N（0，1）。
        
        np.random.normal(0,1,(3,4))
        np.random.normal(0,1,(3,4))
        numpy.random.normal(loc=0.0, scale=1.0, size=None)  

        参数的意义为：
          loc:float
          概率分布的均值，对应着整个分布的中心center
          
          scale:float
          概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高

          size:int or tuple of ints
          输出的shape，默认为None，只输出一个值

          我们更经常会用到np.random.randn(size)所谓标准正太分布（μ=0, σ=1），对应于np.random.normal(loc=0, scale=1, size)
```





### numpy array 数组操作

```
import numpy as np
np.random.seed(0)

x = np.arange(10)
x 
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

X = np.arange(15).reshape((3, 5))
X
#array([[ 0,  1,  2,  3,  4],
#       [ 5,  6,  7,  8,  9],
#       [10, 11, 12, 13, 14]])



reshape :
改变数组的形状,维度,有些不能整除的分割将报错,如10个元素要分成3行就无法整除
-1，表示自动计算此维度ruz.reshape(1,-1)


#ndim 维度数量 
x.ndim #1
X.ndim #2,两个维度

#具体形状,对于一个数组，其shape属性的长度（length）也既是它的ndim.
x.shape #(10,)
X.shape #(3, 5)

x.size  #10
X.size #15


#读取数据
x[1]    #一维数据
X[2,3]  #二维数据访问


切片访问
x[a:b:c]  #返回从索引a开始到b结束,步长为c的数组,abc均可为空
X[:2, :3] #矩阵也一样 几行几列
切片访问到的数据,数值修改后,会修改原数组的数据,所以可以调用copy()方法进行复制
subX = X[:2, :3].copy()


下面这样切就可以把这个矩阵=切成]一个向量数组:
y = array([[ 3],
       [ 7],
       [11],
       [15]])

y[:, 0]
array([ 3,  7, 11, 15])
```



Numpy array 合并连接

```
import numpy as np
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])


1.concatenate
合并一维数组
np.concatenate([x, y]) #array([1, 2, 3, 3, 2, 1])
z = np.array([666, 666, 666])
np.concatenate([x, y, z]) #array([  1,   2,   3,   3,   2,   1, 666, 666, 666])



合并二维数组
A = np.array([[1, 2, 3],
              [4, 5, 6]])
np.concatenate([A, A])
结果
array([[1, 2, 3],
       [4, 5, 6],
       [1, 2, 3],
       [4, 5, 6]])


np.concatenate([A, A], axis=1)
array([[1, 2, 3, 1, 2, 3],
       [4, 5, 6, 4, 5, 6]])
#0轴表示行，1轴表示列，axis=1表示对列进行操作，一列一列的就是行的方向,对列操作行不变

concatenate不支持维度不一样的合并 A Z就不能直接合并
方式1
把z转成一个二维的矩阵 np.concatenate([A, z.reshape(1, -1)])
z.reshape(1, -1) 把z转成了一个二维的矩阵就可以
方式2
np.vstack([A, z]) 
vstack 在垂直方向上,合并连接数据,列不变

#水平方向上的 行不变
B = np.full((2,2), 100)
np.hstack([A, B])
```



Numpy array 分割

```
针对普通向量
x = np.arange(10)
x1, x2, x3 = np.split(x, [3, 7])   [3,7]是分割点,在3,7 这两个索引进行分割

针对矩阵
A = np.arange(16).reshape((4, 4))
A1, A2 = np.split(A, [2])   #axis默认=0针对行的切割,列数是不会变的
A1, A2 = np.split(A, [2], axis=1)  #列的切割 行数是不变的
```



### numpy运算

```
不同于python numpy的 + - * / ** //运算,是针对每一个矩阵中的元素
对于两个矩阵 A B  + - * /也是矩阵每个元素进行计算,而不是矩阵的计算
numpy.dot 矩阵的乘法结果
numpy.T 矩阵的逆运算

矩阵与向量的运算

1.tile方法 针对一个向量在某个方向上 重复的次数

v = np.array([1,2])
np.tile(v,[1,2])  #array([[1, 2, 1, 2]])  行重复1次,列重复两次
np.tile(v,[2,1])  #array([[1, 2],		  行重复两次,重复一次
                  #       [1, 2]])
                  
                  
v*A      #array([[0, 2],
         #[2, 6]])
v.dot(A) #array([4, 7])  这是正常的矩阵的运算
A.dot(v) #array([2, 8])  这是numpy帮我们 处理过的 矩阵的计算



矩阵的逆
invA = np.linalg.inv(A)
```



### 聚合操作

```
import numpy as np

L = np.random.random(100)
sum(L)    #47.44345492788211     python
np.sum(L) #53.73434820387867     numpy

上面的这两个求和运算主要体现在效率上 

np.min(xxx)
np.max(xxx)
xxx.max()
xxx.min()


np.sum(X, axis=1)



```





### arg 索引操作

np.argmax(x)

np.argmin(x)





### fancy indexing

```
x[3:9:2] 我们可以这样针对一个数组进行切片操作,但是对于不是等步长的,我们要怎么取,那么可以使用fancy indexing

x[[3,5,8]] 这样就可以用索引的方式去取数据

此外还可以传一个二维的数据到一维数组,返回一个二维的矩阵数据


fancy indexing在矩阵中使用
X = x.reshape(4, -1)
X
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
对于这样一个矩阵,我们可以这样取数据
row=[0,1,2]
col=[1,2,3]
x[row, col] = array([ 1,  6, 11])   	前三行与第123列相加
X[0, col] = array([1, 2, 3]) 			第一行的 123列数据
X[:2, col] = array([[1, 2, 3],  		针对前两行 分别取123列的数据
       				[5, 6, 7]])
其实就是针对cow描述的行与col描述的列去取数据



数据比较

x = array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])

x '< > <= >= == != '3
x 与3的这些比较 均返回一个 布尔值数组,存储每一个数据与3的比较结果

此外还可以这样进行比较
2 * x == 24 - 4 * x 这样也是得到一个数组每一个元素在等式左右的计算结果是不是一样 的一个数组

二维矩阵也一样可以进行比较




一些常用函数
np.count_nonzero( x <= 3) #4  进去的布尔数组中有多少个非0元素 4个true 其他都是false
np.sum(x <= 3)  #传进去的数组中有多少个<3的元素
np.any(x == 0)  # True是否有任何元素 满足条件  '==0' 
np.all(x > 0)   #False 是不是全部的数据都是 大于0 的
np.sum(X % 2 == 0, axis=0)    #array([4, 0, 4, 0]) 在列的方向上 偶数的个数
np.sum((x > 3) & (x < 10))    # >3 <10 的数据个数 注意是 & 不是&& 因为是针对矩阵做与的操作
```

