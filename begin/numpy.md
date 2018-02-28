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