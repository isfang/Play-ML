Conda 快捷键



###Notebook 快捷键

1. 运行: shift + enter
2. 快速添加cell 
   1. 在当前单元格上面添加: A
   2. 在当前单元格下面添加: B
3. 快速改变单元A格属性（Y，M）



### 一些常用命令

* 运行脚本 %run path/xx.py 

* 加载模块 mymodule文件夹下有个FirstML.py文件里有个xx方法

  *  Import  mymodule.FirstML   这样使用xx方法 :mymodule.FirstML.xx(1)      
  * from mymodule import FirstML  这样使用xx方法 FirstML.xx(1)

* 打印时间

  * %timeit 会loop多次给出一个多次执行的结果,总的执行次数控制在1000000我猜的..呵呵

    * ```
      %timeit L = [i**2 for i in range(1000000)]
      ```

    * ```
      //  %% 为区域命令符
      %%timeit
      L = []
      for n in range(1000):
          L.append(n ** 2)
      ```

  * 不像%timeit那样会多次执行的%time

    * ```
      %time L = [i**2 for i in range(1000)]
      ```

    * ```
      %%time
      L = []
      for n in range(1000):
          L.append(n ** 2)
      ```

* 其他命令

  * 罗列某个命令 %lsmagic
  * 查看某个命令 %run? 