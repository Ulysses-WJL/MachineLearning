#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-31 09:24:00
@Description: numpy dtype 数据类型对象
@LastEditTime: 2019-10-31 10:24:12
'''
import numpy as np
"""
数据类型对象描述了对应于数组的固定内存块的解释，取决于以下方面：
  - 数据类型（整数、浮点或者 Python 对象）
  - 数据大小
  - 字节序（小端或大端）
  - 在结构化类型的情况下，字段的名称，每个字段的数据类型，和每个字段占用
    的内存块部分。
  - 如果数据类型是子序列，它的形状和数据类型。
字节顺序取决于数据类型的前缀 < 或 > 。 < 意味着编码是小端（最小有效字节
存储在最小地址中）。 > 意味着编码是大端（最大有效字节存储在最小地址
中）。

dtype 可由一下语法构造：
numpy.dtype(object, align, copy)
参数为：
- Object ：被转换为数据类型的对象。
- Align ：如果为 true ，则向字段添加间隔，使其类似 C 的结构体。
- Copy ? 生成 dtype 对象的新副本，如果为 flase ，结果是内建数据类型
对象的引用。

每个内建类型都有一个唯一定义它的字符代码：
'b' ：布尔值
'i' ：符号整数
'u' ：无符号整数
'f' ：浮点
'c' ：复数浮点
'm' ：时间间隔
'M' ：日期时间
'O' ：Python 对象
'S', 'a' ：字节串
'U' ：Unicode
'V' ：原始数据（ void ）
"""

# 使用数组标量类型
dt1 = np.dtype(np.int32)
print('dt1: ', dt1)  # dt1:  int32

dt2 = np.dtype('<i4')  # int8 16 32 64 可以使用字符串 i1 i2, i3, i4 替代
print('dt2: ', dt2)  # dt2:  int32

# 创建结构化数据类型
dt_age = np.dtype([('age', '<u1')])  # 字段age名称和相应的标量数据类型uint8
print("dt_age: ", dt_age)  # dt_age:  [('age', 'u1')]

# 使用dt_age类型创建矩阵
array = np.array([(1, ), (2, ), (3, )], dtype=dt_age)
print(array)  # [(1,) (2,) (3,)]
print(array['age'])  # 访问age列 [1 2 3]

# 创建student类型 数据对象
dt_student = np.dtype([('name', 'S20'), ('age', '<u1'), ('marks', 'f4')])
print("dt_student", dt_student)
# dt_student [('name', 'S20'), ('age', 'u1'), ('marks', '<f4')]

students = np.array([('Jack', 16, 30.1), ('Rose', 17, 11.2)], dtype=dt_student)
print('students:',  students)
# [(b'Jack', 16, 30.1) (b'Rose', 17, 11.2)]

print(students.itemsize)  # 每个元素的字节单位长度  20+1+4
# 25
print(students.flags)
"""
  C_CONTIGUOUS : True  数组位于单一的、C 风格的连续区段内
  F_CONTIGUOUS : True  数组位于单一的、Fortran 风格的连续区段内
  OWNDATA : True  数组的内存从其它对象处借用
  WRITEABLE : True  数据区域可写入。 将它设置为 flase 会锁定数据，使其只读
  ALIGNED : True  数据和任何元素会为硬件适当对齐
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False 这个数组是另一数组的副本。当这个数组释放时，
源数组会由这个数组中的元素更新
"""
