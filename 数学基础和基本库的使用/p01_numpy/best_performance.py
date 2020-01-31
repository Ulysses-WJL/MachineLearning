#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Ulysses
@Date: 2019-10-30 14:55:44
@Description: 如何使numpy运行起来更迅速
@LastEditTime: 2019-10-30 17:20:23
'''
from functools import wraps
import time
import numpy as np
import pandas as pd


N = 9999

def time_count(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        time0 = time.time()
        func(*args, **kwargs)
        time1 = time.time()
        return time1 - time0
    return wrapper

# 1 array的结构和axis 行(C-type)/列(Fortran) 排列
#  Numpy 就是C的逻辑, 创建存储容器Array的时候是寻找内存上的一连串区域来存放
row_major = np.zeros((500, 500), order='C')  # C-type
col_major = np.zeros((500, 500), order='F')  # Fortran

@time_count
def f1(a):
    for _ in range(N):
        np.concatenate((a, a), axis=0)

@time_count
def f2(a):
    indices = np.random.randint(0, 500, size=10, dtype=np.int32)
    for _ in range(N):
        a[indices, :]

@time_count
def f3(a):
    indices = np.random.randint(0, 500, size=10, dtype=np.int32)
    for _ in range(N):
        a[:, indices]



# print("对row_major进行按行合并\n", f1(row_major))  # 31.620142221450806s

# print("对col_major进行按行合并\n", f1(col_major))  # 33.557344913482666s

# print('在行上操作', f2(row_major))  # 0.07812905311584473s
# print('在列上操作', f3(row_major))  # 0.2500126361846924s


# vstack 比 concatenate(...axis=0) 慢
@time_count
def f4(a):
    for _ in range(N):
        np.vstack((a, a))

# print("vstack: ", f4(row_major))  # 33.153279304504395s
# print("concatenate: ", f1(row_major))  # 32.45099639892578s

# 2 copy慢 view快

a = np.arange(1, 7).reshape((3, 2))
a_view = a[:2]
a_copy = a[:2].copy()

a_copy[1, 1] = 0
print(a)
"""
[[1 2]
 [3 4]
 [5 6]]
"""

a_view [1, 1] = 0
print(a)
"""
[[1 2]
 [3 0]
 [5 6]]
"""

@time_count
def f5(a):
    for _ in range(N):
        a *= 2

@time_count
def f6(a):
    for _ in range(N):
        a = 2 * a

# print("使用copy: ", f6(row_major))  # 13.0477933883667s
# print("使用view: ", f5(row_major))  # 1.6251144409179688s


# 矩阵展平 ravel 返回的是一个 view
@time_count
def f7(a):
    for _ in range(N):
        a.flatten()

@time_count
def f8(a):
    for _ in range(N):
        a.ravel()

# print("使用flatten\n", f7(row_major) / N)  #  0.0016611032288531588
# print("使用ravel\n", f8(row_major) / N)  #  1.5627039541112313e-06

# 选择数据时 使用view 而不是copy
# a_view1 = a[1:2, 3:6]    # 切片 slice
# a_view2 = a[:100]        # 同上
# a_view3 = a[::2]         # 跳步
# a_view4 = a.ravel()      # 上面提到了

# a_copy1 = a[[1,2], [0, 1]]   # 用 index 选
# a_copy2 = a[[True, True, False], [False, True]]  # 用 mask
# a_copy3 = a[[1,2], :]        # 虽然 1,2 的确连在一起了, 但是他们确实是 copy
# a_copy4 = a[a[1,:] != 0, :]  # fancy indexing
# a_copy5 = a[np.isnan(a), :]  # fancy indexing

# 使用take和compress 替代index/mask选数据
a = np.random.rand(10000, 10)
indices = np.random.randint(0, 10000, size=10)
mask = a[:, 0] < 0.5
N = 9999

@time_count
def f9(a):
    for _ in range(N):
        _ = np.take(a, indices, axis=0)

@time_count
def f10(a):
    for _ in range(N):
        _ = a[indices]

@time_count
def f11(a):
    for _ in range(N):
        _ = np.compress(mask, a, axis=0)

@time_count
def f12(a):
    for _ in range(N):
        _ = a[mask]

# print("使用take", f9(a)/N)  # 4.6871342483979845e-06
# print("使用index", f10(a)/N)  # 7.811874517835084e-06
# print("使用compress", f11(a)/N)  # 0.00014061383669800086
# print("使用mask", f12(a)/N)  # 0.000228106945750117

# 使用out参数
a = np.arange(1, 10)
b = np.arange(1, 10)

@time_count
def f13(a):
    for _ in range(N):
        # a = a + 1
        a = np.add(a, 1)

@time_count
def f14(a):
    for _ in range(N):
        # a += 1
        np.add(a, 1, out=a)

# print("未使用out参数:", f13(a)/N)
# print("使用out参数:", f14(b)/N)

# 给数据一个名字 代替pandas

a = np.zeros(3, dtype=[('foo', np.int32), ('bar', np.float64)])
b = pd.DataFrame(np.zeros((3, 2), dtype=np.int32), columns=['foo', 'bar'])
print(a['bar'])
# [0. 0. 0.]
print(b)
"""
   foo  bar
0    0    0
1    0    0
2    0    0
"""
@time_count
def f15(a):
    for _ in range(N):
        a['bar'] *= a['foo']

print('使用numpy: ', f15(a)/N)  # 4.688135706576029e-06

print('使用pandas: ', f15(b)/N)  # 0.0006688425619371629
