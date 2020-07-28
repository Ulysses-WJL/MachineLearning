'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-07-22 13:57:39
@LastEditors: your name
@Description: 
'''
import numpy as np 
import matplotlib.pyplot as plt
dataLength = 5
label = np.array(['内部一致性','外部公平性','薪资满意度','福利满意度','晋升满意度'])
data = np.array([0.65,0.67,0.87,0.98,0.75])
angles = np.linspace(0, 2*np.pi, dataLength, endpoint=False)
data = np.concatenate((data,[data[0]]))
angles = np.concatenate((angles,[angles[0]]))
fig = plt.figure()
ax = fig.add_subplot(111,polar=True)
ax.set_thetagrids(angles * 180/np.pi, label, fontproperties="SimHei")
ax.plot(angles,data,'bo-',linewidth=2)
ax.set_rlim(0,1)
ax.fill(angles, data, facecolor='r', alpha=0.25)
#ax.set_theta_zero_location('NW')
#ax.set_rlabel_position('55')
ax.set_title("统计结果",fontproperties="SimHei",fontsize=16) #设置标题
plt.show()