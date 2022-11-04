
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y_1 = [0.589, 0.695, 0.788, 0.758]
y_2 = [0.526, 0.586, 0.629, 0.624]
y_3 = [0.561, 0.475, 0.575, 0.585]
y_4 = [0.516, 0.570, 0.600, 0.630]
y_5 = [0.241, 0.211, 0.304, 0.185]

print(x)
print(y_1)
print(y_2)

# 创建画布
plt.figure()
plt.ylim(0.1, 0.85)
plt.xticks(range(1, 7, 1))

plt.plot(x, y_1, marker='*', color='orangered', label='ours')

plt.plot(x, y_2, marker='*', color='darkviolet', label='without online')

plt.plot(x, y_3, marker='*', color='orange', label='without global')

plt.plot(x, y_4, marker='*', color='steelblue', label='without unet')

plt.plot(x, y_5, marker='*', color='violet', label='random pick')

plt.hlines(0.629, xmin=1, xmax=4, ls='-', lw=2, color='y', label='only GT')

# 显示图例（使绘制生效）
plt.legend()

# 横坐标名称
plt.xlabel('step')

# 纵坐标名称
plt.ylabel('normalized score')

# 保存图片到本地
plt.savefig('ablation.png')




