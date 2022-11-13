
import matplotlib.pyplot as plt

# cloth no-unet step3 no-online 0.556
# cloth no-global step3 0.713
# cloth no-unet step4 0.503

x = [1, 2, 3, 4, 5]
y_1 = [0.589, 0.695, 0.754, 0.752, 0.758]
y_2 = [0.526, 0.586, 0.629, 0.624, 0.612]
# y_3 = [0.561, 0.725, 0.713, 0.1, 0.1]
y_4 = [0.516, 0.656, 0.652, 0.632, 0.672]
y_5 = [0.241, 0.211, 0.304, 0.185, 0.190]

print(x)
print(y_1)
print(y_2)

# 创建画布
plt.figure(figsize=(12, 9), dpi=400, linewidth=10)
plt.ylim(0.1, 0.85)
plt.xticks(range(1, 7, 1), fontsize=25)
plt.yticks(fontsize=25)

plt.plot(x, y_1, marker='*', markersize=15, color='orangered', label='Ours', lw=5)

plt.plot(x, y_2, marker='*', markersize=15, color='darkviolet', label='Ours w/o IST', lw=5)

# plt.plot(x, y_3, marker='*', color='orange', label='without global')

plt.plot(x, y_4, marker='*', markersize=15, color='steelblue', label='Ours w/o SC', lw=5)

plt.plot(x, y_5, marker='*', markersize=15, color='violet', label='Ours RandPick', lw=5)

plt.hlines(0.669, xmin=1, xmax=5, ls='-', lw=5, color='y', label='Ours only dist')

# 显示图例（使绘制生效）
plt.legend(fontsize=27)

# 横坐标名称
plt.xlabel('Step', fontsize=35)

# 纵坐标名称
plt.ylabel('Manipulation score', fontsize=35)

# 保存图片到本地
plt.savefig('ablation.png')




