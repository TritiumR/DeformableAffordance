
import matplotlib.pyplot as plt

# Step1 online 0.441
# Step1 no-online 0.359
# Step2 online 0.513
# Step2 no-online 0.418
# Step3 online 0.501
# Step3 no-online 0.437
# Step3 only_gt 0.410
# Drq 0.154
# CURL 0.348
# PlaNet 0.236
# Wu 0.158
x = [1, 2, 3, 4, 5]
y_1 = [0.441, 0.513, 0.518, 0.536, 0.527]
y_2 = [0.359, 0.418, 0.437, 0.479, 0.394]
# y_3 = [0.406, 0.384, 0.1, 0.1]
y_4 = [0.363, 0.379, 0.430, 0.397, 0.443]
y_5 = [0.329, 0.302, 0.332, 0.334, 0.322]

print(x)
print(y_1)
print(y_2)

# 创建画布
plt.figure(figsize=(12, 9), dpi=400, linewidth=10)
plt.ylim(0.1, 0.60)
plt.xticks(range(1, 7, 1), fontsize=25)
plt.yticks(fontsize=25)

plt.plot(x, y_1, marker='*', markersize=15, color='orangered', label='Ours', lw=5)

plt.plot(x, y_2, marker='*', markersize=15, color='darkviolet', label='Ours w/o IST', lw=5)

# plt.plot(x, y_3, marker='*', color='orange', label='without global')

plt.plot(x, y_4, marker='*', markersize=15, color='steelblue', label='Ours w/o SC', lw=5)

plt.plot(x, y_5, marker='*', markersize=15, color='violet', label='Ours RandPick', lw=5)

plt.hlines(0.460, xmin=1, xmax=5, ls='-', lw=5, color='y', label='Ours only dist')

# plt.hlines(0.320, xmin=1, xmax=4, ls='-', lw=2, color='violet', label='random pick')

# 显示图例（使绘制生效）
plt.legend(fontsize=27)

# 横坐标名称
plt.xlabel('Step', fontsize=35)

# 纵坐标名称
plt.ylabel('Manipulation score', fontsize=35)

# 保存图片到本地
plt.savefig('ablation_rope.png')




