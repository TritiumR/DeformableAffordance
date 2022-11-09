
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
# Wu
x = [1, 2, 3, 4]
y_1 = [0.441, 0.513, 0.518, 0.536]
y_2 = [0.359, 0.418, 0.437, 0.489]
y_3 = [0.406, 0.1, 0.1, 0.1]
y_4 = [0.363, 0.379, 0.1, 0.1]
y_5 = [0.329, 0.302, 0.342, 0.419]

print(x)
print(y_1)
print(y_2)

# 创建画布
plt.figure()
plt.ylim(0.1, 0.75)
plt.xticks(range(1, 7, 1))

plt.plot(x, y_1, marker='*', color='orangered', label='ours')

plt.plot(x, y_2, marker='*', color='darkviolet', label='without online')

plt.plot(x, y_3, marker='*', color='orange', label='without global')

plt.plot(x, y_4, marker='*', color='steelblue', label='without unet')

# plt.plot(x, y_5, marker='*', color='violet', label='random pick')

plt.hlines(0.410, xmin=1, xmax=5, ls='-', lw=2, color='y', label='only GT')

plt.hlines(0.320, xmin=1, xmax=5, ls='-', lw=2, color='violet', label='random pick')

# 显示图例（使绘制生效）
plt.legend()

# 横坐标名称
plt.xlabel('step')

# 纵坐标名称
plt.ylabel('normalized score')

# 保存图片到本地
plt.savefig('ablation_rope.png')




