import os
root = '/root/softgym/visual/1117_04'
dirs = os.listdir(root)

total = 0.
i = 0

for fname in dirs:
    path = os.path.join(root, fname)
    print(fname)
    score = float(fname.split('-')[-2])
    # score = float(score.split('.j')[0])
    if len(fname.split('-')) == 7:
        score = -score
    print(score)
    if score > 1:
        score = 1
    total += score
    i += 1

print(i)
total /= i
print(total)
