import os
root = '/root/softgym/visual/1110_14'
dirs = os.listdir(root)

total = 0.
i = 0

for fname in dirs:
    path = os.path.join(root, fname)
    print(fname)
    score = fname.split('-')[-1]
    score = float(score.split('.j')[0])
    if len(fname.split('-')) == 8:
        score = -score
    print(score)
    if score > 1:
        score = 1
    total += score
    i += 1

print(i)
total /= i
print(total)
