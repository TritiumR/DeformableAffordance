import os
root = '/root/softgym/test_video/rope-configuration-Aff_Critic-80001101-01-S-only_gt-400000-aff-300000-critic-step2-set_flat'
dirs = os.listdir(root)

total = 0.
i = 0

for fname in dirs:
    path = os.path.join(root, fname)
    print(fname)
    score = fname.split('-')[-1]
    score = float(score.split('.g')[0])
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
