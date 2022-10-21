import os
root = '/root/softgym/test_video/cloth-flatten-Aff_Critic-80001021-11-only_gt-no-online-300000-step2-set_flat-10'
dirs = os.listdir(root)
total = 0.
i = 0

for fname in dirs:
    path = os.path.join(root, fname)
    print(fname)
    score = fname.split('-')[-1]
    score = float(score.split('.g')[0])
    print(score)
    if score > 1:
        score = 1
    total += score
    i += 1

print(i)
total /= i
print(total)
