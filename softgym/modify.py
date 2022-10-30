import os
root = '/root/softgym/test_video/cloth-flatten-Aff_Critic-80001030-17-step4-online-continue-critic-set_flat-10'
dirs = os.listdir(root)
total = 0.
i = 0

for fname in dirs:
    path = os.path.join(root, fname)
    print(fname)
    score = fname.split('-')[-1]
    score = float(score.split('.g')[0])
    if len(fname.split('-')) == 5:
        score = -score
    print(score)
    if score > 1:
        score = 1
    total += score
    i += 1

print(i)
total /= i
print(total)
