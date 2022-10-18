import os
root = '/root/softgym/test_video/cloth-flatten-Aff_Critic-90001018-15-step-2-true_best-30:50'
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

total /= i
print(total)
