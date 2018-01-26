import os
import sys

model = sys.argv[1]
img_2_count_a, img_2_count_b = {}, {}
images = os.listdir('../input/stage1_train')
for image in images:
    img_2_count_a[image] = 0
    img_2_count_b[image] = 0

f = open('../input/stage1_train_labels/stage1_train_labels.csv')
for line in f:
    img, rle = line.strip().split(',')
    if img == 'ImageId':
        continue
    img_2_count_a[img] += 1
f.close()

f = open('models/sub-train-' + model + '.csv')
for line in f:
    img, rle = line.strip().split(',')
    if img == 'ImageId':
        continue
    img_2_count_b[img] += 1
f.close()

total_deviation, count = 0, 0
rle_a_count, rle_b_count = 0, 0
for image in images:
    rle_a_count += img_2_count_a[image]
    rle_b_count += img_2_count_b[image]
    total_deviation += abs(img_2_count_a[image] - img_2_count_b[image]) * 1.0 / img_2_count_a[image]
    print img_2_count_a[image], img_2_count_b[image]
    count += 1

print 'correct total count =', rle_a_count
print 'model total count =', rle_b_count
print 'average deviation =', total_deviation / count
