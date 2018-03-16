import sys
import pandas as pd
import my_utils

        
def check_subsumed(arr1, arr2):
    set1 = set(arr1)
    set2 = set(arr2)
    if len(set1.intersection(set2)) > 0.9* len(set1):
        print(len(set1.intersection(set2)), 'vs', len(set1))
        return True
    else:
        return False

def remove_extraneous_masks(input_file_name, output_file_name):
    img_2_pixels = my_utils.get_img_2_pixels_from_rle_file(input_file_name)
    test_ids, rles = [], []
    seen, proceeded = 0, 0
    #Iterate over images
    for img, pixel_array in img_2_pixels.items():
        masks_to_be_deleted = []
        #Iterate over each mask of given image
        for i in range(len(pixel_array)):
            seen += 1
            to_be_deleted = False
            # Find if it matches with any other mask
            for j in range(len(pixel_array)):
                # If j is already deleted, you cannot 
                # delete someone because of j
                if i == j or j in masks_to_be_deleted:
                    continue
                # Check if i is subsumed by j
                if check_subsumed(pixel_array[i], pixel_array[j]):
                    print('deleted ', i, 'because of ', j, 'from ', img[0:10])
                    masks_to_be_deleted.append(i)
                    to_be_deleted = True
                    break
            if not to_be_deleted:
                rles.append(my_utils.get_rle(sorted(list(pixel_array[i]))))
                test_ids.append(img)
    my_utils.write_new_rle_file(output_file_name, test_ids, rles)
