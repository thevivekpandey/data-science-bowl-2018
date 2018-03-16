import sys
import my_utils

#src_pixels is a set of pixels
#target_pixels_list is a list of set of pixels
#This function finds the index of the set in target_pixels_list which has max overlap with src_pixels, and also outputs
#the this overlap fraction
def get_index_and_iou_of_max_overlap(src_pixels, target_pixels_list):
    max_iou = 0
    max_iou_index = -1
    for i, target_pixels in enumerate(target_pixels_list):
        intersection = len(src_pixels.intersection(target_pixels))
        union = len(src_pixels.union(target_pixels))
        iou =  intersection * 1.0 / union
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i
    return max_iou, max_iou_index

def handle_siblings_work(in_base_name, out_file_name):
    mutations = ['B', 'C', 'D', 'E', 'F', 'G', 'H']
    all_mutations = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    #This dict has key as mutation and value as another dict.
    #Inner dict has image as key and a list of sets as value. Each set in the list corresponds to a mask.
    print("Start reading the files")
    mutation_2_img_2_pixels = {}
    for m in all_mutations:
        print("Reading mutation", m)
        rle_file = in_base_name + '-mutation-' + m + '.csv'
        mutation_2_img_2_pixels[m] = my_utils.get_img_2_pixels_from_rle_file(rle_file)
    print("Ended reading the files")

    output_rles = []
    output_test_ids = []
    for image, masks in mutation_2_img_2_pixels['A'].items():
        #Each mask is a set
        for mask in masks:
            num_brothers = 0
            union_mask = set()
            iou_product = 1
            union_mask = union_mask.union(mask)
            for mutation in mutations:
                iou, index = get_index_and_iou_of_max_overlap(mask, mutation_2_img_2_pixels[mutation][image])
                matching_mask = mutation_2_img_2_pixels[mutation][image][index]
                union_mask = union_mask.union(matching_mask)
                iou_product *= iou
                if iou > 0.42:
                    num_brothers += 1
            #if num_brothers == 7:
            if pow(iou_product, 1/7) > 0.42:
                output_test_ids.append(image)
                output_rles.append(my_utils.get_rle(sorted(list(union_mask))))
    return output_test_ids, output_rles

def handle_siblings(in_base_name, out_file_name):
    output_test_ids, output_rles = handle_siblings_work(in_base_name, out_file_name)
    my_utils.write_new_rle_file(out_file_name, output_test_ids, output_rles)
    
