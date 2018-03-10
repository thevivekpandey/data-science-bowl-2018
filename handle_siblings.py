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
        r = '\t'.join([str(x) for x in (i, len(src_pixels), len(target_pixels), intersection, union)])
        #print(r)
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i
    return max_iou, max_iou_index

def handle_siblings(in_base_name, out_file_name):
    mutations = ['90', '180', '270', 'fh', 'fv', 'fh_90', 'fv_90']
    all_mutations = ['0', '90', '180', '270', 'fh', 'fv', 'fh_90', 'fv_90']

    #This dict has key as mutation and value as another dict.
    #Inner dict has image as key and a list of sets as value. Each set in the list corresponds to a mask.
    print("Start reading the files")
    mutation_2_img_2_pixels = {}
    for m in all_mutations:
        print("Reading mutation", m)
        rle_file = in_base_name + '-mutation-' + m + '.csv'
        mutation_2_img_2_pixels[m] = my_utils.get_img_2_pixels_from_rle_file(rle_file)
    print("Ended reading the files")

    for image, masks in mutation_2_img_2_pixels['0'].items():
        #Each mask is a set
        for mask in masks:
            print(mask)
            for mutation in mutations:
                iou, index = get_index_and_iou_of_max_overlap(mask, mutation_2_img_2_pixels[mutation][image])
                print(mutation_2_img_2_pixels[mutation][image])
                print(iou, index)
                sys.exit(1)

if __name__ == '__main__':
    in_base_name = sys.argv[1]
    out_file_name = sys.argv[2]
    handle_siblings(in_base_name, out_file_name)
