import sys
import numpy as np
import my_utils
from scipy import ndimage

def run_length_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1: run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def post_process(mask):
    num_masks = mask.shape[2]
    cgs, sizes = [], []
    pixel_2_masks = {}
    pixel_2_closest_mask = {}
    
    width, height = mask.shape[0], mask.shape[1]
    # Put 1 pixel boundary around each mask
    
    xs, ys, zs = np.where(mask == 1)
    num_ones = len(xs)
    for i in range(num_ones):
        x, y, z = xs[i], ys[i], zs[i]
        if x - 1 >= 0     and mask[x-1,y,z] == 0: mask[x-1,y,z] = 2
        if x + 1 < width  and mask[x+1,y,z] == 0: mask[x+1,y,z] = 2
        if y - 1 >= 0     and mask[x,y-1,z] == 0: mask[x,y-1,z] = 2
        if y + 1 < height and mask[x,y+1,z] == 0: mask[x,y+1,z] = 2
    
    # Convert all 2s to 1
    mask[mask == 2] = 1
        
    for i in range(num_masks):
        mask_ = mask[:,:,i:i+1]
        mask_ = mask_.reshape(mask.shape[0], mask.shape[1])

        cg = ndimage.measurements.center_of_mass(mask_)
        cgs.append((cg[0], cg[1]))
        sizes.append(np.sum(mask_))

    # Find all masks for each pixel        
    for i in range(num_masks):
        mask_ = mask[:,:,i:i+1]
        mask_ = mask_.reshape(mask.shape[0], mask.shape[1])

        xs, ys = np.where(mask_==1)
        for x, y in zip(xs, ys):
            pixel_2_masks.setdefault((x, y), []).append(i)

    # Find closest mask for each pixel
    for (x, y), ms in pixel_2_masks.items():
        # ms is the array of mask ids, starting from 0
        min_d = my_utils.distance((x, y), cgs[ms[0]])
        min_idx = ms[0]

        for j in range(1, len(ms)):
            d = my_utils.distance((x, y), cgs[ms[j]])
            if d < min_d:
                min_d = d
                min_idx = ms[j]
        pixel_2_closest_mask[(x, y)] = min_idx

    # Create pseudo mask
    pseudo_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
    for (x, y), m in pixel_2_closest_mask.items():
        pseudo_mask[x, y] = m + 1

    # Create output set of masks
    output_masks = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]), dtype=np.bool)
    for i in range(mask.shape[2]):
        output_masks[:,:,i:i+1]=pseudo_mask[:,:].reshape((mask.shape[0], mask.shape[1],1)) == (i+1)

    return output_masks

#Read rles from in_file_name, handle overlap and write to out_file_name
def handle_overlap(in_file_name, out_file_name):
    image_names = my_utils.get_image_names_from_rle_file(in_file_name)
    sizes = my_utils.get_test_image_sizes(image_names)

    # In following two dicts, key is image name
    img_2_pixels = my_utils.get_img_2_pixels_from_rle_file(in_file_name)
    img_2_masks = my_utils.get_img_2_masks(img_2_pixels, sizes)

    new_test_ids = []
    rles = []
    for image_name in image_names:
        masks = img_2_masks[image_name]
        masks = post_process(masks)
        for i in range(masks.shape[2]):
            mask = masks[:,:,i:i+1]
            rle = run_length_encode(mask > 0)
            rles.append(rle)
            new_test_ids.append(image_name)
    my_utils.write_rle_to_file(new_test_ids, rles, out_file_name)
    
