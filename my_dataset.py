import my_constants

class ImageDataset(utils.Dataset):
    def load_image_names(self, train_or_val):
        if train_or_val == 'train':
            filename = my_constants.VP_TRAIN_FILE_LIST
        else:
            filename = my_constants.VP_VAL_FILE_LIST

        image_ids_to_ignore= []
        f = open(my_constants.ERRONEOUS_IMAGES)
        for line in f:
            image_ids_to_ignore.append(line.split("#")[0].strip())            
        f.close()

        image_ids = []
        f = open(filename)
        for line in f:
            if line.strip() not in image_ids_to_ignore:
                image_ids.append(line.strip())
            else:
                print("Ignored", line.strip())
        f.close()
        return image_ids
    
    def get_path(self, id_):
        return my_constants.TRAIN_PATH + id_ + '/images/' + id_ + '.png'
    
    def load_data(self, train_or_val):
        self.add_class("nuclei", 1, "nuclei")
        self.sizes = []
        self.img_ids = self.load_image_names(train_or_val)
        for i, id_ in tqdm(enumerate(self.img_ids), total=len(self.img_ids)):
            path = self.get_path(id_)
            img = imread(path)[:, :, :3]
            w, h = img.shape[0], img.shape[1]
            self.sizes.append((w, h))
            self.add_image("nuclei", image_id=my_constants.N_VARS*i+0, path=id_+'_0', width=w, height=h)
            #if w > 1024 or h > 1024:
            #    self.add_image("nuclei", image_id=N_VARS*i+0, path=id_+'_0', width=w, height=h)
            #else:
            #    self.add_image("nuclei", image_id=N_VARS*i+0, path=id_+'_0', width=1024, height=1024)

    def load_image(self, image_id):
        image_name = self.img_ids[image_id]
        path = self.get_path(image_name)
        img = imread(path)[:,:,:3]
        w, h = img.shape[0], img.shape[1]
        #if w > 1024 or h > 1024:
        #    return img
        #else:
        #    return_arr = np.zeros((1024, 1024, 3), dtype=np.uint8)
        #    return_arr[0:w,0:h,:] = img
        #return return_arr
        return img
    
    def load_mask(self, image_id):
        id_ = self.img_ids[image_id]

        mask_path = my_constants.TRAIN_PATH + id_ + '/masks/'
        num_masks = len(next(os.walk(mask_path))[2])
        width, height = self.sizes[image_id]
        mask = np.zeros((width, height, num_masks), dtype=np.bool)
        for i, mask_file in enumerate(next(os.walk(mask_path))[2]):
            mask_ = imread(mask_path + mask_file)
            mask_ = mask_.reshape((mask_.shape[0], mask_.shape[1], 1))
            mask[:,:,i:i+1] = mask_

        #if width > 1024 or height > 1024:
        #    return mask, np.ones(num_masks, np.int32)
        #else:
        #    return_arr = np.zeros((1024, 1024, num_masks), dtype=np.bool)
        #    return_arr[0:width, 0:height,:] = mask
        #    return return_arr, np.ones(num_masks, np.int32)
        return mask, np.ones(num_masks, np.int32)

    def image_reference(self, image_id):
        return 'not implemented'

