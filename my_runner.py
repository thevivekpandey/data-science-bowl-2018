from my_config import MyConfig

config = MyConfig()

dataset_train = ImageDataset()
dataset_train.load_data('train')
dataset_train.prepare()

dataset_val = ImageDataset()
dataset_val.load_data('val')
dataset_val.prepare()
