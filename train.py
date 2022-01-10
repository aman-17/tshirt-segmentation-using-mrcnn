import sys
sys.path.append("./demo")
from demo.train_mask_rcnn_demo import *
annotations_path = "../trainingDataset_mrcnn/annotations_train.json"
annotations_folder_path="./trainingDataset_mrcnn/"

dataset_train = load_image_dataset(os.path.join(annotations_folder_path, annotations_path), "trainingDataset_mrcnn", "train")
dataset_val = load_image_dataset(os.path.join(annotations_folder_path, annotations_path), "trainingDataset_mrcnn", "val")



class_number = dataset_train.count_classes()

print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))


#display_image_samples(dataset_train) #display masks

#loading model
config = CustomConfig(class_number)
model = load_training_model(config)

#training model
train_head(model, dataset_train, dataset_train, config) #change the no. of epochs to train in train_head in demo/train_mask_rcnn_demo.py


#load test model, uncomment the below lines after training to test the model.

# test_model, inference_config = load_test_model(class_number)
# test_random_image(test_model, dataset_val, inference_config)
