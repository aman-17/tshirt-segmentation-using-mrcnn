import cv2
from pprint import pprint
from demo.train_mask_rcnn_demo import *
img = cv2.imread("./test1.jpg")
import numpy as np

test_model, inference_config = load_inference_model(1, './logs/object20220106T2057/mask_rcnn_object_0001.h5')
cv2.imshow('original',img)
res = test_model.detect([img],verbose=1)
pprint(res)
mask = np.array(res[0]['masks']*128 , dtype=np.int8)
cv2.imshow('helo',mask)
cv2.waitKey(-1)


#test_random_image(test_model, img, inference_config)
#image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def test_random_image1(image):
    #image_id = random.choice(dataset_val.image_ids)
    #original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    #    modellib.load_image_gt(dataset_val, inference_config,
    #                           image_id, use_mini_mask=False)

    log("original_image", image)
    # log("image_meta", image_meta)
    # log("gt_class_id", gt_class_id)
    # log("gt_bbox", gt_bbox)
    # log("gt_mask", gt_mask)

    # Model result
    print("Trained model result")
    results = test_model.detect([image], verbose=1)
    print(results)
    
# test_random_image1(image)

