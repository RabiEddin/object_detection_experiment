# import the COCO Evaluator to use the COCO Metrics
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import logging
# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig()

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.NOTSET)

# register your data
register_coco_instances(
    "coco2017_val", {},
    "/data/COCODevKit/annotations/instances_val2017.json",
    "/data/COCODevKit/val2017"
)

# load the config file, configure the threshold value, load weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Create predictor
predictor = DefaultPredictor(cfg)

# Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator(
    "coco2017_val", cfg, True, output_dir="detectron2_output/", use_fast_impl=True
)
val_loader = build_detection_test_loader(cfg, "coco2017_val")

# Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)

"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.507
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.358
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
INFO:detectron2.evaluation.coco_evaluation:Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 33.058 | 50.747 | 35.844 | 15.029 | 37.997 | 46.295 |
INFO:detectron2.evaluation.coco_evaluation:Per-category bbox AP: 
| category      | AP     | category     | AP     | category       | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 48.240 | bicycle      | 26.315 | car            | 35.200 |
| motorcycle    | 36.333 | airplane     | 57.756 | bus            | 59.460 |
| train         | 54.663 | truck        | 23.350 | boat           | 20.819 |
| traffic light | 20.332 | fire hydrant | 60.123 | stop sign      | 57.104 |
| parking meter | 39.847 | bench        | 18.034 | bird           | 28.073 |
| cat           | 58.077 | dog          | 48.858 | horse          | 49.081 |
| sheep         | 41.522 | cow          | 44.332 | elephant       | 52.764 |
| bear          | 57.333 | zebra        | 62.553 | giraffe        | 62.805 |
| backpack      | 9.482  | umbrella     | 31.106 | handbag        | 8.220  |
| tie           | 23.174 | suitcase     | 23.152 | frisbee        | 54.698 |
| skis          | 15.900 | snowboard    | 28.928 | sports ball    | 37.620 |
| kite          | 32.533 | baseball bat | 18.615 | baseball glove | 28.647 |
| skateboard    | 42.802 | surfboard    | 29.890 | tennis racket  | 39.080 |
| bottle        | 30.401 | wine glass   | 25.519 | cup            | 33.251 |
| fork          | 22.188 | knife        | 7.828  | spoon          | 8.823  |
| bowl          | 32.210 | banana       | 18.039 | apple          | 13.464 |
| sandwich      | 24.931 | orange       | 24.126 | broccoli       | 18.604 |
| carrot        | 16.636 | hot dog      | 23.944 | pizza          | 46.475 |
| donut         | 33.878 | cake         | 25.005 | chair          | 20.066 |
| couch         | 31.335 | potted plant | 19.435 | bed            | 34.045 |
| dining table  | 22.019 | toilet       | 52.765 | tv             | 48.548 |
| laptop        | 51.658 | mouse        | 49.814 | remote         | 18.690 |
| keyboard      | 44.129 | cell phone   | 26.584 | microwave      | 46.974 |
| oven          | 26.840 | toaster      | 28.621 | sink           | 29.141 |
| refrigerator  | 46.017 | book         | 8.819  | clock          | 42.764 |
| vase          | 27.963 | scissors     | 22.055 | teddy bear     | 40.532 |
| hair drier    | 3.688  | toothbrush   | 10.006 |                |        |
"""
