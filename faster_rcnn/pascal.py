# import the COCO Evaluator to use the COCO Metrics
import os
import tempfile
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances, load_voc_instances, register_pascal_voc
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator

import logging

from detectron2.evaluation.pascal_voc_evaluation import voc_eval
from detectron2.utils import comm

import logging
import numpy as np
import os
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from functools import lru_cache
import torch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC style AP for Pascal VOC dataset.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)

        # Too many tiny files, download all to local for speed.
        annotation_dir_local = PathManager.get_local_path(
            os.path.join(meta.dirname, "Annotations/")
        )
        self._anno_file_template = os.path.join(annotation_dir_local, "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        dirname = "/workspace/doyoon/master/detectron2_scripts/detectron2_voc2007test_classes_output"
        Path(dirname).mkdir(parents=True, exist_ok=True)
        res_file_template = os.path.join(dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        per_class_metric_df = []
        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])

            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))

            for thresh in range(50, 100, 5):
                rec, prec, ap = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                )
                aps[thresh].append(ap * 100)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}


        for cls_id, cls_name in enumerate(self._class_names):
            per_class_metric_df.append(
                {"category": cls_name, "AP@50": np.mean(aps[50][cls_id])}
            )

        return ret


# This sets the root logger to write to stdout (your console).
# Your script/app needs to call this somewhere at least once.
logging.basicConfig()

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.NOTSET)

# register your data
register_pascal_voc(
    "voc2007_test",
    "/data/VOCdevkit/VOC2007",
    "test",
    2007
)

# load the config file, configure the threshold value, load weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml")

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Create predictor
predictor = DefaultPredictor(cfg)

# Call the COCO Evaluator function and pass the Validation Dataset
evaluator = PascalVOCDetectionEvaluator(
    "voc2007_test"
)

val_loader = build_detection_test_loader(cfg, "voc2007_test")

# Use the created predicted model in the previous step
res = inference_on_dataset(predictor.model, val_loader, evaluator)
print()

"""
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.763
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.502
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.267
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661
INFO:detectron2.evaluation.coco_evaluation:Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 46.716 | 76.273 | 50.198 | 15.410 | 36.147 | 54.233 |
INFO:detectron2.evaluation.coco_evaluation:Per-category bbox AP: 
| category    | AP     | category    | AP     | category   | AP     |
|:------------|:-------|:------------|:-------|:-----------|:-------|
| aeroplane   | 55.424 | bicycle     | 49.507 | bird       | 44.401 |
| boat        | 31.911 | bottle      | 34.192 | bus        | 58.412 |
| car         | 57.973 | cat         | 57.977 | chair      | 30.446 |
| cow         | 48.074 | diningtable | 38.011 | dog        | 52.352 |
| horse       | 55.793 | motorbike   | 49.151 | person     | 49.710 |
| pottedplant | 24.764 | sheep       | 47.731 | sofa       | 41.744 |
| train       | 53.342 | tvmonitor   | 53.393 |            |        |
"""
