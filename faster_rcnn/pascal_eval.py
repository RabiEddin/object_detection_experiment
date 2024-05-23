import json

with open("../detectron2_voc2007test_output/voc2007_test_coco_format.json", "r") as fOut:
    voc2007_target = json.load(fOut)

with open("../detectron2_voc2007test_output/coco_instances_results.json", "r") as fOut:
    voc2007_output = json.load(fOut)


annotations = {}
for annotation in voc2007_target["annotations"]:
    img_id = annotation["image_id"]
    annotations[img_id] = annotation

results = {}
for result in voc2007_output:
    img_id = result["image_id"]
    results[img_id] = result

assert len(annotations) == len(results)

print()

