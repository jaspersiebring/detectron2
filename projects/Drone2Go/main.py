import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from scripts.utils import retinanet_dataloader
from scripts.utils import Trainer #custom trainer
config_path = r"/custom_configs/retinanet_ambulance.yaml"

# Starting logger for some verbose info (i.e. loss printing)
setup_logger()

#Data preparation
dataset_name = 'toy_ambulance'
ratio = 0.8
data_path = Path(r"/home/jsieb/Downloads/data/drone2go")

dataset_name_train = dataset_name + '_train'
dataset_name_test = dataset_name + '_test'

DatasetCatalog.register(dataset_name, lambda : retinanet_dataloader(input_path=data_path))
DatasetCatalog.register(dataset_name_train, lambda : retinanet_dataloader(input_path=data_path, mode = 'train', ratio = ratio))
DatasetCatalog.register(dataset_name_test, lambda : retinanet_dataloader(input_path=data_path, mode = 'test', ratio = ratio))
#Reserved by detectron2:
#thing_class, thing_colors. stuff_classes, stuff_colors, keypoint_names, keypoint_flip_map, keypoint_connection_rules
MetadataCatalog.get(dataset_name).set(thing_classes=['ambulance'])
MetadataCatalog.get(dataset_name_train).set(thing_classes=['ambulance'])
MetadataCatalog.get(dataset_name_test).set(thing_classes=['ambulance'])
#MetadataCatalog.get(dataset_name).data_folder = data_path

#dataset_name = dataset_name_train
data = DatasetCatalog.get(dataset_name)
meta = MetadataCatalog.get(dataset_name)

#meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) #if external
#from detectron2.data.datasets import load_coco_json
#data_dict = load_coco_json(json_file=str(json_path), image_root=str(image_root), dataset_name=dataset_name)
## If you want to save the data/meta as json, can be registered with register_coco_instances
#from detectron2.data.datasets.coco import convert_to_coco_json
#from detectron2.data.datasets import register_coco_instances
#json_path = Path.joinpath(data_path, "labels_images.json")
# convert_to_coco_json(dataset_name=dataset_name, output_file=str(json_path), allow_cached=False)
#image_root = str(Path.joinpath(data_path, 'images'))
# register_coco_instances(dataset_name, meta.as_dict(), str(json_path),  image_root)

# Visualize random image to verify correct data loading
# d = np.random.choice(data, 1)[0]
# random_image_path = d['file_name']
# im = read_image(random_image_path)
# visualizer = Visualizer(im[:, :, ::-1], metadata=meta)
# out = visualizer.draw_dataset_dict(d)
# plt.imshow(out.get_image()[:, :, ::-1])

# Training with loaded data
config_path = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

cfg = get_cfg()
cfg.INPUT
config_path

cfg.merge_from_file(config_path)
cfg.merge_from_file(model_zoo.get_config_file(config_path))
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.INPUT.CROP.ENABLED = True #allow for cropping (without cropping through the labels)
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

#google colab added
cfg.INPUT.MAX_SIZE_TRAIN = 3000
cfg.INPUT.MIN_SIZE_TRAIN = 1000

# Let training initialize from model zoo
try:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)
except RuntimeError as e:
    print(e)
    print("Keeping default weights")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 12   #128 faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
#cfg.MODEL.RETINANET.NUM_CLASSES = 1
#cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
cfg.INPUT.CROP.ENABLED = True
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


''' Loading trainer, should do the following:
- Creates a model, optimizer, scheduler and dataloader from the given config (CFG)
- Loads a checkpoint or cfg.MODEL.WEIGHT if exists, when resume_or_load is called
- Register a few common hooks defined by the config (e.g. CallbackHook, IterationTimer, LRScheduler)
See https://detectron2.readthedocs.io/_modules/detectron2/engine/hooks.html
'''


# the actual Data Loader (will serialize data, load everything at once)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# Custom trainer (with augmentation)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
#trainer.resume_or_load(resume=False)
trainer.train()

############################
# Visualizing augmentation #
############################

# loader = trainer.build_train_loader(cfg)
# sample_image_batches = 2
# f = plt.figure(figsize=(20, 10 * sample_image_batches))
# idx = 1
#
# for sample_image_batch_idx, train_image_batch in enumerate(loader):
#
#     for idx, train_image in enumerate(train_image_batch):
#         image = train_image["image"].numpy().transpose(1, 2, 0)
#
#         target_fields = train_image["instances"].get_fields()
#         labels = [meta.thing_classes[i] for i in target_fields["gt_classes"]]
#
#         # visualize ground truth
#         gt_visualizer = Visualizer(
#             image[:, :, ::-1], metadata=meta, scale=1
#         )
#         gt_image_vis = gt_visualizer.overlay_instances(
#             labels=labels,
#             boxes=target_fields.get("gt_boxes", None),
#             masks=target_fields.get("gt_masks", None),
#             keypoints=target_fields.get("gt_keypoints", None),
#         )
#
#         f.add_subplot(sample_image_batches, 2, (2 * sample_image_batch_idx) + idx + 1)
#         plt.title(f"Image {train_image['file_name']}")
#         plt.imshow(gt_image_vis.get_image())#[:, :, ::-1])
#
#     if sample_image_batch_idx >= sample_image_batches -1:
#         break
# plt.show()

#############
# Inference #
#############

# Tensorboard logdir
#logdir = str(Path(cfg.OUTPUT_DIR).absolute())
# Running inference from trained model
cfg = get_cfg()
cfg.TEST.AUG.ENABLED=True
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_2000iterations_aug_600ims_08.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set the testing threshold for this model 0.5
cfg.SCORE_THRESH_TEST = 0.2 #0.6
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3 #0.5 ##box filtering
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.2 #default 0.5
cfg.DATASETS.TEST = (dataset_name, )

predictor = DefaultPredictor(cfg)

# Predicting random image from existing dataset

ravu_folder = Path(r"/media/jsieb/ED598/drone2go_ravu/drone2go_ravu")
image_paths = ravu_folder.rglob("*.JPG")
image_paths = [image_path for image_path in image_paths]

d = np.random.choice(data,size=1)[0]
random_image_path = d['file_name']
#random_image_path = str(np.random.choice(image_paths, size =1)[0])

#d = np.random.choice(data, 1)[0]
#random_image_path = d['file_name']
#random_image_path = r"/tmp/.X11-unix/e.jpeg"
#json_path = os.path.join(os.path.dirname(random_image_path),  f"{os.path.basename(random_image_path).split('.')[0]}.JSON")
im = read_image(random_image_path)
outputs = predictor(im[:,:,::-1])
cpu_instances = outputs['instances'].to("cpu")

print(len(cpu_instances))

plt.imshow(im)

# Filtering
# threshold = 0.5
# class_check = cpu_instances.pred_classes.numpy() == 0
# score_check = cpu_instances.scores.numpy() > threshold
# indices = np.logical_and(class_check, score_check)
# if indices.any():
#     filtered_instances = cpu_instances[indices]
#     found = True
#     centers = filtered_instances.pred_boxes.get_centers().numpy()
#     scores = filtered_instances.scores.numpy()
# else:
#     found = False
#     centers = None
#     scores = None

# Save as JSON
# pred_json = {'image_path' : random_image_path, 'ambulance_found' : found, 'scores' : scores.tolist(), 'centers':centers.tolist()}
# json.dump(pred_json, open(json_path, 'w+'))
cpu_instances
cfg.INPUT.MAX_SIZE_TRAIN

if len(cpu_instances) > 0:
    v = Visualizer(im,metadata=meta, scale=1)
    #v = Visualizer(im,metadata=MetadataCatalog.get('coco_2017_train'), scale=0.3)
    out = v.draw_instance_predictions(cpu_instances)
    plt.imshow(out.get_image())

else:
    print(False)


##############
# Evaluation #
##############

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# eval_dir = r"/home/jsieb/Downloads/repos/detectron2/output/eval"
# evaluator = COCOEvaluator(dataset_name, cfg, distributed=True, output_dir=eval_dir)
# val_loader = build_detection_test_loader(cfg, dataset_name)
# inference_on_dataset(trainer.model, val_loader, evaluator) #or use trainer.test
