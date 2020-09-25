import atexit
import bisect
import multiprocessing as mp
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


def build_augmentations(cfg):
    #     augs = [
    #         T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
    #         T.RandomBrightness(0.7, 0.9),
    #         T.RandomFlip(prob=0.5),
    #         T.RandomApply(T.RandomRotation(angle=[-25, 25]), prob=0.5),
    #         T.RandomApply(prob=0.2, tfm_or_aug=T.RandomLighting(scale=0.5))
    #     ]
    augs = [
        T.RandomFlip(prob=0.5),
        T.RandomApply(T.RandomRotation(angle=[-25, 25]), prob=0.9)]
    if cfg.INPUT.CROP.ENABLED:
        #instead of appending, putting cropping in front
        augs = [T.RandomCrop_CategoryAreaConstraint(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)] + augs

    return augs


def get_model_zoo_configs(path=Path(r"/configs")):
    """

    :param path:
    :return:
    """

    # Getting some premade model configs (and weight paths)
    model_zoo_paths = []
    for x in path.rglob("*yaml"):
        x = str(Path(*x.parts[np.where('configs' == np.array(x.parts))[0][0] + 1:]))
        model_zoo_paths.append(x)

    return model_zoo_paths

def ratio_split(file_list, ratio):
    '''

    :param file_list:
    :param ratio:
    :return:
    '''

    fil_num = len(file_list)
    index = int(fil_num * ratio)
    set_a, set_b = file_list[:index], file_list[index:]
    return set_a, set_b

def retinanet_dataloader(input_path, mode='train', ratio = 1.0):

    """

    :param input_path:
    :return:
    """

    images_path = Path.joinpath(input_path, "images")
    csv_path = Path.joinpath(input_path, "labels_images.csv")
    columns = ['class', 'x', 'y', 'w', 'h', 'path', 'width', 'height']
    input_csv = pd.read_csv(csv_path, header=None, names=columns)

    train, test = ratio_split(input_csv.index.values, ratio = ratio)
    if mode == 'train':
        input_csv = input_csv.loc[train]
    else:
        input_csv = input_csv.loc[test]

    data_dict = []
    for i, file_name in enumerate(input_csv['path'].unique()):
        input_subset = input_csv.loc[input_csv['path'] == file_name]

        annotations = []
        for index, row in input_subset.iterrows():
            single_instance_dict = {'bbox': [int(row['x']), int(row['y']), int(row['w']), int(row['h'])],
                                    'bbox_mode': BoxMode.XYWH_ABS, 'category_id': 0, 'iscrowd': 0}
            annotations.append(single_instance_dict)

        image_instance = {'file_name': os.path.join(images_path, file_name), 'image_id': f'image_{i}',
                          'height': int(input_subset['height'].values[0]),
                          'width': int(input_subset['width'].values[0]),
                          'annotations': annotations}
        data_dict.append(image_instance)
    return data_dict

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]

class Trainer(DefaultTrainer):
    '''
    Making custom trainer so we can use our own list of augmentations
    '''

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations= build_augmentations(cfg) )
        return build_detection_train_loader(cfg=cfg, mapper=mapper)


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5






