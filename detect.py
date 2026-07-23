# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run YOLOv3 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    """Run YOLOv3 detection inference on various input sources such as images, videos, streams, and YouTube URLs.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL (default: 'yolov5s.pt').
        source (str | Path): Source of input data such as a file, directory, URL, glob pattern, or device identifier
        (default: 'data/images').
        data (str | Path): Path to the dataset YAML file (default: 'data/coco128.yaml').
        imgsz (tuple[int, int]): Inference size as a tuple (height, width) (default: (640, 640)).
        conf_thres (float): Confidence threshold for detection (default: 0.25).
        iou_thres (float): Intersection Over Union (IOU) threshold for Non-Max Suppression (NMS) (default: 0.45).
        max_det (int): Maximum number of detections per image (default: 1000).
        device (str): CUDA device identifier, e.g., '0', '0,1,2,3', or 'cpu' (default: '').
        view_img (bool): Whether to display results during inference (default: False).
        save_txt (bool): Whether to save detection results to text files (default: False).
        save_conf (bool): Whether to save detection confidences in the text labels (default: False).
        save_crop (bool): Whether to save cropped detection boxes (default: False).
        nosave (bool): Whether to prevent saving images or videos with detections (default: False).
        classes (list[int] | None): List of class indices to filter, e.g., [0, 2, 3] (default: None).
        agnostic_nms (bool): Whether to perform class-agnostic NMS (default: False).
        augment (bool): Whether to apply augmented inference (default: False).
        visualize (bool): Whether to visualize feature maps (default: False).
        update (bool): Whether to update all models (default: False).
        project (str | Path): Path to the project directory where results will be saved (default: 'runs/detect').
        name (str): Name for the specific run within the project directory (default: 'exp').
        exist_ok (bool): Whether to allow existing project/name directory without incrementing run index (default:
            False).
        line_thickness (int): Thickness of bounding box lines in pixels (default: 3).
        hide_labels (bool): Whether to hide labels in the results (default: False).
        hide_conf (bool): Whether to hide confidences in the results (default: False).
        half (bool): Whether to use half-precision (FP16) for inference (default: False).
        dnn (bool): Whether to use OpenCV DNN for ONNX inference (default: False).
        vid_stride (int): Stride for video frame rate (default: 1).

    Returns:
        None

    Examples:
        ```python
        # Run YOLOv3 inference on an image
        run(weights='yolov5s.pt', source='data/images/bus.jpg')

        # Run YOLOv3 inference on a video
        run(weights='yolov5s.pt', source='data/videos/video.mp4', view_img=True)

        # Run YOLOv3 inference on a webcam
        run(weights='yolov5s.pt', source='0', view_img=True)
        ```

    Notes:
        This function supports a variety of input sources such as image files, video files, directories, URL patterns,
        webcam streams, and YouTube links. It also supports multiple model formats including PyTorch, ONNX, OpenVINO,
        TensorRT, CoreML, TensorFlow, PaddlePaddle, and others. The results can be visualized in real-time or saved to
        specified directories. Use command-line arguments to modify the behavior of the function.
    """
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1e3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """解析并返回运行YOLOv3模型检测所需的命令行参数。.

    参数说明:
        --weights (list[str]): 模型权重路径或Triton服务器URL。默认值: ROOT / "yolov3-tiny.pt"。
        --source (str): 输入数据源，支持文件/目录/URL/glob模式/屏幕截图/摄像头(0)。默认值: ROOT / "data/images"。
        --data (str): 数据集配置文件路径（可选）。默认值: ROOT / "data/coco128.yaml"。
        --imgsz (list[int]): 推理尺寸，格式为[高度, 宽度]。可接受单个或两个值。默认值: [640]。
        --conf-thres (float): 预测置信度阈值。默认值: 0.25。
        --iou-thres (float): 非极大值抑制(NMS)的IoU阈值。默认值: 0.45。
        --max-det (int): 每张图像的最大检测数量。默认值: 1000。
        --device (str): CUDA设备标识，如 "0" 或 "0,1,2,3" 或 "cpu"。默认值: ""（自动选择）。
        --view-img (bool): 是否显示检测结果。默认值: False。
        --save-txt (bool): 是否将结果保存到*.txt文件。默认值: False。
        --save-conf (bool): 是否在文本标签中保存置信度分数。默认值: False。
        --save-crop (bool): 是否保存裁剪后的检测框图像。默认值: False。
        --nosave (bool): 是否不保存图像/视频。默认值: False。
        --classes (list[int] | None): 按类别过滤结果，如 [0, 2, 3]。默认值: None。
        --agnostic-nms (bool): 是否执行类别无关的NMS。默认值: False。
        --augment (bool): 是否应用增强推理。默认值: False。
        --visualize (bool): 是否可视化特征图。默认值: False。
        --update (bool): 是否更新所有模型。默认值: False。
        --project (str): 保存结果的目录；结果将保存到"project/name"。默认值: ROOT / "runs/detect"。
        --name (str): 运行名称；结果将保存到"project/name"。默认值: "exp"。
        --exist-ok (bool): 是否允许将结果保存到已存在的目录（不自动递增）。默认值: False。
        --line-thickness (int): 检测框线宽（像素）。默认值: 3。
        --hide-labels (bool): 是否隐藏检测标签。默认值: False。
        --hide-conf (bool): 是否隐藏标签上的置信度分数。默认值: False。
        --half (bool): 是否使用FP16半精度推理。默认值: False。
        --dnn (bool): 是否使用OpenCV DNN后端进行ONNX推理。默认值: False。
        --vid-stride (int): 视频输入的帧速率步长。默认值: 1。

    返回:
        argparse.Namespace: 包含YOLOv3推理配置的解析后命令行参数对象。

    示例:
        ```python
        options = parse_opt()
        run(**vars(options))
        ```
    """
    # 创建命令行参数解析器实例
    parser = argparse.ArgumentParser()
    # ========== 模型与输入配置 ==========
    # 添加模型权重参数，支持多个权重路径或Triton服务URL
    parser.add_argument(
        "--weights", nargs="+", type=str, default=ROOT / "yolov3-tiny.pt", help="model path or triton URL"
    )
    # 添加输入源参数，支持文件/目录/URL/glob模式/屏幕截图/摄像头
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    # 添加数据集配置文件路径参数
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    # 添加推理尺寸参数，支持简写形式 --img 和 --img-size，接受1个或2个整数
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")

    # ========== 推理阈值配置 ==========
    # 添加置信度阈值参数，过滤低置信度检测结果
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    # 添加NMS的IoU阈值参数，控制非极大值抑制的严格程度
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    # 添加单图最大检测数量参数，限制每张图的检测框数量
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")

    # ========== 设备配置 ==========
    # 添加CUDA设备选择参数，支持多GPU或CPU
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    # ========== 结果输出控制 ==========
    # 添加实时显示结果参数
    parser.add_argument("--view-img", action="store_true", help="show results")
    # 添加保存检测结果到txt文件参数
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    # 添加在txt标签中保存置信度参数
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    # 添加保存裁剪后的检测框图像参数
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    # 添加不保存图像/视频参数
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")

    # ========== 检测结果过滤 ==========
    # 添加按类别索引过滤参数，支持多个类别
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    # 添加类别无关NMS参数，合并不同类别的重叠检测框
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    # ========== 推理增强选项 ==========
    # 添加测试时数据增强参数，提升检测鲁棒性
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    # 添加特征图可视化参数
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    # 添加模型自动更新参数
    parser.add_argument("--update", action="store_true", help="update all models")

    # ========== 结果保存路径配置 ==========
    # 添加结果保存项目目录参数
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    # 添加运行名称参数，用于区分不同实验结果
    parser.add_argument("--name", default="exp", help="save results to project/name")
    # 添加允许覆盖已存在目录参数
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")

    # ========== 可视化配置 ==========
    # 添加检测框线宽参数
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    # 添加隐藏标签参数
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    # 添加隐藏置信度参数
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")

    # ========== 推理优化配置 ==========
    # 添加FP16半精度推理参数，提升推理速度
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # 添加使用OpenCV DNN后端参数，用于ONNX模型推理
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    # 添加视频帧率步长参数，控制视频采样间隔
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    # ========== 参数解析与后处理 ==========
    # 解析命令行参数，返回Namespace对象
    opt = parser.parse_args()
    # 处理imgsz参数：如果只提供一个值，自动扩展为正方形（如[640] → [640, 640]）
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    # 打印所有解析后的参数配置
    print_args(vars(opt))
    # 返回包含所有配置参数的Namespace对象
    return opt


def main(opt):
    """Entry point for running the YOLO model; checks requirements and calls `run` with parsed options.

    Args:
        opt (argparse.Namespace): Parsed command-line options, which include:
            - weights (str | list of str): Path to the model weights or Triton server URL.
            - source (str): Input source, can be a file, directory, URL, glob, screen, or webcam index.
            - data (str): Path to the dataset configuration file (.yaml).
            - imgsz (tuple of int): Inference image size as (height, width).
            - conf_thres (float): Confidence threshold for detections.
            - iou_thres (float): Intersection over Union (IoU) threshold for Non-Maximum Suppression (NMS).
            - max_det (int): Maximum number of detections per image.
            - device (str): Device to run inference on; options are CUDA device id(s) or 'cpu'
            - view_img (bool): Flag to display inference results.
            - save_txt (bool): Save detection results in .txt format.
            - save_conf (bool): Save detection confidences in .txt labels.
            - save_crop (bool): Save cropped bounding box predictions.
            - nosave (bool): Do not save images/videos with detections.
            - classes (list of int): Filter results by class, e.g., --class 0 2 3.
            - agnostic_nms (bool): Use class-agnostic NMS.
            - augment (bool): Enable augmented inference.
            - visualize (bool): Visualize feature maps.
            - update (bool): Update the model during inference.
            - project (str): Directory to save results.
            - name (str): Name for the results directory.
            - exist_ok (bool): Allow existing project/name directories without incrementing.
            - line_thickness (int): Thickness of bounding box lines.
            - hide_labels (bool): Hide class labels on bounding boxes.
            - hide_conf (bool): Hide confidence scores on bounding boxes.
            - half (bool): Use FP16 half-precision inference.
            - dnn (bool): Use OpenCV DNN backend for ONNX inference.
            - vid_stride (int): Video frame-rate stride.

    Returns:
        None

    Examples:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```

    Notes:
        Run this function as the entry point for using YOLO for object detection on a variety of input sources such as
        images, videos, directories, webcams, streams, etc. This function ensures all requirements are checked and
        subsequently initiates the detection process by calling the `run` function with appropriate options.
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
