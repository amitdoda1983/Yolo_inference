import os
import ctypes
import json
import torch
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CUDA image buffer dimensions (initialize as required)
image_height = 1250
image_width = 1920
image_channels = 3
image_size = image_height * image_width * image_channels

conf_threshold = 0.1
class_names = ["person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"]

# Loading the model to cuda
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').cuda()
model.eval()

# Perform a dummy inference to initialize model and CUDA
start_image = torch.rand((640, 640, 3), device='cuda')  # Random input tensor on GPU
start_image = start_image.permute(2, 0, 1).unsqueeze(0).float() / 255.0

with torch.no_grad():
    model(start_image)

# YOLO Inference Handler
class YOLOInferenceHandler:
    def __init__(self):
        """
        Initialize YOLO inference handler.
        Args:
        """
        self.class_names = class_names
    def preprocess_gpu_buffer(self, gpu_memory_ptr):
        """
        Preprocess a GPU memory buffer containing raw image data for YOLO inference.
        Args:
            gpu_memory_ptr : Pointer to the GPU memory containing raw image data.
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Create a dummy PyTorch ByteTensor on the GPU directly from raw byte data
        #image_tensor = torch.cuda.ByteTensor(image_size)  # Allocate ByteTensor on GPU

        image_tensor = torch.empty(image_size, dtype=torch.uint8, device='cuda')  # Allocate ByteTensor on GPU

        cuda.memcpy_dtod(image_tensor.data_ptr(), gpu_memory_ptr, image_size)  # Copy raw data bytes

        # Reshape and normalize the tensor
        image_tensor = image_tensor.view(height, width, channels)  # Reshape to HWC format
        image_tensor = image_tensor.float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> CHW -> NCHW

        # Resize to 640x640 for YOLOv5
        image_tensor = torch.nn.functional.interpolate(image_tensor, size=(640, 640), mode='bilinear', align_corners=False)

        # Normalize using YOLOv5 standards
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor

    def infer(self, image_tensor):
        """
        Perform inference on a given image tensor.
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.
        Returns:
            torch.Tensor: YOLO predictions.
        """
        model.eval()
        with torch.no_grad():
            return model(image_tensor)

    def postprocess(self, predictions):
        """
        Postprocess YOLO predictions.
        Args:
            predictions (torch.Tensor): YOLO predictions.
        Returns:
            tuple: Filtered bounding boxes, confidences, and class IDs.
        """
        pred = predictions[0]
        boxes, scores = pred[:, :4], pred[:, 4]
        class_ids = pred[:, 5:].argmax(1)
        confidences = scores * pred[:, 5:].max(1).values
        keep = confidences > conf_threshold
        return boxes[keep], confidences[keep], class_ids[keep]

    def serialize_detections(self, boxes, confidences, class_ids):
        """
        Serialize YOLO detections into JSON format.
        Args:
            boxes (torch.Tensor): Bounding boxes.
            confidences (torch.Tensor): Confidence scores.
            class_ids (torch.Tensor): Class IDs.
        Returns:
            str: Serialized JSON string.
        """
        detections = [{
            "xmin": float(box[0]), "ymin": float(box[1]),
            "xmax": float(box[2]), "ymax": float(box[3]),
            "confidence": float(conf),
            "class": int(cls),
            "name": self.class_names[int(cls)]
        } for box, conf, cls in zip(boxes, confidences, class_ids)]

        detections.sort(key=lambda x: x['confidence'], reverse=True)

        return json.dumps(detections, indent=4)


# Main Application
class YOLOTCPApplication:
    def __init__(self):
        self.yolo_handler = YOLOInferenceHandler()

    def run(self, gpu_memory_ptr):
        while True:
            # Preprocess image from GPU memory
            image_tensor = self.yolo_handler.preprocess_gpu_buffer(gpu_memory_ptr)

            # Perform YOLO inference
            predictions = self.yolo_handler.infer(image_tensor)

            boxes, confidences, class_ids = self.yolo_handler.postprocess(predictions)

            #print(len(boxes), len(confidences), len(class_ids))
            #print(class_ids.device, confidences.device, boxes.device)

            # Serialize detections to JSON
            #detections_json = self.yolo_handler.serialize_detections(boxes, confidences, class_ids)

            #print(detections_json)

            # # Allocate GPU memory for the JSON buffer
            # detections_bytes = detections_json.encode()
            # inference_buffer_size = len(detections_bytes)
            # inference_buffer_ptr = cuda.mem_alloc(inference_buffer_size)

            # # copy json from host to cuda
            # cuda.memcpy_htod(inference_buffer, detections_bytes)

            # # Launch the kernel
            # self.packet_sender.launch_kernel(int(inference_buffer_ptr), inference_buffer_size)

            #time.sleep(10)
            break

if __name__ == "__main__":
  inference_times = []
  images_names = []
  image_folder = "images"
  images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

  for image in images:
    raw_image_data = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    # Step 1: Simulate receiving raw byte data (this would be the image TCP packet data)
    height, width, channels = raw_image_data.shape  # Image dimensions
    image_size = height * width * channels  # Total number of bytes for image
    raw_image_data_bytes = raw_image_data.tobytes()

    # Step 2: Allocate GPU memory for the raw byte stream
    gpu_memory = cuda.mem_alloc(image_size)  # Allocate GPU memory for the raw byte data

    # Step 3: Copy the raw byte data to GPU memory
    cuda.memcpy_htod(gpu_memory, raw_image_data_bytes)  # Copy from host to device

    start_time = time.time()
    image_gpu_memory_ptr = int(gpu_memory)  # Placeholder for GPU memory pointer
    app = YOLOTCPApplication()
    app.run(image_gpu_memory_ptr)
    end_time = time.time()
    inference_times.append((end_time - start_time)*1000)
    images_names.append(image)

  print(f"Images : {images_names}")
  print(f"Inference_times : {inference_times}")
  print(f"Average inference time : {(np.average(inference_times)):.6f} miliseconds.")
  print(f"Median inference time : {np.median(inference_times):.6f} miliseconds.")