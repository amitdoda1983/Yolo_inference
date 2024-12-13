# YOLO Inference Handler
class YOLOInferenceHandler:
    def __init__(self, model_name='yolov5s', conf_threshold=0.5):
        """
        Initialize YOLO inference handler.
        Args:
            model_name (str): YOLO model to load (e.g., 'yolov5s', 'yolov5m').
            conf_threshold (float): Confidence threshold for filtering predictions.
        """
        self.model = torch.hub.load('ultralytics/yolov5:v6.2', model_name).cuda()
        self.conf_threshold = conf_threshold
        self.image_size = (1250, 1920, 3)  # Fixed image size (height, width, channels)
        self.class_names = [
            "person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def preprocess_gpu_buffer(self, gpu_memory):
        """
        Preprocess a GPU memory buffer containing raw image data for YOLO inference.
        Args:
            gpu_memory (int): Pointer to the GPU memory containing raw image data.
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        height, width, channels = self.image_size
        image_size = height * width * channels

        # Create a dummy PyTorch ByteTensor on the GPU directly from raw byte data
        image_tensor = torch.cuda.ByteTensor(image_size)  # Allocate ByteTensor on GPU
        cuda.memcpy_dtod(image_tensor.data_ptr(), gpu_memory, image_size)  # Copy raw data bytes

        # Reshape and normalize the tensor
        image_tensor = image_tensor.view(channels, height, width).float() / 255.0  # CHW format
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

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
        with torch.no_grad():
            return self.model(image_tensor)

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
        keep = confidences > self.conf_threshold
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
        return json.dumps(detections, indent=4)


# TCP Packet Sender
class TCPPacketSender:
    def __init__(self, shared_library_path='./yolo_tcp_kernel.so'):
        """
        Initialize the TCP packet sender.
        Args:
            shared_library_path (str): Path to the compiled shared library.
        """
        self.kernel_lib = ctypes.CDLL(shared_library_path)
        self.kernel_lib.kernel_yolo_tcp_server.argtypes = [
            ctypes.c_void_p,  # CUDA stream
            ctypes.POINTER(ctypes.c_uint32),  # Exit condition pointer
            ctypes.c_void_p,  # Transmission queues pointer
            ctypes.c_uint32   # Buffer size
        ]
        self.kernel_lib.kernel_yolo_tcp_server.restype = ctypes.c_int

    def launch_kernel(self, gpu_buffer_ptr, buffer_size):
        """
        Launch the CUDA kernel with the inference results buffer.
        Args:
            gpu_buffer_ptr (int): GPU buffer pointer.
            buffer_size (int): Size of the GPU buffer in bytes.
        """
        exit_cond = ctypes.c_uint32(0)  # Exit condition
        result = self.kernel_lib.kernel_yolo_tcp_server(
            ctypes.c_void_p(0),  # Default CUDA stream
            ctypes.byref(exit_cond),
            ctypes.c_void_p(gpu_buffer_ptr),
            buffer_size
        )
        if result != 0:
            print("Kernel execution failed!")
        else:
            print("Kernel executed successfully!")


# Main Application
class YOLOTCPApplication:
    def __init__(self, gpu_memory, shared_library_path='./yolo_tcp_kernel.so', conf_threshold=0.5):
        self.gpu_memory = gpu_memory
        self.yolo_handler = YOLOInferenceHandler(conf_threshold=conf_threshold)
        self.packet_sender = TCPPacketSender(shared_library_path=shared_library_path)

    def run(self):
        # Preprocess image from GPU memory
        image_tensor = self.yolo_handler.preprocess_gpu_buffer(self.gpu_memory)

        # Perform YOLO inference
        predictions = self.yolo_handler.infer(image_tensor)
        boxes, confidences, class_ids = self.yolo_handler.postprocess(predictions)

        # Serialize detections to JSON
        detections_json = self.yolo_handler.serialize_detections(boxes, confidences, class_ids)

        # Allocate GPU memory for the JSON buffer
        detections_bytes = detections_json.encode()
        buffer_size = len(detections_bytes)
        gpu_buffer = cuda.mem_alloc(buffer_size)
        cuda.memcpy_htod(gpu_buffer, detections_bytes)

        # Launch the kernel
        self.packet_sender.launch_kernel(int(gpu_buffer), buffer_size)


if __name__ == "__main__":
    # Replace with the actual GPU memory pointer
    dummy_gpu_memory = 0x12345678  # Placeholder for GPU memory pointer
    app = YOLOTCPApplication(gpu_memory=dummy_gpu_memory, shared_library_path="./yolo_tcp_kernel.so", conf_threshold=0.5)
    app.run()
