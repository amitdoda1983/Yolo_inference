# iith_GPU_inference

#### compile cuda code
nvcc -shared -Xcompiler -fPIC -o yolo_tcp_kernel.so cuda_kernel_http_server_with_yolo.cu

#### Run python inference
python3 yolo_inference_with_tcp.py
