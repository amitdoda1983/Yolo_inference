/*
 * Copyright (c) 2023-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

 #include <NvInfer.h>
 #include <cuda_runtime_api.h>
 #include <fstream>
 #include <vector>
 #include <algorithm>   
 #include <cstring>
 #include <pthread.h>
 #include <unistd.h>
 #include <doca_log.h>
 #include <doca_gpunetio_dev_buf.cuh>
 #include <doca_gpunetio_dev_sem.cuh>
 #include <doca_gpunetio_dev_eth_txq.cuh>
 
 #include "common.h"
 #include "packets.h"
 #include "inference_engine.h"
 
 // Define only if not already defined
 #ifndef INFERENCE_STATUS_IDLE
 #define INFERENCE_STATUS_IDLE      0
 #define INFERENCE_STATUS_REQUESTED 1
 #define INFERENCE_STATUS_COMPLETED 2
 #define INFERENCE_STATUS_ERROR     3
 #define MAX_INFERENCE_RESULT_SIZE 1048576 // 1MB for inference results
 #endif
 
 DOCA_LOG_REGISTER(TENSORRT_INFERENCE);

 //***************************************************//

 #define TOTAL_DETECTIONS 25200
 #define MAX_DETECTIONS 100
 #define NUM_CLASSES 80
 #define NMS_THRESHOLD 0.45
 #define CONF_THRESHOLD 0.25

 // Detection result structure
 // Detection result structure
 struct Detection {
    float bbox[4];    // [x, y, w, h] in pixels
    float conf;      // confidence score
    int class_id;   // class ID
 };

 // Comparator to sort Detection objects in descending order based on confidence.
struct DescendingConf {
    __host__ __device__
    bool operator()(const Detection& a, const Detection& b) const {
        return a.conf > b.conf;  // Sort descending by confidence
    }
};
//***************************************************//
 
// Only declare the device variable (it's external in the header)
__device__ struct inference_request* device_inference_request;
 
// Host variable for communicating with device
struct inference_request* host_inference_request = nullptr;

// TensorRT global objects
static nvinfer1::IRuntime* runtime = nullptr;
static nvinfer1::ICudaEngine* engine = nullptr;
static nvinfer1::IExecutionContext* context = nullptr;
static void* inputBuffer = nullptr;
static void* outputBuffer = nullptr;
static size_t inputSize = 0;
static size_t outputSize = 0;
static cudaStream_t inference_cuda_stream = nullptr;

// Variables for storing discovered tensor names
static std::string inputTensorName;
static std::string outputTensorName;


//*********************************************************************//
// Preprocessing kernel - converts image to model input format
__global__ void preprocess_kernel(const uint8_t* input, float* output,
                               int input_w, int input_h, int input_c,
                               int model_w, int model_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model_w * model_h) return;
    
    // mean and std values (based on the YOLOv5)
    // float mean[] = {0.485f, 0.456f, 0.406f};
    // float std[] = {0.229f, 0.224f, 0.225f};
    
    // convert 1D index to 2D coordinates
    int x = idx % model_w;
    int y = idx / model_w;
    
    // Scale coordinates
    float scale_w = static_cast<float>(input_w) / model_w;
    float scale_h = static_cast<float>(input_h) / model_h;

    // Resize the image to model input size
    int input_x = min(static_cast<int>(x * scale_w), input_w - 1);
    int input_y = min(static_cast<int>(y * scale_h), input_h - 1);
    
    // Convert BGR to RGB and normalize to [0, 1]
    for (int c = 0; c < 3; c++) {
        int input_idx = (input_y * input_w + input_x) * input_c + (2 - c);
        int output_idx = c * model_h * model_w + y * model_w + x;
        
        output[output_idx] = static_cast<float>(input[input_idx]) / 255.0f;

        // Normalize the pixel value to [0, 1]
        // float normalized_value = static_cast<float>(input[input_idx]) / 255.0f;
        
        // Apply YOLO-specific normalization (subtract mean and divide by std)
        // output[output_idx] = (normalized_value - mean[c]) / std[c];

    }
}

// filter detections  kernel
__global__ void filter_detections_kernel(float* yolo_output, Detection* detections, 
                                      int* detection_count, int max_detections,
                                      int num_outputs, int num_classes, 
                                      float conf_threshold, int img_w, int img_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_anchors = num_outputs / (5 + num_classes);
    
    if (idx >= num_anchors) return;
    
    // Process YOLO output format (x, y, w, h, obj_conf, class_probs...)
    int offset = idx * (5 + num_classes);
    float obj_conf = yolo_output[offset + 4];
    
    if (obj_conf < conf_threshold) return;
    
    // Find class with highest probability
    int class_id = 0;
    float max_prob = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        float prob = yolo_output[offset + 5 + c];
        if (prob > max_prob) {
            max_prob = prob;
            class_id = c;
        }
    }
    
    // Combined confidence
    float confidence = obj_conf * max_prob;
    if (confidence < conf_threshold) return;
    
    // Add detection using atomic operation
    int det_idx = atomicAdd(detection_count, 1);
    detections[det_idx].bbox[0] = yolo_output[offset] * img_w;     // x
    detections[det_idx].bbox[1] = yolo_output[offset + 1] * img_h; // y
    detections[det_idx].bbox[2] = yolo_output[offset + 2] * img_w; // w
    detections[det_idx].bbox[3] = yolo_output[offset + 3] * img_h; // h
    detections[det_idx].conf = confidence;
    detections[det_idx].class_id = class_id;
    
}

// IOU
__device__ float computeIoU(Detection a, Detection b) {
    // Compute Intersection over Union (IoU) between two bounding boxes
    float x1 = fmaxf(a.bbox[0], b.bbox[0]);
    float y1 = fmaxf(a.bbox[1], b.bbox[1]);
    float x2 = fminf(a.bbox[2], b.bbox[2]);
    float y2 = fminf(a.bbox[3], b.bbox[3]);
    
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float areaA = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    float areaB = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    
    float unionArea = areaA + areaB - intersection;
    return intersection / unionArea;
}

// NMS kernel using IOU
__global__ void nms_kernel(const Detection* inputDetections, const int* detectionCount, float nmsThreshold,
                            Detection* outputDetections, int* outputCount, int maxDetections) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= *detectionCount) return;

    Detection current  = inputDetections[idx];
    
    // Initialize a flag to mark whether the detection is suppressed
    bool suppressed = false;
    
    // Compare with all previous detections (since they are sorted in descending order of confidence)
    for (int i = 0; i < idx; i++) {
        Detection other = inputDetections[i];
        if (computeIoU(current , other) > nmsThreshold) {
            suppressed = true;
            break;
        }
    }
    
    // If the detection is not suppressed, keep it
    if (!suppressed) {
        // Atomically reserve an output index
        int outIdx = atomicAdd(outputCount, 1);

        // If we exceed the limit, roll back the atomic count
        if (outIdx < maxDetections) {
            outputDetections[outIdx] = current;
        } else {
            // Rollback since we're ignoring this detection
            atomicSub(outputCount, 1);
        }
    }
}
//*********************************************************************//

// Logger for TensorRT messages
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            DOCA_LOG_ERR("TensorRT: %s", msg);
        } else if (severity == Severity::kWARNING) {
            DOCA_LOG_WARN("TensorRT: %s", msg);
        } else if (severity == Severity::kINFO) {
            DOCA_LOG_INFO("TensorRT: %s", msg);
        }
    }
} gLogger;
 
// Initialize the TensorRT engine
extern "C" bool initialize_inference_engine(const char* enginePath) {
    DOCA_LOG_INFO("Initializing TensorRT engine: %s", enginePath);
    
    // Read the serialized model from disk
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        DOCA_LOG_ERR("Failed to open engine file: %s", enginePath);
        return false;
    }
     
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();
    
    DOCA_LOG_INFO("Engine file size: %zu bytes", engineSize);
    
    // Create TensorRT runtime
    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime) {
        DOCA_LOG_ERR("Failed to create TensorRT runtime");
        return false;
    }
    
    // Deserialize engine
    engine = runtime->deserializeCudaEngine(engineData.data(), engineSize);
    if (!engine) {
        DOCA_LOG_ERR("Failed to deserialize TensorRT engine");
        delete runtime;
        runtime = nullptr;
        return false;
    }
     
    // Create execution context
    context = engine->createExecutionContext();
    if (!context) {
        DOCA_LOG_ERR("Failed to create TensorRT execution context");
        delete engine;
        delete runtime;
        engine = nullptr;
        runtime = nullptr;
        return false;
    }
     
    // Discover tensor names dynamically
    int numIO = engine->getNbIOTensors();
    DOCA_LOG_INFO("Engine contains %d I/O tensors", numIO);
    
    // Print all tensor names and find input/output tensors
    printf("=== TensorRT Model Tensors ===\n");
    bool foundInput = false;
    bool foundOutput = false;
     
    for (int i = 0; i < numIO; i++) {
        const char* tensorName = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(tensorName);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            printf("Found input tensor: %s\n", tensorName);
            DOCA_LOG_INFO("Found input tensor: %s", tensorName);
            inputTensorName = tensorName;
            foundInput = true;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            printf("Found output tensor: %s\n", tensorName);
            DOCA_LOG_INFO("Found output tensor: %s", tensorName);
            outputTensorName = tensorName;
            foundOutput = true;
        }
    }
    printf("===========================\n");
     
    // Verify we found both input and output tensors
    if (!foundInput) {
        DOCA_LOG_ERR("No input tensor found in engine");
        return false;
    }
    
    if (!foundOutput) {
        DOCA_LOG_ERR("No output tensor found in engine");
        return false;
    }
    
    // Get dimensions using discovered tensor names
    auto inputDims = engine->getTensorShape(inputTensorName.c_str());
    auto outputDims = engine->getTensorShape(outputTensorName.c_str());
    
    //********************************************************//
    int inputW = 0, inputH = 0, inputC = 3;
    if (inputDims.nbDims == 4) {
    // Explicit batch mode: [N, C, H, W]
    inputC = inputDims.d[1];
    inputH   = inputDims.d[2];
    inputW    = inputDims.d[3];
    } else if (inputDims.nbDims == 3) {
    // Implicit batch mode: [C, H, W]
    inputC = inputDims.d[0];
    inputH   = inputDims.d[1];
    inputW    = inputDims.d[2];
    } else {
    DOCA_LOG_ERROR("Unexpected input dimensions: %d", inputDims.nbDims);
    }

    //*******************************************************//
    DOCA_LOG_INFO("Input tensor: %s, Dimensions: %ld x %ld x %ld x %ld", 
                inputTensorName.c_str(), 
                (long)inputDims.d[0], 
                (long)inputDims.d[1], 
                (long)inputDims.d[2], 
                inputDims.nbDims > 3 ? (long)inputDims.d[3] : 1);
                
    DOCA_LOG_INFO("Output tensor: %s, Dimensions: %ld x %ld", 
                outputTensorName.c_str(), 
                (long)outputDims.d[0], 
                outputDims.nbDims > 1 ? (long)outputDims.d[1] : 1);
    
    // Calculate buffer sizes
    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        inputSize *= inputDims.d[i];
    }
    inputSize *= sizeof(float);
    
    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputSize *= outputDims.d[i];
    }
    outputSize *= sizeof(float);
    
    DOCA_LOG_INFO("Input buffer size: %zu bytes", inputSize);
    DOCA_LOG_INFO("Output buffer size: %zu bytes", outputSize);
    
    // Allocate GPU memory for input and output
    cudaMalloc(&inputBuffer, inputSize);
    cudaMalloc(&outputBuffer, outputSize);
     
    // Create a dedicated CUDA stream for inference
    cudaStreamCreate(&inference_cuda_stream);
    
    // Create inference request structure - initialize host side first
    cudaHostAlloc(&host_inference_request, sizeof(struct inference_request), cudaHostAllocMapped);
    
    // Initialize request structure
    host_inference_request->status = INFERENCE_STATUS_IDLE;
    host_inference_request->inputSize = 0;
    host_inference_request->outputSize = 0;
     
    // Allocate device memory and copy initial values
    void* device_ptr;
    cudaMalloc(&device_ptr, sizeof(struct inference_request));
    cudaMemcpy(device_ptr, host_inference_request, sizeof(struct inference_request), cudaMemcpyHostToDevice);
    
    // Use cudaMemcpyToSymbol to set the device_inference_request pointer
    cudaError_t err = cudaMemcpyToSymbol(device_inference_request, &device_ptr, sizeof(void*));
    if (err != cudaSuccess) {
        DOCA_LOG_ERR("Failed to set device_inference_request: %s", cudaGetErrorString(err));
        return false;
    }
     
    DOCA_LOG_INFO("TensorRT engine initialized successfully");
    return true;
}
 
// Clean up TensorRT resources
extern "C" void cleanup_inference_engine() {
    if (inputBuffer) cudaFree(inputBuffer);
    if (outputBuffer) cudaFree(outputBuffer);
    
    // Get the device pointer first
    void* device_ptr = nullptr;
    cudaMemcpyFromSymbol(&device_ptr, device_inference_request, sizeof(void*));
    
    if (device_ptr) cudaFree(device_ptr);
    if (host_inference_request) cudaFreeHost(host_inference_request);
    
    if (inference_cuda_stream) cudaStreamDestroy(inference_cuda_stream);
    
    // Clean up TensorRT objects
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    
    inputBuffer = nullptr;
    outputBuffer = nullptr;
    host_inference_request = nullptr;
    inference_cuda_stream = nullptr;
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
    
    // Reset the device pointer to nullptr
    void* null_ptr = nullptr;
    cudaMemcpyToSymbol(device_inference_request, &null_ptr, sizeof(void*));
    
    DOCA_LOG_INFO("TensorRT resources cleaned up");
}
 
// Print the inference results
void print_inference_results(float* results, size_t size) {
    // Print raw tensor values
    printf("\n=== INFERENCE RESULTS ===\n");
    
    // Get the number of elements
    int numElements = size / sizeof(float);
    
    // Print model outputs
    printf("Output tensor contains %d elements\n", numElements);
    
    // Limit to first 10 values if there are many
    int displayCount = (numElements > 10) ? 10 : numElements;
    printf("First %d values:\n", displayCount);
    
    for (int i = 0; i < displayCount; i++) {
        printf("[%d]: %.6f\n", i, results[i]);
    }
     
    // Simple top 5 finder without std::sort
    if (numElements > 5) {
        printf("\nTop 5 highest values:\n");
        
        // Find top 5 values
        float top5[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int top5_idx[5] = {-1, -1, -1, -1, -1};
        
        for (int i = 0; i < numElements; i++) {
            // Check if this value should be in top 5
            for (int j = 0; j < 5; j++) {
                if (results[i] > top5[j]) {
                    // Shift everything down
                    for (int k = 4; k > j; k--) {
                        top5[k] = top5[k-1];
                        top5_idx[k] = top5_idx[k-1];
                    }
                    // Insert this value
                    top5[j] = results[i];
                    top5_idx[j] = i;
                    break;
                }
            }
        }
        
        for (int i = 0; i < 5; i++) {
            if (top5_idx[i] >= 0) {
                printf("Class %d: %.6f\n", top5_idx[i], top5[i]);
            }
        }
    }
    
    printf("==========================\n");
}
 
// Function to run inference using the TensorRT engine
bool run_inference(void* input_data, void* output_data, size_t input_size, 
cudaStream_t stream, int imgWidth, int imgHeight, int imgChannels) {
    if (!context || !engine) {
        DOCA_LOG_ERR("TensorRT engine not initialized");
        return false;
    }

    //************************************************************************//
    // Preprocess image
    float* preprocessedInput;
    cudaMalloc(&preprocessedInput, input_size);

    dim3 preBlock(256);
    dim3 preGrid((inputW * inputH + preBlock.x - 1) / preBlock.x);

    preprocess_kernel<<<preGrid, preBlock, 0, stream>>>(
    input_data, preprocessedInput, 
    imgWidth, imgHeight, imgChannels, 
    inputW, inputH);
    
    // Wait for pre-process to complete
    cudaStreamSynchronize(stream);

    // Copy preprocessed to input buffer
    cudaMemcpyAsync(inputBuffer, preprocessedInput, inputSize, cudaMemcpyDeviceToDevice, stream);
    //************************************************************************//

    //  // Check if we need to copy input data
    //  if (input_data != inputBuffer && input_data != nullptr) {
    //      cudaMemcpyAsync(inputBuffer, input_data, input_size > inputSize ? inputSize : input_size, 
    //                    cudaMemcpyDeviceToDevice, stream);
    //  }
     
    // Set input and output tensor addresses using discovered tensor names
    context->setTensorAddress(inputTensorName.c_str(), inputBuffer);
    context->setTensorAddress(outputTensorName.c_str(), outputBuffer);
    
    // Perform inference
    bool status = context->enqueueV3(stream);
    
    // Wait for inference to complete
    cudaStreamSynchronize(stream);

    //**********************************************************//
    // Filter detections
    Detection* deviceDetections;
    int* deviceCount;
    
    cudaMalloc(&deviceDetections, TOTAL_DETECTIONS * sizeof(Detection));
    cudaMalloc(&deviceCount, sizeof(int));
    cudaMemsetAsync(deviceCount, 0, sizeof(int), stream);

    dim3 filterBlock(256);
    int numAnchors = outputSize / sizeof(float) / (5 + NUM_CLASSES);
    dim3 filterGrid((numAnchors + filterBlock.x - 1) / filterBlock.x);

    filter_detections_kernel<<<filterGrid, filterBlock, 0, stream>>>(
    (float*)outputBuffer, deviceDetections, deviceCount,
    MAX_DETECTIONS, outputSize / sizeof(float), NUM_CLASSES,
    CONF_THRESHOLD, imgWidth, imgHeight);
    
    // Wait for filter detection to complete
    cudaStreamSynchronize(stream);

    int filterCount;
    cudaMemcpy(&filterCount, deviceCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Apply Sorting descending based on confidence score .conf
    // Thrust sort on deviceDetection descending by confidence
    {
    thrust::device_ptr<Detection> dev_ptr(deviceDetections);
    thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + filterCount, DescendingConf());
    }

    // Wait for sorting to complete
    cudaStreamSynchronize(stream);
    
    // Apply NMS
    Detection* deviceNmsDetections;
    int* deviceNmsCount;
    cudaMalloc(&deviceNmsDetections, MAX_DETECTIONS * sizeof(Detection));
    cudaMalloc(&deviceNmsCount, sizeof(int));
    cudaMemsetAsync(deviceNmsCount, 0, sizeof(int), stream);
    
    // Launch NMS kernel
    dim3 nmsBlock(256);
    dim3 nmsGrid((filterCount + nmsBlock.x - 1) / nmsBlock.x); // Adjust based on the number of detections
    nms_kernel<<<nmsGrid, nmsBlock, 0, stream>>>(
        deviceDetections, deviceCount, NMS_THRESHOLD, deviceNmsDetections, deviceNmsCount, MAX_DETECTIONS);
    
    // Wait for nms to complete
    cudaStreamSynchronize(stream);

    // Get results after NMS to host
    int numElements = 0;
    std::vector<Detection> tempDetections(MAX_DETECTIONS);
    std::vector<Detection> host_output;

    cudaMemcpyAsync(&numElements, deviceNmsCount, sizeof(int), 
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tempDetections.data(), deviceNmsDetections,
                    MAX_DETECTIONS * sizeof(Detection),
                    cudaMemcpyDeviceToHost, stream); 
    
    // Resize output vector to actual detections count
    tempDetections.resize(numElements);
    host_output = std::move(tempDetections);

    if (status) {
    if (host_output) {
        // Print results to console
        print_inference_results(host_output, numElements*sizeof(float));
        free(host_output);
    }
        
    // Copy to output location if provided (for compatibility)
    if (output_data != nullptr && output_data != deviceNmsDetections) {
        cudaMemcpyAsync(output_data, deviceNmsDetections, numElements * sizeof(Detection), cudaMemcpyDeviceToDevice, stream);
    }
    }
     
    //**********************************************************//
     
    //  if (status) {
    //      // Allocate CPU memory and copy results from GPU
    //      float* host_output = (float*)malloc(outputSize);
    //      if (host_output) {
    //          cudaMemcpy(host_output, outputBuffer, outputSize, cudaMemcpyDeviceToHost);
             
    //          // Print results to console
    //          print_inference_results(host_output, outputSize);
             
    //          free(host_output);
    //      }
         
    //      // Copy to output location if provided (for compatibility)
    //      if (output_data != nullptr && output_data != outputBuffer) {
    //          cudaMemcpyAsync(output_data, outputBuffer, outputSize, cudaMemcpyDeviceToDevice, stream);
    //      }
    //  }
     
    return status;
}
 
// Thread to monitor and process inference requests
void* inference_polling_thread(void* arg) {
    cudaStream_t stream = *(cudaStream_t*)arg;
    struct inference_request local_request;
    
    while (true) {
        // Copy from host memory to local
        memcpy(&local_request, host_inference_request, sizeof(struct inference_request));
        
        // Check if there's a pending inference request
        if (local_request.status == INFERENCE_STATUS_REQUESTED) {
            printf("Processing inference request...\n");
            
            // Run inference
            bool success = run_inference(
                local_request.inputData,
                local_request.outputData,
                local_request.inputSize,
                stream
            );
            
            // Update status
            host_inference_request->status = success ? 
                INFERENCE_STATUS_COMPLETED : INFERENCE_STATUS_ERROR;
            
            // Copy updated status to device
            void* device_ptr = nullptr;
            cudaMemcpyFromSymbol(&device_ptr, device_inference_request, sizeof(void*));
            
            if (device_ptr) {
                cudaMemcpyAsync(device_ptr, host_inference_request, 
                            sizeof(struct inference_request), 
                            cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            
            printf("Inference request %s\n", success ? "completed successfully" : "failed");
        }
        
        // Sleep to avoid CPU overuse
        usleep(100);
    }
    
    return nullptr;
}
 
// Device function for processing HTTP data
__device__ bool process_image_for_inference(const uint8_t* image_data, 
                                        size_t image_size, 
                                        void* output_buffer, 
                                        size_t* output_size) {
    // Process image data for inference - placeholder implementation
    return true;
}
 
// Processing kernel for HTTP data
__global__ void yolo_process_http_data(uint8_t* httpData, uint32_t dataOffset, 
                                    uint32_t dataSize, float* outputBuffer) {
    // Extract image data from HTTP payload
    uint8_t* imageData = httpData + dataOffset;
    
    // Implementation depends on your HTTP data format
}
 
// Main inference kernel
__global__ void cuda_kernel_inference_engine(uint32_t* exit_cond) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only one thread needs to handle the control flow
    if (tid == 0) {
        while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
            // Optional: Check for any post-processing needed for completed inference requests
            
            // Add a small delay to avoid tight loop
            for (volatile int i = 0; i < 10000; i++) { }
        }
    }
}
 
// Kernel launch function
extern "C" doca_error_t kernel_inference_engine(cudaStream_t stream, uint32_t* exit_cond) {
    cudaError_t result = cudaSuccess;
    
    // Check for valid parameters
    if (exit_cond == nullptr) {
        DOCA_LOG_ERR("kernel_inference_engine: invalid input values");
        return DOCA_ERROR_INVALID_VALUE;
    }
    
    // Check for previous CUDA errors
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        DOCA_LOG_ERR("CUDA error before kernel launch: %s", cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }
     
    // Launch the kernel
    cuda_kernel_inference_engine<<<1, 32, 0, stream>>>(exit_cond);
    
    // Check for kernel launch errors
    result = cudaGetLastError();
    if (result != cudaSuccess) {
        DOCA_LOG_ERR("CUDA kernel launch error: %s", cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }
     
    // Start a CPU thread to handle inference requests if not already started
    static pthread_t polling_thread;
    static bool thread_started = false;
    
    if (!thread_started) {
        // Pass the inference stream to the polling thread
        cudaStream_t* stream_ptr = new cudaStream_t(inference_cuda_stream ? 
                                                inference_cuda_stream : stream);
        
        // Create polling thread
        if (pthread_create(&polling_thread, NULL, inference_polling_thread, stream_ptr) != 0) {
            DOCA_LOG_ERR("Failed to create inference polling thread");
            delete stream_ptr;
            return DOCA_ERROR_INITIALIZATION;
        }
        
        thread_started = true;
    }
    
    return DOCA_SUCCESS;
}
 
// Minimal stub to maintain API compatibility with existing code
extern "C" doca_error_t create_inference_results_buffer(
    struct tx_buf* buf,
    struct doca_gpu* gpu_dev,
    struct doca_dev* ddev,
    uint32_t max_size) {
    // Just doing nothing - we're printing results instead
    DOCA_LOG_INFO("Inference results will be printed to console (buffer not created)");
    return DOCA_SUCCESS;
}
 