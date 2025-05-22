#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <vector>
#include <doca_log.h>
#include <thrust/device_ptr.h>  // for creating device pointers (wrappers around raw CUDA pointers)
#include <thrust/sort.h> // for performing sorting operations directly on the GPU


// Register DOCA log source
DOCA_LOG_REGISTER(YOLO_INFERENCE);

// Constants
#define TOTAL_DETECTIONS 25200
#define MAX_DETECTIONS 100
#define MAX_CLASSES 80

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


// TensorRT logger for messages
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity == Severity::kERROR) {
            DOCA_LOG_ERR("TensorRT: %s", msg);
        } else if (severity == Severity::kWARNING) {
            DOCA_LOG_WARN("TensorRT: %s", msg);
        }
    }
} gLogger;

// Preprocessing kernel - converts image to model input format
__global__ void preprocess_kernel(const uint8_t* input, float* output,
                               int input_w, int input_h, int input_c,
                               int model_w, int model_h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model_w * model_h) return;
    
    // mean and std values (based on the YOLOv5)
    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[] = {0.229f, 0.224f, 0.225f};
    

    int x = idx % model_w;
    int y = idx / model_w;
    
    // Scale coordinates
    float scale_w = static_cast<float>(input_w) / model_w;
    float scale_h = static_cast<float>(input_h) / model_h;
    int input_x = min(static_cast<int>(x * scale_w), input_w - 1);
    int input_y = min(static_cast<int>(y * scale_h), input_h - 1);
    
    // Convert BGR to RGB and normalize to [0, 1]
    for (int c = 0; c < 3; c++) {
        int input_idx = (input_y * input_w + input_x) * input_c + (2 - c);
        int output_idx = c * model_h * model_w + y * model_w + x;
        
        // output[output_idx] = static_cast<float>(input[input_idx]) / 255.0f;

        // Normalize the pixel value to [0, 1]
        float normalized_value = static_cast<float>(input[input_idx]) / 255.0f;
        
        // Apply YOLO-specific normalization (subtract mean and divide by std)
        
        // Normalize using the mean and std for each channel
        output[output_idx] = (normalized_value - mean[c]) / std[c];

    }
}

// Detection filtering kernel
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
    // if (det_idx < max_detections) {
    //     // Store detection
    //     detections[det_idx].bbox[0] = yolo_output[offset] * img_w;     // x
    //     detections[det_idx].bbox[1] = yolo_output[offset + 1] * img_h; // y
    //     detections[det_idx].bbox[2] = yolo_output[offset + 2] * img_w; // w
    //     detections[det_idx].bbox[3] = yolo_output[offset + 3] * img_h; // h
    //     detections[det_idx].conf = confidence;
    //     detections[det_idx].class_id = class_id;
    // }
    detections[det_idx].bbox[0] = yolo_output[offset] * img_w;     // x
    detections[det_idx].bbox[1] = yolo_output[offset + 1] * img_h; // y
    detections[det_idx].bbox[2] = yolo_output[offset + 2] * img_w; // w
    detections[det_idx].bbox[3] = yolo_output[offset + 3] * img_h; // h
    detections[det_idx].conf = confidence;
    detections[det_idx].class_id = class_id;
    
}


// NMS kernel using IOU
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


// TensorRT engine wrapper
class YoloTensorRT {
private:
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    void* buffers[2] = {nullptr, nullptr};
    
    // TensorRT 10.7 uses named tensors instead of binding indices
    std::string inputName = "images";
    std::string outputName = "output";
    
    cudaStream_t stream;
    
    int inputW = 0, inputH = 0, inputC = 3;
    int numClasses = 80;
    
    size_t inputSize = 0, outputSize = 0;
    bool initialized = false;

public:
    YoloTensorRT() { cudaStreamCreate(&stream); }
    
    ~YoloTensorRT() {
        if (initialized) {
            if (buffers[0]) cudaFree(buffers[0]);
            if (buffers[1]) cudaFree(buffers[1]);
            
            // Use delete instead of destroy()
            if (context) delete context;
            if (engine) delete engine;
            if (runtime) delete runtime;
            
            cudaStreamDestroy(stream);
        }
    }
    
    bool loadEngine(const std::string& engineFile, int classes = 80) {
        // Read engine file
        std::ifstream file(engineFile, std::ios::binary);
        if (!file.good()) {
            DOCA_LOG_ERR("Failed to open engine file: %s", engineFile.c_str());
            return false;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        // Create TensorRT runtime and engine
        runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) return false;
        
        engine = runtime->deserializeCudaEngine(engineData.data(), size);
        //if (!runtime) {
        if (!engine) {
            delete runtime;
            return false;
        }
        
        context = engine->createExecutionContext();
        if (!context) {
            delete engine;
            delete runtime;
            return false;
        }
        
        // Verify tensor names exist in engine
        if (engine->getTensorIOMode(inputName.c_str()) != nvinfer1::TensorIOMode::kINPUT) {
            DOCA_LOG_ERR("Input tensor '%s' not found in engine", inputName.c_str());
            return false;
        }
        
        if (engine->getTensorIOMode(outputName.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) {
            DOCA_LOG_ERR("Output tensor '%s' not found in engine", outputName.c_str());
            return false;
        }
        
        numClasses = classes;
        
        // Get dimensions - using newer API
        auto inputDims = engine->getTensorShape(inputName.c_str());
        inputC = inputDims.d[1];  // NCHW format
        inputH = inputDims.d[2];
        inputW = inputDims.d[3];
        
        // Calculate buffer sizes
        inputSize = inputC * inputH * inputW * sizeof(float);
        
        auto outputDims = engine->getTensorShape(outputName.c_str());
        int outputElements = 1;
        for (int i = 1; i < outputDims.nbDims; i++) {
            outputElements *= outputDims.d[i];
        }
        outputSize = outputElements * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&buffers[0], inputSize);
        cudaMalloc(&buffers[1], outputSize);
        
        initialized = true;
        return true;
    }
    
    bool infer(const uint8_t* imageData, int imgWidth, int imgHeight, int imgChannels,
             std::vector<Detection>& finalDetections, float confThreshold = 0.25, float nmsThreshold = 0.45) {
        if (!initialized) return false;
        
        // Copy image to GPU
        uint8_t* deviceImageData;
        cudaMalloc(&deviceImageData, imgWidth * imgHeight * imgChannels);
        cudaMemcpyAsync(deviceImageData, imageData, 
                     imgWidth * imgHeight * imgChannels, 
                     cudaMemcpyHostToDevice, stream);
        
        // Preprocess image
        float* preprocessedInput;
        cudaMalloc(&preprocessedInput, inputSize);
        
        dim3 preBlock(256);
        dim3 preGrid((inputW * inputH + preBlock.x - 1) / preBlock.x);
        
        preprocess_kernel<<<preGrid, preBlock, 0, stream>>>(
            deviceImageData, preprocessedInput, 
            imgWidth, imgHeight, imgChannels, 
            inputW, inputH);
        
        // Copy to input buffer
        cudaMemcpyAsync(buffers[0], preprocessedInput, inputSize, 
                     cudaMemcpyDeviceToDevice, stream);
        
        // Run inference - use updated API for TensorRT 10.7
        context->setTensorAddress(inputName.c_str(), buffers[0]);
        context->setTensorAddress(outputName.c_str(), buffers[1]);
        context->enqueueV3(stream);
        
        // Filter detections
        Detection* deviceDetections;
        int* deviceCount;
        
        cudaMalloc(&deviceDetections, TOTAL_DETECTIONS * sizeof(Detection));
        cudaMalloc(&deviceCount, sizeof(int));
        cudaMemsetAsync(deviceCount, 0, sizeof(int), stream);
        
        dim3 filterBlock(256);
        int numAnchors = outputSize / sizeof(float) / (5 + numClasses);
        dim3 filterGrid((numAnchors + filterBlock.x - 1) / filterBlock.x);
        
        filter_detections_kernel<<<filterGrid, filterBlock, 0, stream>>>(
            (float*)buffers[1], deviceDetections, deviceCount,
            MAX_DETECTIONS, outputSize / sizeof(float), numClasses,
            confThreshold, imgWidth, imgHeight);
        
        
        int filterCount;
        cudaMemcpy(&filterCount, deviceCount, sizeof(int), cudaMemcpyDeviceToHost);
            
        //**********************************************************************************//
        // Apply Sorting descending based on .conf
        //**********************************************************************************//
        // Thrust sort on device descending by confidence
        {
        thrust::device_ptr<Detection> dev_ptr(deviceDetections);
        thrust::sort(thrust::cuda::par.on(stream), dev_ptr, dev_ptr + filterCount, DescendingConf());
        }

        //*********************************************************************************//
        
        
        // Apply NMS
        Detection* deviceFinalDetections;
        int* deviceFinalCount;
        cudaMalloc(&deviceFinalDetections, MAX_DETECTIONS * sizeof(Detection));
        cudaMalloc(&deviceFinalCount, sizeof(int));
        cudaMemsetAsync(deviceFinalCount, 0, sizeof(int), stream);

        // Launch NMS kernel
        dim3 nmsBlock(256);
        dim3 nmsGrid((filterCount + nmsBlock.x - 1) / nmsBlock.x); // Adjust based on the number of detections
        nms_kernel<<<nmsGrid, nmsBlock, 0, stream>>>(
            deviceDetections, deviceCount, nmsThreshold, deviceFinalDetections, deviceFinalCount, MAX_DETECTIONS);

        cudaStreamSynchronize(stream);


        // Get results after NMS
        int finalCount = 0;
        std::vector<Detection> tempDetections(MAX_DETECTIONS);

        cudaMemcpyAsync(&finalCount, deviceFinalCount, sizeof(int), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(tempDetections.data(), deviceFinalDetections,
                        MAX_DETECTIONS * sizeof(Detection),
                        cudaMemcpyDeviceToHost, stream);

        
         // Resize output vector to actual detections count
        tempDetections.resize(finalCount);
        finalDetections = std::move(tempDetections);

        //*********************************************************************************//

        // Get results
        // int count = 0;
        // std::vector<Detection> rawDetections(MAX_DETECTIONS);
        
        // cudaMemcpyAsync(&count, deviceCount, sizeof(int), 
        //              cudaMemcpyDeviceToHost, stream);
        // cudaMemcpyAsync(rawDetections.data(), deviceDetections,
        //              MAX_DETECTIONS * sizeof(Detection),
        //              cudaMemcpyDeviceToHost, stream);
        
        
        // Clean up
        cudaFree(deviceFinalDetections);
        cudaFree(deviceFinalCount);
        cudaFree(deviceImageData);
        cudaFree(preprocessedInput);
        cudaFree(deviceDetections);
        cudaFree(deviceCount);
        
        return true;
    }
    
    // Getters
    int getInputWidth() const { return inputW; }
    int getInputHeight() const { return inputH; }
};

// Global engine instance
static YoloTensorRT* g_yolo_engine = nullptr;

// Host functions
extern "C" bool initialize_yolov5_engine(const char* engineFile, int numClasses = 80) {
    if (g_yolo_engine) delete g_yolo_engine;
    g_yolo_engine = new YoloTensorRT();
    return g_yolo_engine->loadEngine(engineFile, numClasses);
}

extern "C" bool run_yolov5_inference(const uint8_t* data, int width, int height, 
                                 int channels, Detection* detections, int* count) {
    std::vector<Detection> results;
    bool success = g_yolo_engine->infer(data, width, height, channels, results);
    
    if (success) {
        *count = results.size();
        memcpy(detections, results.data(), results.size() * sizeof(Detection));
    }
    
    return success;
}

// HTTP data processing kernel - integrate with reassembly buffer
__global__ void yolo_process_http_data(uint8_t* httpData, uint32_t dataOffset, 
                                      uint32_t dataSize, float* outputBuffer) {
    // This kernel would extract image data from HTTP payload
    // and prepare it for the inference engine
    uint8_t* imageData = httpData + dataOffset;
    
    // Access would continue with preprocessing and inference calls
    // Implementation depends on your HTTP data format
}
