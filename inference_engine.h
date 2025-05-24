#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <stdbool.h>
#include "common.h"

/* Status constants for inference requests */
#define INFERENCE_STATUS_IDLE      0
#define INFERENCE_STATUS_REQUESTED 1
#define INFERENCE_STATUS_COMPLETED 2
#define INFERENCE_STATUS_ERROR     3
#define MAX_INFERENCE_RESULT_SIZE 1048576 // 1MB buffer size

/* Structure for inference requests */
struct inference_request {
    volatile int status;       // Status flags defined above
    void* inputData;           // Pointer to input data in GPU memory
    void* outputData;          // Pointer to output buffer in GPU memory
    size_t inputSize;          // Size of input data in bytes
    size_t outputSize;         // Size of output buffer in bytes
};

#ifdef __cplusplus
extern "C" {
#endif

bool initialize_inference_engine(const char* engine_path);
void cleanup_inference_engine(void);

/* Add proper declaration for create_inference_results_buffer */
doca_error_t create_inference_results_buffer(
    struct tx_buf* buf,
    struct doca_gpu* gpu_dev,
    struct doca_dev* ddev,
    uint32_t max_size
);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_ENGINE_H */
