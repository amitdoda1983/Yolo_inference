#include <json/json.h> // Ensure this is included for JSON serialization
#include <cuda_runtime.h>
#include <stdio.h>
#include <doca_gpunetio_dev_buf.cuh>
#include <doca_gpunetio_dev_eth_txq.cuh>
#include <doca_gpunetio_dev_sem.cuh>

__global__ void cuda_kernel_http_server_with_yolo(
    uint32_t *exit_cond,
    struct doca_gpu_eth_txq *txq0, struct doca_gpu_eth_txq *txq1, struct doca_gpu_eth_txq *txq2, struct doca_gpu_eth_txq *txq3,
    int sem_num,
    struct doca_gpu_semaphore_gpu *sem_http0, struct doca_gpu_semaphore_gpu *sem_http1, struct doca_gpu_semaphore_gpu *sem_http2, struct doca_gpu_semaphore_gpu *sem_http3,
    struct doca_gpu_buf_arr *buf_yolo_inference, uint32_t nbytes_yolo_inference)
{
    doca_error_t ret;
    struct doca_gpu_eth_txq *txq = NULL;
    struct doca_gpu_semaphore_gpu *sem_http = NULL;
    struct doca_gpu_buf *buf = NULL;
    uintptr_t buf_addr;
    struct eth_ip_tcp_hdr *hdr;
    uint8_t *payload;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t sem_http_idx = lane_id;
    uint64_t doca_gpu_buf_idx = lane_id;
    uint16_t send_pkts = 0;

    // Map warp to corresponding TX queue and semaphore
    if (warp_id == 0) {
        txq = txq0;
        sem_http = sem_http0;
    } else if (warp_id == 1) {
        txq = txq1;
        sem_http = sem_http1;
    } else if (warp_id == 2) {
        txq = txq2;
        sem_http = sem_http2;
    } else if (warp_id == 3) {
        txq = txq3;
        sem_http = sem_http3;
    } else {
        return;
    }

    while (DOCA_GPUNETIO_VOLATILE(*exit_cond) == 0) {
        send_pkts = 0;
        if (txq && sem_http) {
            // Fetch YOLO inference result buffer
            ret = doca_gpu_dev_buf_get_buf(buf_yolo_inference, doca_gpu_buf_idx, &buf);
            if (ret != DOCA_SUCCESS) {
                if (lane_id == 0) {
                    printf("Error %d in doca_gpu_dev_buf_get_buf block %d thread %d\n", ret, warp_id, lane_id);
                    DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                }
                break;
            }

            // Fetch buffer address
            ret = doca_gpu_dev_buf_get_addr(buf, &buf_addr);
            if (ret != DOCA_SUCCESS) {
                if (lane_id == 0) {
                    printf("Error %d in doca_gpu_dev_buf_get_addr block %d thread %d\n", ret, warp_id, lane_id);
                    DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                }
                break;
            }

            // Prepare JSON payload for YOLO inference results
            raw_to_tcp(buf_addr, &hdr, &payload);
            if (lane_id == 0) { // Leader prepares payload
                snprintf((char *)payload, nbytes_yolo_inference, "{\"results\": \"%s\"}", (char *)buf_addr);
            }

            __syncwarp();

            // Enqueue buffer for transmission
            ret = doca_gpu_dev_eth_txq_send_enqueue_strong(txq, buf, sizeof(*hdr) + nbytes_yolo_inference, 0);
            if (ret != DOCA_SUCCESS) {
                if (lane_id == 0) {
                    printf("Error %d in doca_gpu_dev_eth_txq_send_enqueue_strong block %d thread %d\n", ret, warp_id, lane_id);
                    DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                }
                break;
            }

            // Mark semaphore status as done
            ret = doca_gpu_dev_semaphore_set_status(sem_http, sem_http_idx, DOCA_GPU_SEMAPHORE_STATUS_DONE);
            if (ret != DOCA_SUCCESS) {
                if (lane_id == 0) {
                    printf("Error %d in doca_gpu_dev_semaphore_set_status block %d thread %d\n", ret, warp_id, lane_id);
                    DOCA_GPUNETIO_VOLATILE(*exit_cond) = 1;
                }
                break;
            }

            sem_http_idx = (sem_http_idx + WARP_SIZE) % sem_num;
            doca_gpu_buf_idx = (doca_gpu_buf_idx + WARP_SIZE) % TX_BUF_NUM;
            send_pkts++;
        }

        __syncwarp();

        // Commit and push packets if necessary
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            send_pkts += __shfl_down_sync(WARP_FULL_MASK, send_pkts, offset);
        __syncwarp();

        if (lane_id == 0 && send_pkts > 0) {
            doca_gpu_dev_eth_txq_commit_strong(txq);
            doca_gpu_dev_eth_txq_push(txq);
        }

        __syncwarp();
    }
}

extern "C" {

doca_error_t kernel_yolo_tcp_server(
    cudaStream_t stream, uint32_t *exit_cond, struct txq_yolo_queues *yolo_queues, uint32_t nbytes_yolo_inference)
{
    cudaError_t result;

    if (!exit_cond || !yolo_queues || yolo_queues->numq == 0) {
        printf("Invalid input values for YOLO TCP server\n");
        return DOCA_ERROR_INVALID_VALUE;
    }

    // Launch kernel with YOLO inference buffer
    cuda_kernel_http_server_with_yolo<<<1, yolo_queues->numq * WARP_SIZE, 0, stream>>>(
        exit_cond,
        yolo_queues->eth_txq_gpu[0], yolo_queues->eth_txq_gpu[1], yolo_queues->eth_txq_gpu[2], yolo_queues->eth_txq_gpu[3],
        yolo_queues->sem_num,
        yolo_queues->sem_http_gpu[0], yolo_queues->sem_http_gpu[1], yolo_queues->sem_http_gpu[2], yolo_queues->sem_http_gpu[3],
        yolo_queues->buf_yolo_inference.buf_arr_gpu, nbytes_yolo_inference);

    result = cudaGetLastError();
    if (result != cudaSuccess) {
        printf("CUDA kernel failed with error: %s\n", cudaGetErrorString(result));
        return DOCA_ERROR_BAD_STATE;
    }

    return DOCA_SUCCESS;
}

} // extern "C"
