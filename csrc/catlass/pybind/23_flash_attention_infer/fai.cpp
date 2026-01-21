/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// By setting the K_MAX_SHAPE_DIM macro, the dimension of the AscendC Tensor's ShapeInfo is configured to 0,
// optimizing stack space. If you need to use the ShapeInfo of the AscendC Tensor, please undefine this macro.
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

// Helper methods to check for errors
#include "fai_kernel.cpp"
#include "fai_tiling.cpp"
#include "golden.hpp"
#include "helper.hpp"
#include "kernel_common.hpp"

using namespace std;
using namespace optiling;

#define QF16_KVF16_OUTF16_NOLSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000100103
#define QF16_KVF16_OUTF16_NOLSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010100103
#define QF16_KVF16_OUTF16_LSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000101103
#define QF16_KVF16_OUTF16_LSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010101103
// split fuse FP16 no mask BSND
#define QF16_KVF16_OUTF16_NOLSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000100100
#define QF16_KVF16_OUTF16_NOLSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010100100
#define QF16_KVF16_OUTF16_LSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000101100
#define QF16_KVF16_OUTF16_LSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010101100
// split fuse BF16 causal mask BSND
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000100203
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010100203
#define QBF16_KVBF16_OUTBF16_LSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000101203
#define QBF16_KVBF16_OUTBF16_LSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010101203
// split fuse BF16 no mask BSND
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000100200
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010100200
#define QBF16_KVBF16_OUTBF16_LSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000101200
#define QBF16_KVBF16_OUTBF16_LSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010101200

#define QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000200103
#define QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010200103
#define QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000201103
#define QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010201103
// split fuse FP16 no mask TND
#define QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING 5000000000000210100
#define QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000200100
#define QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010200100
#define QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING 5000000000000211100
#define QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000201100
#define QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010201100
// split fuse BF16 causal mask TND
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000200203
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010200203
#define QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000000201203
#define QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING 5000000000010201203
// split fuse BF16 no mask TND
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000200200
#define QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010200200
#define QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING 5000000000000201200
#define QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING 5000000000010201200

// This code section describes the parameters to execute the run function.
struct Options {
    static constexpr auto HELPER =
        "Usage: fai batch qSeqlen kvSeqlen numHeads kvHeads embeddingSize isVariedLen maskType [--dtype DTYPE "
        "--datapath DATA_PATH --device DEVICE_ID]\n";
    static constexpr auto MIN_ARGS = 7;

    // Define default value.
    uint32_t batch{0};
    uint32_t qSeqlen{0};
    uint32_t kvSeqlen{0};
    uint32_t numHeads{0};
    uint32_t kvHeads{0};
    uint32_t embeddingSize{0};
    uint32_t isVariedLen{0};
    uint32_t maskType{0};
    uint32_t deviceId{0};
    uint32_t blockSize{128};
    uint32_t cacheMode{0};
    uint32_t layout{0};
    uint32_t numBlocks{0};
    uint32_t innerPrec{0};
    uint32_t lseFlag{0};
    string dataType = "half";
    string dataPath = "../../examples/23_flash_attention_infer/data";

    Options() = default;

    // Define function to parse the command-line arguments.
    int Parse(int argc, const char **argv) {
        // The number of arguments must >= 7.
        if (argc < MIN_ARGS) {
            printf(HELPER);
            return -1;
        }

        // Allocate arguments to parameters.
        uint32_t argIndex = 1;
        batch = atoi(argv[argIndex++]);
        qSeqlen = atoi(argv[argIndex++]);
        kvSeqlen = atoi(argv[argIndex++]);
        numHeads = atoi(argv[argIndex++]);
        kvHeads = atoi(argv[argIndex++]);
        embeddingSize = atoi(argv[argIndex++]);
        isVariedLen = atoi(argv[argIndex++]);
        maskType = atoi(argv[argIndex++]);
        cacheMode = atoi(argv[argIndex++]);
        layout = atoi(argv[argIndex++]);
        numBlocks = atoi(argv[argIndex++]);
        innerPrec = atoi(argv[argIndex++]);
        lseFlag = atoi(argv[argIndex++]);
        while (argIndex < argc) {
            string flag = string(argv[argIndex++]);
            if (flag == "--datapath") {
                dataPath = string(argv[argIndex++]);
            } else if (flag == "--device") {
                deviceId = atoi(argv[argIndex++]);
            } else if (flag == "--dtype") {
                dataType = string(argv[argIndex++]);
            } else {
                printf(HELPER);
                return -1;
            }
        }
        return 0;
    }
};

static void AllocMem(uint8_t **host, uint8_t **device, size_t size) {
    ACL_CHECK(aclrtMallocHost(reinterpret_cast<void **>(host), size));
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(device), size, ACL_MEM_MALLOC_HUGE_FIRST));
}

static void FreeMem(uint8_t *host, uint8_t *device) {
    ACL_CHECK(aclrtFreeHost(host));
    ACL_CHECK(aclrtFree(device));
}

// Allocate several matrices in NPU device memory and call a
// CATLASS FAI kernel.
static void Run(const Options &options) {
    aclrtStream stream{nullptr};
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));

    // Get the number of cube cores of the current hardware
    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    // Parameters initialization.
    int32_t batch = options.batch;
    int32_t qSeqlen = options.qSeqlen;
    int32_t kvSeqlen = options.kvSeqlen;
    int32_t numHeads = options.numHeads;
    int32_t kvHeads = options.kvHeads;
    int32_t embeddingSize = options.embeddingSize;
    int32_t blockSize = options.blockSize;
    int32_t maskType = options.maskType;
    int32_t cacheMode = options.cacheMode;
    int32_t lseFlag = options.lseFlag;
    int32_t numBlocks = options.numBlocks;
    string layout = "BSND";
    if(options.layout == 1) {
        layout = "TND";
    }
    string dataType = options.dataType;
    string dataPath = options.dataPath;
    int32_t maxKvSeqlen = kvSeqlen;
    if (numBlocks == 0) {
        numBlocks = batch * ((maxKvSeqlen + blockSize - 1) / blockSize);
    }

    if ((dataType != "half") && (dataType != "bf16")) {
        cerr << "[ERROR] dtype must be 'half' or 'bf16'." << endl;
        return;
    }

    // read qNtokens num
    void *qNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&qNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/q_ntokens.bin", qNtokens, 1 * sizeof(int32_t));
    int32_t numTokens = static_cast<int32_t *>(qNtokens)[0];

    void *kvNtokens = nullptr;
    ACL_CHECK(aclrtMallocHost(&kvNtokens, 1 * sizeof(int32_t)));
    ReadFile(dataPath + "/kv_ntokens.bin", kvNtokens, 1 * sizeof(int32_t));
    int32_t kvNumTokens = static_cast<int32_t *>(kvNtokens)[0];

    uint64_t seqArraySize = batch * sizeof(int64_t);
    uint64_t qoSize = (uint64_t)numTokens * (uint64_t)numHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    uint64_t lseSize = (uint64_t)numTokens * (uint64_t)numHeads * sizeof(int32_t);
    uint64_t kvSize = 0;
    if (cacheMode == 1) {
        kvSize = (uint64_t)numBlocks * (uint64_t)blockSize * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    } else {
        kvSize = (uint64_t)kvNumTokens * (uint64_t)kvHeads * (uint64_t)embeddingSize * sizeof(fp16_t);
    }
    uint64_t maskSize = 2048 * 2048 * sizeof(fp16_t);
    uint64_t blockTableSize = static_cast<uint64_t>(
        batch * ((maxKvSeqlen + blockSize - 1) / blockSize) * sizeof(int32_t)
    );
    // ?????
    uint32_t tilingSize = sizeof(FAInferTilingData);

    // Allocate matrices in host and device memory.
    uint8_t *qSeqHost;
    uint8_t *qSeqDevice;
    AllocMem(&qSeqHost, &qSeqDevice, seqArraySize);
    ReadFile(dataPath + "/q_seqlen.bin", qSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(qSeqDevice, seqArraySize, qSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory.
    uint8_t *kvSeqHost;
    uint8_t *kvSeqDevice;
    AllocMem(&kvSeqHost, &kvSeqDevice, seqArraySize);
    ReadFile(dataPath + "/kv_seqlen.bin", kvSeqHost, seqArraySize);
    ACL_CHECK(aclrtMemcpy(kvSeqDevice, seqArraySize, kvSeqHost, seqArraySize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix q.
    uint8_t *qHost;
    uint8_t *qDevice;
    AllocMem(&qHost, &qDevice, qoSize);
    ReadFile(dataPath + "/q.bin", qHost, qoSize);
    ACL_CHECK(aclrtMemcpy(qDevice, qoSize, qHost, qoSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix k.
    uint8_t *kHost;
    uint8_t *kDevice;
    AllocMem(&kHost, &kDevice, kvSize);
    ReadFile(dataPath + "/k.bin", kHost, kvSize);
    ACL_CHECK(aclrtMemcpy(kDevice, kvSize, kHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *vHost;
    uint8_t *vDevice;
    AllocMem(&vHost, &vDevice, kvSize);
    ReadFile(dataPath + "/v.bin", vHost, kvSize);
    ACL_CHECK(aclrtMemcpy(vDevice, kvSize, vHost, kvSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate matrices in host and device memory and load Matrix v.
    uint8_t *maskHost;
    uint8_t *maskDevice;
    if (maskType == 1) {
        AllocMem(&maskHost, &maskDevice, maskSize);
        ReadFile(dataPath + "/mask.bin", maskHost, maskSize);
        ACL_CHECK(aclrtMemcpy(maskDevice, maskSize, maskHost, maskSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // Allocate matrices in host and device memory and load Matrix block_table.
    uint8_t *blockTableHost;
    uint8_t *blockTableDevice;
    if (cacheMode == 1) {
        AllocMem(&blockTableHost, &blockTableDevice, blockTableSize);
        ReadFile(dataPath + "/block_table.bin", blockTableHost, blockTableSize);
        ACL_CHECK(aclrtMemcpy(blockTableDevice, blockTableSize, blockTableHost, blockTableSize, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    FAInferTilingData faiTilingData;
    FAInferContext faiContext;

    faiContext.pagedCacheFlag = cacheMode == 1;
    faiContext.numHeads = numHeads;
    faiContext.numBlocks = numBlocks;
    faiContext.blockSize = blockSize;
    faiContext.kvHeads = kvHeads;
    faiContext.scaleValue = static_cast<float>(1.0 / std::sqrt(1.0 * embeddingSize));;
    faiContext.layout = layout;
    faiContext.lseFlag = lseFlag;
    if (faiContext.pagedCacheFlag) {
        faiContext.maxNumBlocksPerBatch = (maxKvSeqlen + blockSize - 1) / blockSize;
    }
    faiContext.embeddingSize = embeddingSize;
    faiContext.embeddingSizeV = faiContext.embeddingSize;
    
    faiContext.maskType = static_cast<optiling::MaskType>(maskType);
    faiContext.dataType = static_cast<optiling::DataType>(dataType == "bf16");
    faiContext.batch = batch;
    faiContext.qSeqlenList = reinterpret_cast<int64_t *>(qSeqHost);
    faiContext.kvSeqlenList = reinterpret_cast<int64_t *>(kvSeqHost);
    faiContext.isTilingSink = false;
    FAInferTiling fai_tiling(faiContext);
    fai_tiling.SetCoreNum(aicCoreNum);
    fai_tiling.DoTiling(faiTilingData);
    uint64_t tilingKey = fai_tiling.GetTilingKey();

    // uint8_t *sDevice;
    // ACL_CHECK(aclrtMalloc((void **)(&sDevice), faiTilingData.mm1OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // uint8_t *pDevice;
    // ACL_CHECK(aclrtMalloc((void **)(&pDevice), faiTilingData.smOnlineOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // uint8_t *oTempDevice;
    // ACL_CHECK(aclrtMalloc((void **)(&oTempDevice), faiTilingData.mm2OutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // uint8_t *oUpdateDevice;
    // ACL_CHECK(aclrtMalloc((void **)(&oUpdateDevice), faiTilingData.UpdateSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *workspaceDevice{nullptr};
    cout << "faiTilingData.workSpaceSize " << faiTilingData.workSpaceSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&workspaceDevice), faiTilingData.workSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    uint8_t *oDevice{nullptr};
    cout << "qoSize " << qoSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&oDevice), qoSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *lseDevice{nullptr};
    cout << "lseSize " << lseSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&lseDevice), lseSize * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *tilingDevice;
    cout << "tilingSize " << tilingSize << endl;
    ACL_CHECK(aclrtMalloc((void **)(&tilingDevice), tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // get tiling
    void *tilingHost = nullptr;
    ACL_CHECK(aclrtMallocHost(&tilingHost, tilingSize));
    uint32_t blockDim = aicCoreNum;


    // tiling output
    cout << "faiTilingData.numHeads" << faiTilingData.numHeads << endl;
    cout << "faiTilingData.embeddingSize" << faiTilingData.embeddingSize << endl;
    cout << "faiTilingData.embeddingSizeV" << faiTilingData.embeddingSizeV << endl;
    cout << "faiTilingData.numBlocks" << faiTilingData.numBlocks << endl;
    cout << "faiTilingData.blockSize" << faiTilingData.blockSize << endl;
    cout << "faiTilingData.maxQSeqlen" << faiTilingData.maxQSeqlen << endl;
    cout << "faiTilingData.maxKvSeqlen" << faiTilingData.maxKvSeqlen << endl;
    cout << "faiTilingData.kvHeads" << faiTilingData.kvHeads << endl;
    cout << "faiTilingData.batch" << faiTilingData.batch << endl;
    cout << "faiTilingData.maxNumBlocksPerBatch" << faiTilingData.maxNumBlocksPerBatch << endl;
    cout << "faiTilingData.totalTaskNum" << faiTilingData.totalTaskNum << endl;
    cout << "faiTilingData.maskType" << faiTilingData.maskType << endl;
    cout << "faiTilingData.mm1OutSize" << faiTilingData.mm1OutSize << endl;
    cout << "faiTilingData.smOnlineOutSize" << faiTilingData.smOnlineOutSize << endl;
    cout << "faiTilingData.mm2OutSize" << faiTilingData.mm2OutSize << endl;
    cout << "faiTilingData.UpdateSize" << faiTilingData.UpdateSize << endl;
    cout << "faiTilingData.workSpaceSize" << faiTilingData.workSpaceSize << endl;
    cout << "faiTilingData.scaleValue" << faiTilingData.scaleValue << endl;
    tilingHost = reinterpret_cast<void *>(&faiTilingData);
    ACL_CHECK(aclrtMemcpy(tilingDevice, tilingSize, tilingHost, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));
    cout << "tilingkey: " << tilingKey << endl;
    for (int i = 0; i < 10; i++) {
        if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if(tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if(tilingKey == QBF16_KVBF16_OUTBF16_NOLSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL, FaiKenel::inputLayout::BSND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, false, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_BSND_NOCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_BSND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_BSND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, false, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QBF16_KVBF16_OUTBF16_LSEOUT_BSND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<bfloat16_t, bfloat16_t, float, true, FaiKenel::MaskType::MASK_CAUSAL,
                FaiKenel::inputLayout::BSND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, half, false, FaiKenel::MaskType::NO_MASK, FaiKenel::inputLayout::TND><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        } else if (tilingKey == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING) {
            SplitFuse::FAInfer<half, half, half, false, FaiKenel::MaskType::NO_MASK,
                FaiKenel::inputLayout::TND, Catlass::Epilogue::LseMode::OUT_ONLY><<<blockDim, nullptr, stream>>>(
                fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, lseDevice,
                qSeqDevice, kvSeqDevice, workspaceDevice, tilingDevice);
        }
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Copy the result from device to host
        vector<fp16_t> oHostHalf(qoSize / sizeof(fp16_t));
        vector<bfloat16> oHostBf16(qoSize / sizeof(bfloat16));
        if (dataType == "half") {
            ACL_CHECK(aclrtMemcpy(oHostHalf.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        } else if (dataType == "bf16") {
            ACL_CHECK(aclrtMemcpy(oHostBf16.data(), qoSize, oDevice, qoSize, ACL_MEMCPY_DEVICE_TO_HOST));
        }

        // Compute the golden result
        vector<float> goldenHost(qoSize / sizeof(fp16_t));
        const size_t goldenSize = qoSize * 2;
        ReadFile(dataPath + "/golden.bin", goldenHost.data(), goldenSize);

        // Compare the result
        vector<uint64_t> errorIndices = (dataType == "half") ? golden::CompareData(oHostHalf, goldenHost, kvSeqlen)
                                                             : golden::CompareData(oHostBf16, goldenHost, kvSeqlen);
        if (errorIndices.empty()) {
            cout << "Compare success." << endl;
        } else {
            cerr << "Compare failed. Error count: " << errorIndices.size() << endl;
        }
    }

    // Free host memory allocations.
    FreeMem(qSeqHost, qSeqDevice);
    FreeMem(kvSeqHost, kvSeqDevice);
    FreeMem(qHost, qDevice);
    FreeMem(kHost, kDevice);
    FreeMem(vHost, vDevice);
    if (maskType == 1) {
        FreeMem(maskHost, maskDevice);
    }
    if (cacheMode == 1) {
        FreeMem(blockTableHost, blockTableDevice);
    }
    aclrtFree(oDevice);
    aclrtFree(tilingDevice);
    // aclrtFree(sDevice);
    // aclrtFree(pDevice);
    // aclrtFree(oTempDevice);
    // aclrtFree(oUpdateDevice);
    aclrtFree(workspaceDevice);
    aclrtFreeHost(tilingHost);
    aclrtFreeHost(qNtokens);

    // Destroy specified Stream and reset device.
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceId));
    ACL_CHECK(aclFinalize());
}

/// Entry point to mla example.

int main(int argc, const char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) {
        return -1;
    }
    Run(options);
    return 0;
}