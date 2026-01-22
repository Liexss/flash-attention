import sys
import os
import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
torch.npu.set_device(1)
sys.path.append(os.getcwd())
import flash_attn_2_cuda

from flash_attn import (
    flash_attn_with_kvcache,
)

def group_matmul(head, kv_head, left, right, high_prec = 1):
    group_num = head // kv_head
    score = None
    for i in range(kv_head):
        if high_prec == 0:
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :],
                                    right[i:(i + 1), :, :]).astype(np.float32)
        else:
            group_score = np.matmul(left[i * group_num:(i + 1) * group_num, :, :].astype(np.float32),
                                    right[i:(i + 1), :, :].astype(np.float32))
        if score is None:
            score = group_score
        else:
            score = np.concatenate((score, group_score), 0)
    return score

def softmax_numpy(sim):
    row_max = np.max(sim, axis=-1, keepdims=True)
    sim_sub = sim - row_max

    sim_sub = np.exp(sim_sub)
    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)

    soft_res = sim_sub / row_sum
    lse = np.squeeze((np.log(row_sum) + row_max), axis=-1)

    return soft_res, lse, row_max

def softmax1(
    qk_result,
    is_first,
    gm,
    interm_dtype = np.float16
):
    sim = qk_result.astype(interm_dtype)
    lm = np.max(sim, axis=-1, keepdims=True)
    if is_first:
        hm = lm
        dm = 0
    else:
        hm = np.maximum(gm, lm)
        dm = gm - hm
    gm = hm
    sim_sub = sim - hm
    sim_sub = np.exp(sim_sub.astype(interm_dtype))

    row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
    return sim_sub, row_sum, dm, gm


def qkMM1(
    query,
    key
):
    result = None
    qk_k = key.shape[1]
    qk_k_split = 128
    qk_k_loop = (qk_k + 127) // 128
    for qk_k_loop_idx in range(qk_k_loop):
        sub_k = 128 if qk_k_loop_idx != (qk_k_loop - 1) else (qk_k - qk_k_loop_idx * 128)
        partial_Query = query[:, :, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k]
        partial_Key = key[:, qk_k_loop_idx * 128: qk_k_loop_idx * 128 + sub_k, :]
        result_split = group_matmul(partial_Query.shape[0], partial_Key.shape[0], partial_Query, partial_Key, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def pvMM2(
    p,
    value
):
    result = None
    pv_k = value.shape[1]
    pv_k_split = 128
    pv_k_loop = (pv_k + 127) // 128
    for pv_k_loop_idx in range(pv_k_loop):
        sub_k = 128 if pv_k_loop_idx != (pv_k_loop - 1) else (pv_k - pv_k_loop_idx * 128)

        partial_P = p[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        # query_k = query[:, :, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k]
        partial_Value = value[:, pv_k_loop_idx * 128: pv_k_loop_idx * 128 + sub_k, :] 
        # key_k = key[:, qk_k_split: qk_k_split + sub_k, :]
        result_split = group_matmul(partial_P.shape[0], partial_Value.shape[0], partial_P, partial_Value, 0)
        if result is None:
            result = result_split
        else:
            result = result + result_split
    return result

def ref_flash_attention(
    query,
    key,
    value,
    scale,
    mask,
):
    data_type = np.float16
    inner_prec = 0
    interm_dtype = np.float16 if inner_prec == 1 else np.float32
    query = np.transpose(query, (1, 0, 2))
    key = np.transpose(key, (1, 2, 0))
    value = np.transpose(value, (1, 0, 2))
    scale = np.float16(scale) if inner_prec == 1 else np.float32(scale)
    context_len = key.shape[2]
    context_size = 512
    group_num = query.shape[0] // key.shape[0]
    gl = None
    gl_high = None
    go = None
    go_high = None
    if mask is not None:
        mask = mask.cpu().numpy()
    for kv_start in range(0, context_len, context_size):
        sub_len = context_size
        if kv_start + context_size > context_len:
            sub_len = context_len - kv_start
        sub_key = key[:, :, kv_start: kv_start + sub_len]
        sub_mask = None
        if mask is not None:
            print("add mask!")
            sub_mask = mask[:query.shape[1], kv_start: kv_start + sub_len].astype(interm_dtype) * (-1e4)
        sub_value = value[:, kv_start: kv_start + sub_len, :]
        qk_result = qkMM1(query, sub_key).astype(interm_dtype)
        qk_result = qk_result * scale
        if mask is not None:
            qk_result += sub_mask
        if kv_start == 0:
            gm = None
        p_result, row_sum, dm, gm = softmax1(qk_result, kv_start == 0, gm, interm_dtype)
        p_result = p_result.astype(data_type)
        if kv_start == 0:
            gm_high = None
        lo = pvMM2(p_result, sub_value).astype(interm_dtype)
        if kv_start == 0:
            gl = row_sum
            go = lo
        else:
            dm = np.exp(dm)
            gl = gl * dm
            gl = gl + row_sum

            go = go * dm
            go = go + lo
    go = go / gl
    go = np.transpose(go, (1, 0, 2))
    lse = np.squeeze((np.log(gl) + gm), axis=-1).astype(np.float32)
    # lse_high = np.squeeze((np.log(gl_high) + gm_high), axis=-1)
    return go.astype(data_type), lse

def test_fa_custom_ops():
    cache_mode = 0
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0

    batch_size = 1
    q_seqlen = 1024
    kv_seqlen = 1024
    num_heads = 1
    kv_heads = 1
    head_size = 128
    block_size = 128
    num_blocks = 64

    query = np.random.uniform(q_min_range, q_max_range,
        size=(batch_size, q_seqlen, num_heads, head_size)).astype(np.float16)
    key_cache = None
    value_cache = None
    block_tables = []
    if cache_mode == 1:
        key_cache = np.random.uniform(kv_min_range, kv_max_range,
            size=(num_blocks, block_size,
            kv_heads, head_size)).astype(np.float16)

        value_cache = np.random.uniform(kv_min_range, kv_max_range,
            size=(num_blocks, block_size,
            kv_heads, head_size)).astype(np.float16)
        max_num_blocks_per_seq = (kv_seqlen + block_size - 1) // block_size
        for i in range(batch_size):
            block_table = [
                max_num_blocks_per_seq * i + j
                for j in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int64).npu()
    else:
        key_cache = np.random.uniform(kv_min_range, kv_max_range,
                size=(batch_size, kv_seqlen, kv_heads, head_size)).astype(np.float16)
        value_cache = np.random.uniform(kv_min_range, kv_max_range,
            size=(batch_size, kv_seqlen, kv_heads, head_size)).astype(np.float16)
        block_tables = None
    kv_seqlen_list = [kv_seqlen] * batch_size
    scale = 1.0 / (head_size ** 0.5)
    is_causal = False
    window_size_left = -1
    window_size_right = -1
    is_rotary_interleaved = False
    softcap = 0
    num_splits = 0
    query = torch.from_numpy(query).npu()
    key_cache = torch.from_numpy(key_cache).npu()
    value_cache = torch.from_numpy(value_cache).npu()
    kv_seqlen_list = torch.tensor(kv_seqlen_list, dtype=torch.int64).cpu()
    rotary_cos = None
    rotary_sin = None
    cache_batch_idx = None
    leftpad_k = None
    alibi_slopes = None
    # out = torch.zeros((batch_size, q_seqlen, num_heads, head_size), dtype=torch.float16)
    import pdb;pdb.set_trace()
    # out_out, softmax_lse = flash_attn_2_cuda.fwd_kvcache(query, key_cache, value_cache, key_cache, value_cache, kv_seqlen_list, rotary_cos, rotary_sin, cache_batch_idx,
    #                         leftpad_k, block_tables, alibi_slopes, None, scale, is_causal, window_size_left, window_size_right, is_rotary_interleaved, softcap, num_splits)
    out_out = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        None,
        None,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=kv_seqlen_list,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=leftpad_k,
        block_table=block_tables,
        causal=is_causal,
        window_size=[window_size_left,window_size_right],
        rotary_interleaved=is_rotary_interleaved,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
    )

    golden_out = (torch.empty((batch_size, q_seqlen, num_heads, head_size), dtype=torch.float16))
    for i in range(batch_size):
        # import pdb;pdb.set_trace()
        if is_causal:
            print("causal!")
            output, golden_lse = ref_flash_attention(query.detach().cpu().numpy()[i], key_cache.detach().cpu().numpy()[i], value_cache.detach().cpu().numpy()[i], scale, None)
        else:
            output, golden_lse = ref_flash_attention(query.detach().cpu().numpy()[i], key_cache.detach().cpu().numpy()[i], value_cache.detach().cpu().numpy()[i], scale, None)
        # import pdb;pdb.set_trace()
        out = output.reshape(q_seqlen, num_heads, head_size)
        golden_out[i:i+1] = torch.from_numpy(out)
    import pdb;pdb.set_trace()
    print("end")

test_fa_custom_ops()