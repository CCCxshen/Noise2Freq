import numpy as np
import os, sys
import json

def check(current_row, current_col, row, col):
    if current_row < 0 or current_row >= row: 
        return False
    if current_col < 0 or current_col >= col: 
        return False
    return True

def gen_scan_idx(stp = 3):
    scan_idx = [0]
    scan_idx_R = [0]*(stp * stp)
    crow = 0; ccol = 0
    a = 0; b = 0
    cnt = 1
    
    while cnt < (stp * stp):
        if a == 0: nrow = crow + 1; ncol = ccol
        if a == 1: nrow = crow; ncol = ccol + 1
        if check(nrow, ncol, stp, stp):
            cnt += 1
            crow = nrow; ccol = ncol
            idx = crow * stp + ccol
            scan_idx.append(idx)
            scan_idx_R[idx] = len(scan_idx) - 1
            while True:
                if b == 0: nrow = crow - 1; ncol = ccol + 1
                if b == 1: nrow = crow + 1; ncol = ccol - 1
                if check(nrow, ncol, stp, stp):
                    cnt += 1
                    crow = nrow; ccol = ncol
                    idx = crow * stp + ccol
                    scan_idx.append(idx)
                    scan_idx_R[idx] = len(scan_idx) - 1
                else: 
                    b = (b + 1) % 2
                    break
        a = (a + 1) % 2
        
    idx_arr = [scan_idx, scan_idx_R]
    # np.save(f"{stp}_ZScan", idx_arr)
    return idx_arr

index_dict = {}
max_size = 512
while max_size > 0:
    idx_arr = gen_scan_idx(max_size)
    index_dict[f"{max_size}_ZScan"] = idx_arr
    max_size //= 2

with open("ZScan.json", "w", encoding="utf-8") as f:
    # ensure_ascii=False：允许存储中文（否则中文会变成 \uXXX 编码）
    json.dump(index_dict, f, ensure_ascii=False, indent=4)
    



