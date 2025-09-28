import numpy as np

def solve(N: int) -> int:
    # B1: đếm số x có bình phương ≡ r (mod N)
    cnt = np.zeros(N, dtype=np.int64)
    for x in range(1, N):
        cnt[(x * x) % N] += 1
    
    # B2: tính pair[r] = số cặp (a,b) với tổng bình phương ≡ r (mod N)
    # thực hiện convolution tròn qua FFT
    size = 1
    while size < 2 * N:
        size *= 2
    A = np.fft.rfft(np.concatenate([cnt, np.zeros(size - N, dtype=np.int64)]))
    B = A * A
    conv = np.fft.irfft(B).round().astype(np.int64)
    conv = conv[:2*N-1]
    
    # gộp về mod N (do circular convolution)
    pair = np.zeros(N, dtype=np.int64)
    for i in range(len(conv)):
        pair[i % N] += conv[i]
    
    # B3: tính diag[r] = số cặp (a,a)
    diag = np.zeros(N, dtype=np.int64)
    for x in range(1, N):
        diag[(2 * (x * x % N)) % N] += 1
    
    # half_pair[r]
    half_pair = (pair + diag) // 2
    
    # B4: tổng kết
    ans = 0
    for r in range(N):
        ans += half_pair[r] * cnt[r]
    
    return ans