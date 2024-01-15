# GPUMP
# Multiple-Precision Arithmetic and Barrett Reduction on GPU

## Overview
This project implements multiple-precision arithmetic operations with a focus on the Barrett reduction algorithm optimized for GPUs using CUDA. The goal is to facilitate operations on very large integers that standard data types cannot handle, such as those required in cryptographic computations.

## Multiple-Precision Arithmetic Operations

### A. Comparison, Addition, and Subtraction

#### Algorithm 1: Multiple-Precision Comparison
```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits.
OUTPUT: 1 if x > y; 0 if x = y; -1 if x < y.

1. i <- n;
2. while (i >= 0 && x[i] == y[i])
3.     i <- i - 1;
4. end while
5. if (x[i] > y[i])
6.     return 1;
7. else if (x[i] < y[i])
8.     return -1;
9. else
10.    return 0;
```
#### Algorithm 2: Multiple-Precision Addition
```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits.
OUTPUT: x + y = (z_n ... z_1 z_0)_b.

1. c <- 0; // carry digit
2. for (i from 0 to n) do
3.     z_i <- (x_i + y_i + c) mod b;
4.     c <- (x_i + y_i + c) / b;
5. end for
6. z_{n+1} <- c;
7. return (z_n ... z_1 z_0)_b;
```
#### Algorithm 3: Multiple-Precision Subtraction

```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits, x >= y.
OUTPUT: x - y = (z_n ... z_1 z_0)_b.

1. c <- 0; // carry digit
2. for (i from 0 to n) do
3.     z_i <- (x_i - y_i - c) mod b;
4. if ((x_i - y_i - c) < 0) then c <- 1;
5. else c <- 0;
6. end if
7. end for
8. return (z_n ... z_1 z_0)_b;
```

### B. Modular Addition and Subtraction

#### Algorithm 4: Multiple-Precision Modular Addition
```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits, x < m, y < m.
OUTPUT: (x + y) mod m = (z_{n+1} z_n ... z_1 z_0)_b.

1. c <- 0; // carry digit
2. for (i from 0 to n) do
3.     z_i <- (x_i + y_i + c) mod b;
4.     if ((x_i + y_i + c) >= b) then c <- 1;
5.     else c <- 0;
6.     end if
7. end for
8. z_{n+1} <- m_{n+1} - c; // m_{n+1} is 0
9. if ((z_{n+1} z_n ... z_1 z_0)_b >= (m_{n+1} m_n ... m_1 m_0)_b) then
10.    return (z_{n+1} z_n ... z_1 z_0)_b - (m_{n+1} m_n ... m_1 m_0)_b;
11. else return (z_{n+1} z_n ... z_1 z_0)_b;
```
#### Algorithm 5: Multiple-Precision Modular Subtraction

```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits, x < m, y < m.
OUTPUT: (x - y) mod m = (z_{n+1} z_n ... z_1 z_0)_b.

1. if (x >= y)
2.     return (x - y);
3. else
4.     t <- (m - y);
5.     return (x + t) mod m;
6. end if
```
#### Algorithm 6: Multiple-Precision Multiplication
```plaintext
INPUT: non-negative integers x and y, each with n+1 radix b digits.
OUTPUT: x * y = (z_{2n+s+1} z_{2n+s} ... z_1 z_0)_b.

1. for (i from 0 to n+s+1) do
2.     z_i <- 0;
3. end for
4. for (i from 0 to s) do
5.     c <- 0; // carry digit
6.     for (j from 0 to n) do
7.         (uv)_b <- z_{i+j} + x_j * y_i + c;
8.         z_{i+j} <- v; c <- u;
9.     end for
10.    z_{n+i+1} <- u;
11. end for
12. return (z_{2n+s+1} z_{2n+s} ... z_1 z_0)_b;
```

### Barrett Reduction Algorithm

```plaintext
INPUT: 
- Non-negative integers `x` and modulus `p`.
- Radix `b`, the base of `x` and `p` representation.
- Integer `k` such that `k = ⌊log_b(p)⌋ + 1`.
- Integer `z` such that `0 ≤ z < b^(2k)`.
- Precomputed `µ` as `µ = ⌊b^(2k) / p⌋`.

OUTPUT: `z mod p`.

1. Compute `q̄` as `⌊⌊z / b^(k-1)⌋ * µ / b^(k+1)⌋`.
2. Compute `r` as `(z mod b^(k+1)) - (q̄ * p mod b^(k+1))`.
3. If `r < 0` then `r <- r + b^(k+1)`.
4. While `r ≥ p` do `r <- r - p`.
5. Return `r`.
```

## Implementation Details
This implementation is designed for 32-bit limbs with a base of \(2^{26}\), allowing efficient use of the 32-bit integer operations available on GPUs. Each function handles carry and overflow conditions to ensure correctness across all limbs.

## Acknowledgements
Special thanks to my friends who accompanied me through this project!❤️
