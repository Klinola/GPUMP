#include "gpump.cuh"

/*Example divisor n*/
__constant__ fe_num_t n_fe = {
    {3555649, 9937716, 33799165, 60472610, 45788892, 67108863, 67108863, 67108863, 67108863, 4194303}
};

/*Precalculated Barret parameter of n*/
__constant__ fe_num_t mu = {
    {29278365, 6081522, 4457178, 21159295, 22094617, 81, 0, 0, 0, 0, 16}
};

/*A test function used to print fe_num_t numbers*/
__device__ void printNumberAsHex(const fe_num_t* num) {
    uint8_t bytes[25 * 26 / 8 + 1] = { 0 }; 
    unsigned long long temp = 0;
    int shift = 0;
    int byteIndex = 0;


    for (int i = 0; i < 25; i++) {
        temp |= ((unsigned long long)num->n[i]) << shift;
        shift += 26; 

        while (shift >= 8) {
            bytes[byteIndex++] = temp & 0xff;
            temp >>= 8;
            shift -= 8;
        }
    }

        if (shift > 0) {
        bytes[byteIndex] = temp & 0xff;
    }

    for (int i = byteIndex; i >= 0; i--) {
        printf("%02x", bytes[i]);
        if (i % 4 == 0 && i != 0) printf(" "); 
    }
    printf("\n");
}

/*  Converters  */
__device__ void fe_num_set_b32(fe_num_t* r, const unsigned char* a) {

    for (int i = 0; i < 25; i++) {
        r->n[i] = 0;
    }

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 4; j++) {
            int limb = (8 * i + 2 * j) / 26;
            if (limb < 25) {
                int shift = (8 * i + 2 * j) % 26;
                r->n[limb] |= (uint32_t)((a[31 - i] >> (2 * j)) & 0x3) << shift;
            }
        }
    }
}

__device__ void fe_num_get_b32(unsigned char* r, const fe_num_t* a) {
    for (int i = 0; i < 32; i++) {
        r[i] = 0;
    }
    for (int i = 0; i < 32; i++) {
        int c = 0;
        for (int j = 0; j < 4; j++) {
            int limb = (8 * i + 2 * j) / 26;
            if (limb < 25) {
                int shift = (8 * i + 2 * j) % 26;
                c |= ((a->n[limb] >> shift) & 0x3) << (2 * j);
            }
        }
        r[31 - i] = c;
    }
}

__device__ int mpCompare(const fe_num_t* x, const fe_num_t* y) {
    for (int i = 24; i >= 0; i--) {
        if (x->n[i] > y->n[i]) return 1;
        else if (x->n[i] < y->n[i]) return -1;
    }
    return 0;
}

__device__ void mpAdd(fe_num_t* result, const fe_num_t* x, const fe_num_t* y) {
    unsigned int carry = 0;
    for (int i = 0; i < 25; i++) {
        unsigned long long sum = (unsigned long long)x->n[i] + y->n[i] + carry;
        result->n[i] = sum & ((1ULL << 26) - 1); // sum mod 2^26
        carry = sum >> 26; // sum / 2^26
    }
}


__device__ void addBkPlusOne(fe_num_t* num, int k) {
    if (k + 1 < 25) {
        num->n[k + 1] += 1;
        int i = k + 1;
        while (i < 24 && num->n[i] == 0) {
            num->n[i + 1] += 1;
            i++;
        }
    }
}



__device__ void mpSubtract(fe_num_t* result, const fe_num_t* x, const fe_num_t* y) {
    int borrow = 0;
    for (int i = 0; i < 25; i++) {
        int sub = x->n[i] - y->n[i] - borrow;
        if (sub < 0) {
            sub += (1 << 26);
            borrow = 1;
        }
        else {
            borrow = 0;
        }
        result->n[i] = sub;
    }
}

/*Subtract function used in Barret Reduction*/
__device__ void mpSubtractSafe(fe_num_t* result, const fe_num_t* x, const fe_num_t* y, int k) {
    fe_num_t adjusted_x;
    for (int i = 0; i < 25; i++) {
        adjusted_x.n[i] = x->n[i];
    }
    if (mpCompare(x, y) < 0) {
        addBkPlusOne(&adjusted_x, k);
    }
    mpSubtract(result, &adjusted_x, y);
}

__device__ void mpModularAdd(fe_num_t* result, const fe_num_t* x, const fe_num_t* y, const fe_num_t* mod) {
    fe_num_t temp_sum;
    mpAdd(&temp_sum, x, y);
    if (mpCompare(&temp_sum, mod) >= 0) {
        mpSubtract(result, &temp_sum, mod);
    }
    else {
        *result = temp_sum;
    }
}

__device__ void mpMul(fe_num_t* result, const fe_num_t* x, const fe_num_t* y) {
    int n = 24;
    unsigned long long carry, uv, v;

    for (int i = 0; i <= n + n + 1; i++) {
        result->n[i] = 0;
    }

    for (int i = 0; i <= n; i++) {  // Loop over each limb of y
        carry = 0;
        for (int j = 0; j <= n; j++) {  // Loop over each limb of x
            if (i + j <= n + n) {
                uv = (unsigned long long)x->n[j] * (unsigned long long)y->n[i] + (unsigned long long)result->n[i + j] + carry;
                v = uv & ((1ULL << 26) - 1); // Extract the lower 26 bits
                carry = uv >> 26; // Extract the carry (upper bits)
                result->n[i + j] = (unsigned int)v;  // Store the result
            }
        }
        if (i + n <= n + n) {
            result->n[i + n + 1] += (unsigned int)carry; // Store the last carry
        }
    }
}





__device__ void rightShift(const fe_num_t* num, fe_num_t* result, int shift_bits) {
    int limb_shift = shift_bits / 26;
    int bit_shift = shift_bits % 26;

    for (int i = 0; i < 25; i++) {
        result->n[i] = 0;
    }

    // Perform the shift for each limb.
    for (int i = limb_shift; i < 25; i++) {
        result->n[i - limb_shift] = num->n[i];
    }
}

__device__ void calculateQBar(fe_num_t* q_bar, const fe_num_t* z, const fe_num_t* mu, int k) {
    // Step 1: Compute q by shifting z right by k bits.
    fe_num_t q;
    rightShift(z, &q, (k - 1) * 26); 
    // Step 2: Compute q * mu.
    fe_num_t q_mul_mu;
    mpMul(&q_mul_mu, &q, mu);

    // Step 3: Compute q_bar by shifting (q * mu) right by k + 1 bits.
    rightShift(&q_mul_mu, q_bar, (k + 1) * 26);

}

__device__ void modBkPlus1(fe_num_t* result, const fe_num_t* num, int k) {
    // This function computes num mod b^(k+1) where b = 2^26.
    // It effectively just copies the first k+1 limbs from num to result.
    int limb_count = k + 1;  // Assuming each limb is a power of 2^26.

    // Initialize result to zero.
    for (int i = 0; i < 25; i++) {
        result->n[i] = 0;
    }

    // Copy the relevant limbs.
    for (int i = 0; i < limb_count && i < 25; i++) {
        result->n[i] = num->n[i];
    }
}

__device__ void calculateR(fe_num_t* r, const fe_num_t* z, const fe_num_t* q_bar, const fe_num_t* p, int k) {
    // Step 1: Compute z mod b^(k+1)
    
    fe_num_t z_mod_bk_plus_1;
    modBkPlus1(&z_mod_bk_plus_1, z, k);

    // Step 2: Compute (q_bar * p) mod b^(k+1)
    fe_num_t q_bar_mul_p;
    mpMul(&q_bar_mul_p, q_bar, p); 
    fe_num_t q_bar_mul_p_mod_bk_plus_1;
    modBkPlus1(&q_bar_mul_p_mod_bk_plus_1, &q_bar_mul_p, k);

    // Step 3: Compute r = (z mod b^(k+1)) - ((q_bar * p) mod b^(k+1))
    mpSubtractSafe(r, &z_mod_bk_plus_1, &q_bar_mul_p_mod_bk_plus_1, k);


}



__device__ void barrettReduction(fe_num_t* r, const fe_num_t* z, const fe_num_t* p, const fe_num_t* mu, int k) {
    // Calculate q_bar
    fe_num_t q_bar;
    calculateQBar(&q_bar, z, mu, k);

    // Calculate r
    calculateR(r, z, &q_bar, p, k);

    // While r >= p, subtract p from r
    while (mpCompare(r, p) >= 0) {
        mpSubtract(r, r, p);
    }
}


/*This function is used to compute the s-signature of an Ethernet EIP 1559 type transaction, and is an example of GPUMP usage*/

__device__ void calculateS(uint8_t* s, const uint8_t* k_inv, const uint8_t* h, const uint8_t* rd) {
    fe_num_t k_inv_fe, h_fe, rd_fe;
    fe_num_set_b32(&k_inv_fe, k_inv);
    fe_num_set_b32(&h_fe, h);
    fe_num_set_b32(&rd_fe, rd);

    // (h + rd) mod n
    fe_num_t h_plus_rd;
    mpModularAdd(&h_plus_rd, &h_fe, &rd_fe, &n_fe);
    
    // k_inv * (h + rd)
    fe_num_t product;
    mpMul(&product, &k_inv_fe, &h_plus_rd);
  
    fe_num_t r;
    barrettReduction(&r, &product, &n_fe, &mu, 10);

    fe_num_get_b32(s, &r);
}