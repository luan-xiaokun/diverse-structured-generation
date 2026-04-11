#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "uthash.h"

// rolling hash parameters
#define HASH_BASE 256        // base B
#define HASH_MOD 1000000007  // mod M

#define ERROR_RETURN_VALUE -1.0

typedef struct {
  int *positions;
  int count;
  int capacity;
} PositionList;

typedef struct {
  char *kmer;
  PositionList pos_list;
  UT_hash_handle hh;
} KmerEntry;

void init_pos_list(PositionList *list) {
  list->positions = NULL;
  list->count = 0;
  list->capacity = 0;
}

void add_position(PositionList *list, int pos) {
  if (list->count >= list->capacity) {
    list->capacity = (list->capacity == 0) ? 8 : list->capacity * 2;
    int *new_positions =
        (int *)realloc(list->positions, list->capacity * sizeof(int));
    if (!new_positions) { /* Handle allocation error */
      return;
    }
    list->positions = new_positions;
  }
  list->positions[list->count++] = pos;
}

void free_position_list(PositionList *list) {
  if (list->positions) {
    free(list->positions);
    list->positions = NULL;
    list->count = 0;
    list->capacity = 0;
  }
}

long long power(long long base, int exp, long long mod) {
  long long res = 1;
  base %= mod;
  while (exp > 0) {
    if (exp % 2 == 1) {
      res = (res * base) % mod;
    }
    base = (base * base) % mod;
    exp /= 2;
  }
  return res;
}

double wd_shift_kernel_core(const char *s1, int l1, const char *s2, int l2,
                            int d, int s, double *weights) {
  double kernel_value = 0.0;
  KmerEntry *kmer_table = NULL;

  // step 1, preprocess s1
  long long B_pow_k_minus_1 = 1;  // B^(k-1) mod M

  for (int k = 1; k <= d; ++k) {
    if (k > l1) continue;

    if (k > 1) {
      B_pow_k_minus_1 = power(HASH_BASE, k - 1, HASH_MOD);
    } else {
      B_pow_k_minus_1 = 1;  // B^0 = 1
    }

    unsigned long long current_hash = 0;
    char *kmer_buffer = (char *)malloc(k + 1);
    if (!kmer_buffer) {
      return -1.0;  // malloc error
    }

    for (int i = 0; i < k; ++i) {
      current_hash =
          (current_hash * HASH_BASE + (unsigned char)s1[i]) % HASH_MOD;
      kmer_buffer[i] = s1[i];
    }
    kmer_buffer[k] = '\0';

    KmerEntry *entry = NULL;
    HASH_FIND_STR(kmer_table, kmer_buffer, entry);
    if (!entry) {
      entry = (KmerEntry *)malloc(sizeof(KmerEntry));
      entry->kmer = strdup(kmer_buffer);
      init_pos_list(&entry->pos_list);
      HASH_ADD_KEYPTR(hh, kmer_table, entry->kmer, strlen(entry->kmer), entry);
    }
    add_position(&entry->pos_list, 0);

    for (int i = 1; i <= l1 - k; ++i) {
      unsigned char char_out = (unsigned char)s1[i - 1];
      unsigned char char_in = (unsigned char)s1[i + k - 1];
      current_hash =
          (current_hash - (char_out * B_pow_k_minus_1) % HASH_MOD + HASH_MOD) %
          HASH_MOD;
      current_hash = (current_hash * HASH_BASE) % HASH_MOD;
      current_hash = (current_hash + char_in) % HASH_MOD;

      memmove(kmer_buffer, kmer_buffer + 1, k - 1);
      kmer_buffer[k - 1] = s1[i + k - 1];
      kmer_buffer[k] = '\0';

      HASH_FIND_STR(kmer_table, kmer_buffer, entry);
      if (!entry) {
        entry = (KmerEntry *)malloc(sizeof(KmerEntry));
        entry->kmer = strdup(kmer_buffer);
        init_pos_list(&entry->pos_list);
        HASH_ADD_KEYPTR(hh, kmer_table, entry->kmer, strlen(entry->kmer),
                        entry);
      }
      add_position(&entry->pos_list, i);
    }
    free(kmer_buffer);
  }

  // step 2, iterate s2, compute kernel value
  for (int k = 1; k <= d; ++k) {
    if (k > l2) continue;

    long long B_pow_k_minus_1 = 1;
    if (k > 1) {
      B_pow_k_minus_1 = power(HASH_BASE, k - 1, HASH_MOD);
    } else {
      B_pow_k_minus_1 = 1;  // B^0 = 1
    }

    unsigned long long current_hash = 0;
    char *kmer_buffer = (char *)malloc(k + 1);
    if (!kmer_buffer) {
      return -1.0;  // malloc error
    }

    for (int i = 0; i < k; ++i) {
      current_hash =
          (current_hash * HASH_BASE + (unsigned char)s2[i]) % HASH_MOD;
      kmer_buffer[i] = s2[i];
    }
    kmer_buffer[k] = '\0';

    KmerEntry *entry = NULL;
    HASH_FIND_STR(kmer_table, kmer_buffer, entry);
    if (entry) {
      PositionList *plist1 = &entry->pos_list;
      for (size_t p_idx = 0; p_idx < plist1->count; ++p_idx) {
        int dist = abs(plist1->positions[p_idx] - 0);
        if (dist <= s) {
          kernel_value += weights[k] / 2 / (dist + 1);
        }
      }
    }

    for (int j = 1; j <= l2 - k; ++j) {
      unsigned char char_out = (unsigned char)s2[j - 1];
      unsigned char char_in = (unsigned char)s2[j + k - 1];
      current_hash =
          (current_hash - (char_out * B_pow_k_minus_1) % HASH_MOD + HASH_MOD) %
          HASH_MOD;
      current_hash = (current_hash * HASH_BASE) % HASH_MOD;
      current_hash = (current_hash + char_in) % HASH_MOD;

      memmove(kmer_buffer, kmer_buffer + 1, k - 1);
      kmer_buffer[k - 1] = s2[j + k - 1];
      kmer_buffer[k] = '\0';

      entry = NULL;
      HASH_FIND_STR(kmer_table, kmer_buffer, entry);
      if (entry) {
        PositionList *plist1 = &entry->pos_list;
        for (size_t p_idx = 0; p_idx < plist1->count; ++p_idx) {
          int dist = abs(plist1->positions[p_idx] - j);
          if (dist <= s) {
            kernel_value += weights[k] / 2 / (dist + 1);
          }
        }
      }
    }
    free(kmer_buffer);
  }

  KmerEntry *current_entry, *tmp;
  HASH_ITER(hh, kmer_table, current_entry, tmp) {
    HASH_DEL(kmer_table, current_entry);
    free(current_entry->kmer);
    free_position_list(&current_entry->pos_list);
    free(current_entry);
  }

  return kernel_value;
}

double wd_shift_kernel(const char *s1, int l1, const char *s2, int l2, int d,
                       int s) {
  // precomuted weights
  double *weights = NULL;
  weights = (double *)malloc((d + 1) * sizeof(double));
  if (!weights) {
    errno = ENOMEM;
    return ERROR_RETURN_VALUE;
  }
  if (d > 0) {
    double normalization_factor = (double)d * (d + 1);
    for (int k = 1; k <= d; ++k) {
      weights[k] = 2.0 * (d - k + 1) / normalization_factor;
    }
    double kernel_value = wd_shift_kernel_core(s1, l1, s2, l2, d, s, weights);
    free(weights);
    return kernel_value;
  } else {
    errno = EINVAL;
    free(weights);
    return ERROR_RETURN_VALUE;
  }
}

int wd_shift_kernel_matrix_omp(
    const char **strings,  // Array of C strings
    int n,                 // Number of strings
    const int *lengths,    // Array of string lengths
    int d, int s,
    double *output_matrix  // Pre-allocated buffer of size n*n (row-major)
) {
  if (n < 0) {
    errno = EINVAL;
    return -1;  // Indicate error
  }
  if (n == 0) {
    return 0;  // Nothing to do, success
  }
  if (!strings || !lengths || !output_matrix) {
    errno = EINVAL;  // Null pointers
    return -1;
  }

  double *weights = NULL;
  int global_error_flag = 0;  // Shared flag to signal errors from threads
  int first_errno = 0;        // Store the first errno encountered by any thread

  // 1. Pre-calculate weights (once, outside parallel region)
  weights = (double *)malloc((d + 1) * sizeof(double));
  if (!weights) {
    errno = ENOMEM;
    return -1;
  }
  if (d == 0) {  // Handle d=0 case where weights are not really used but
    // avoid div by zero
    errno = EDOM;
    free(weights);
    return -1;
  } else if (d > 0) {  // Ensure d is non-negative
    double normalization_factor = (double)d * (d + 1);
    for (int k = 1; k <= d; ++k) {
      weights[k] = 2.0 * (d - k + 1) / normalization_factor;
    }
  } else {
    errno = EINVAL;
    free(weights);
    return -1;
  }

// 2. Parallel calculation of the matrix (upper triangle + diagonal)
#pragma omp parallel for schedule(dynamic) \
    shared(global_error_flag, first_errno)
  for (int i = 0; i < n; ++i) {
    // Check if another thread has already reported an error
    if (global_error_flag)
      continue;  // Simple way to stop doing work if error occurred

    // Calculate diagonal element K(s_i, s_i)
    double K_ii = wd_shift_kernel_core(strings[i], lengths[i], strings[i],
                                       lengths[i], d, s, weights);

    if (K_ii == ERROR_RETURN_VALUE) {
      int current_errno = errno;  // Capture errno immediately
#pragma omp critical              // Protect access to shared error flags
      {
        if (!global_error_flag) {  // Record only the first error
          global_error_flag = 1;
          first_errno = current_errno;
        }
      }
      continue;  // Skip the rest of this iteration for thread i
    }
    output_matrix[i * n + i] = K_ii;

    // Calculate off-diagonal elements K(s_i, s_j) for j > i
    for (int j = i + 1; j < n; ++j) {
      // Check again for global error before starting expensive computation
      if (global_error_flag)
        break;  // Exit inner loop if error occurred elsewhere

      double K_ij = wd_shift_kernel_core(strings[i], lengths[i], strings[j],
                                         lengths[j], d, s, weights);

      if (K_ij == ERROR_RETURN_VALUE) {
        int current_errno = errno;
#pragma omp critical
        {
          if (!global_error_flag) {
            global_error_flag = 1;
            first_errno = current_errno;
          }
        }
        // No need to continue inner loop for this i if an error happened
        // Or just 'continue;' to the next 'j' might also be valid if errors are
        // isolated. Let's break the inner loop for simplicity upon first error
        // in row i.
        break;
      }

      output_matrix[i * n + j] = K_ij;
      output_matrix[j * n + i] = K_ij;  // Exploit symmetry
    }  // end inner loop (j)
  }  // end parallel for loop (i)

  // 3. Cleanup
  free(weights);

  // 4. Return status
  if (global_error_flag) {
    errno = first_errno;  // Set errno for the calling process based on the
                          // first error found
    return -1;            // Indicate error
  } else {
    errno = 0;  // Ensure errno is 0 on success
    return 0;   // Indicate success
  }
}

int wd_shift_kernel_multiple_omp(
    const char *str, int l, const char **strings, int n, const int *lengths,
    int d, int s,
    double *output_vector  // Pre-allocated buffer of size n
) {
  if (n < 0) {
    errno = EINVAL;
    return -1;  // Indicate error
  }
  if (n == 0) {
    return 0;  // Nothing to do, success
  }
  if (!strings || !lengths || !output_vector) {
    errno = EINVAL;  // Null pointers
    return -1;
  }

  double *weights = NULL;
  int global_error_flag = 0;  // Shared flag to signal errors from threads
  int first_errno = 0;        // Store the first errno encountered by any thread

  weights = (double *)malloc((d + 1) * sizeof(double));
  if (!weights) {
    errno = ENOMEM;
    return -1;
  }
  if (d == 0) {
    errno = EDOM;
    free(weights);
    return -1;
  } else if (d > 0) {  // Ensure d is non-negative
    double normalization_factor = (double)d * (d + 1);
    for (int k = 1; k <= d; ++k) {
      weights[k] = 2.0 * (d - k + 1) / normalization_factor;
    }
  } else {
    errno = EINVAL;
    free(weights);
    return -1;
  }

#pragma omp parallel for schedule(dynamic) \
    shared(global_error_flag, first_errno)
  for (int i = 0; i < n; ++i) {
    if (global_error_flag) continue;
    double K_i =
        wd_shift_kernel_core(str, l, strings[i], lengths[i], d, s, weights);
    if (K_i == ERROR_RETURN_VALUE) {
      int current_errno = errno;  // Capture errno immediately
#pragma omp critical              // Protect access to shared error flags
      {
        if (!global_error_flag) {  // Record only the first error
          global_error_flag = 1;
          first_errno = current_errno;
        }
      }
      continue;  // Skip the rest of this iteration for thread i
    }
    output_vector[i] = K_i;
  }

  free(weights);

  if (global_error_flag) {
    errno = first_errno;  // Set errno for the calling process based on the
                          // first error found
    return -1;            // Indicate error
  } else {
    errno = 0;  // Ensure errno is 0 on success
    return 0;   // Indicate success
  }
}
