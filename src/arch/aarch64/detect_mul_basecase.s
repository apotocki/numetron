# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2025 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

    .text
    .align  2

    .globl  detect_mul_basecase
    .type   detect_mul_basecase, %function

detect_mul_basecase:
    mov     x0, #0
    ret

    .size   detect_mul_basecase, .-detect_mul_basecase
    .section .note.GNU-stack,"",@progbits
