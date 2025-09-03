; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2025 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; detect_mul_basecase.asm

EXTERN __k8_mul_basecase:PROC
EXTERN __alderlake_mul_basecase:PROC
EXTERN __core2_mul_basecase:PROC

.CODE
detect_mul_basecase PROC
    ; rcx = platform_descriptor (Microsoft x64 calling convention)

    cmp rcx, 010000h
    jl  amd_platform
    cmp rcx, 020000h
    jl  intel_platform
    cmp rcx, 030000h
    jl  atom_platform
    cmp rcx, 040000h
    jl via_platform
    xor rax, rax                        ; Unknown platform
    ret
    
amd_platform:
    lea rax, __k8_mul_basecase
    ret

intel_platform:
    cmp rcx, 01069Eh ; Core2
    jge core2_platform
    lea rax, __k8_mul_basecase   ; Pre-Core2 Intel
    ret

core2_platform:
    cmp rcx, 010697h ; Alder Lake
    jge alderlake_platform
    lea rax, __core2_mul_basecase
    ret

alderlake_platform:
    lea rax, __alderlake_mul_basecase
    ret

atom_platform:
via_platform:
    xor rax, rax                        ; VIA - Unknown
    ret
detect_mul_basecase ENDP
END