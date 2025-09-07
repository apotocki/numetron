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
    cmp rcx, 001530                     ; Excavator
    jge amd_15ge
    lea rax, __k8_mul_basecase
    ret
amd_jaguar:
    lea rax, __k8_mul_basecase
    ret
amd_15ge:
    cmp rcx, 001700                     ; less than Zen
    jl amd_jaguar
    lea rax, __alderlake_mul_basecase   ; All Zen
    ret

intel_platform:
    ; Check Intel family 06h
    mov eax, ecx
    and eax, 0FFFF00h               ; vendor + family
    cmp eax, 0010600h               ; Intel + family 06h
    jne intel_non6                  ; Non-06h (e.g., NetBurst) -> k8 kernel
    
    ; family 06h:
    mov eax, ecx
    and eax, 0FFh
    cmp eax, 097h
    jge alderlake_platform
    lea rax, __core2_mul_basecase
    ret

alderlake_platform:
    lea rax, __alderlake_mul_basecase
    ret

intel_non6:
    lea rax, __k8_mul_basecase      ; Non-06h Intel (e.g., family 0Fh)
    ret

atom_platform:
via_platform:
    xor rax, rax                    ; VIA - Unknown
    ret

detect_mul_basecase ENDP
END