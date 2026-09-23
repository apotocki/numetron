; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2025 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; add_sub_n.asm -- r[0..n) = u[0..n) +/- v[0..n), returning the carry/borrow out (0 or 1).
;
;   uint64_t numetron_add_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n);
;   uint64_t numetron_sub_n(uint64_t* rp, const uint64_t* up, const uint64_t* vp, size_t n);
;
; Microsoft x64 calling convention: rcx = rp, rdx = up, r8 = vp, r9 = n. Leaf functions that
; only touch volatile registers (no pushes), so they need no unwind info.
;
; The carry/borrow stays in CF for the whole call: between the adc/sbb instructions the loops use
; only mov, lea, dec, jnz and jrcxz, none of which write CF. The n % 4 odd limbs are done first,
; one at a time (CF is still clear on entry), then n / 4 blocks of four. Every limb of up and vp
; is read before the same index of rp is written, so rp may coincide with up or vp (in-place use)
; or trail them.

OPTION CASEMAP:NONE

PUBLIC numetron_add_n
PUBLIC numetron_sub_n

.code

ALIGN 16
numetron_add_n PROC
    mov     r11, rcx                    ; rp
    mov     rcx, r9
    mov     eax, r9d
    shr     rcx, 2                      ; rcx = n / 4
    and     eax, 3                      ; eax = n % 4; clears CF
    jz      add_blocks
add_odd:
    mov     r10, QWORD PTR [rdx]
    adc     r10, QWORD PTR [r8]
    mov     QWORD PTR [r11], r10
    lea     rdx, [rdx + 8]
    lea     r8, [r8 + 8]
    lea     r11, [r11 + 8]
    dec     eax
    jnz     add_odd
add_blocks:
    jrcxz   add_done
add_loop:
    mov     r9, QWORD PTR [rdx]
    mov     r10, QWORD PTR [rdx + 8]
    adc     r9, QWORD PTR [r8]
    adc     r10, QWORD PTR [r8 + 8]
    mov     QWORD PTR [r11], r9
    mov     QWORD PTR [r11 + 8], r10
    mov     r9, QWORD PTR [rdx + 16]
    mov     r10, QWORD PTR [rdx + 24]
    adc     r9, QWORD PTR [r8 + 16]
    adc     r10, QWORD PTR [r8 + 24]
    mov     QWORD PTR [r11 + 16], r9
    mov     QWORD PTR [r11 + 24], r10
    lea     rdx, [rdx + 32]
    lea     r8, [r8 + 32]
    lea     r11, [r11 + 32]
    dec     rcx
    jnz     add_loop
add_done:
    mov     eax, 0                      ; mov, not xor: xor would clear CF
    adc     eax, eax                    ; rax = CF
    ret
numetron_add_n ENDP

ALIGN 16
numetron_sub_n PROC
    mov     r11, rcx                    ; rp
    mov     rcx, r9
    mov     eax, r9d
    shr     rcx, 2                      ; rcx = n / 4
    and     eax, 3                      ; eax = n % 4; clears CF
    jz      sub_blocks
sub_odd:
    mov     r10, QWORD PTR [rdx]
    sbb     r10, QWORD PTR [r8]
    mov     QWORD PTR [r11], r10
    lea     rdx, [rdx + 8]
    lea     r8, [r8 + 8]
    lea     r11, [r11 + 8]
    dec     eax
    jnz     sub_odd
sub_blocks:
    jrcxz   sub_done
sub_loop:
    mov     r9, QWORD PTR [rdx]
    mov     r10, QWORD PTR [rdx + 8]
    sbb     r9, QWORD PTR [r8]
    sbb     r10, QWORD PTR [r8 + 8]
    mov     QWORD PTR [r11], r9
    mov     QWORD PTR [r11 + 8], r10
    mov     r9, QWORD PTR [rdx + 16]
    mov     r10, QWORD PTR [rdx + 24]
    sbb     r9, QWORD PTR [r8 + 16]
    sbb     r10, QWORD PTR [r8 + 24]
    mov     QWORD PTR [r11 + 16], r9
    mov     QWORD PTR [r11 + 24], r10
    lea     rdx, [rdx + 32]
    lea     r8, [r8 + 32]
    lea     r11, [r11 + 32]
    dec     rcx
    jnz     sub_loop
sub_done:
    mov     eax, 0                      ; mov, not xor: xor would clear CF
    adc     eax, eax                    ; rax = CF (the borrow)
    ret
numetron_sub_n ENDP

END
