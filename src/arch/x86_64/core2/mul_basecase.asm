OPTION CASEMAP:NONE

PUBLIC __core2_mul_basecase

.code
ALIGN 16

__core2_mul_basecase PROC
    ; Save non-volatile we clobber in the small paths
    push    rdi
    push    rsi

    ; Map Win64 args to the original register convention:
    ; rdi=rp, rsi=up, rdx=un, rcx=vp  (vn stays on stack)
    mov     rdi, rcx          ; rp
    mov     rsi, rdx          ; up
    mov     rdx, r8           ; un
    mov     rcx, r9           ; vp
    mov     r8, QWORD PTR  [rsp+56] ; vn

    mov	rax, QWORD PTR [rsi]
    push	rbx
    push	rbp
    push	r12
    push	r13
    push	r14

    mov	r10, QWORD PTR [rcx]

    xor	r9, r9
    sub	r9, rdx

    lea	rsi, QWORD PTR [rsi + rdx*8]
    lea	rdi, QWORD PTR [rdi + rdx*8]

    mul	r10

    test	r8b, 1
    jz	Lm2

    lea	r14, QWORD PTR [rcx + 8]

    test	r9b, 1
    jnz	Lm1x1

Lm1x0:
    test	r9b, 2
    jnz	Lm1s2

Lm1s0:
    lea	r13, QWORD PTR [r9]
    mov	QWORD PTR [rdi + r9*8], rax
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    mov	rbx, rdx
    lea	rbp, Ldo_am0
    jmp	Lm1e0

Lm1s2:
    lea	r13, QWORD PTR [r9 + 2]
    mov	QWORD PTR [rdi + r9*8], rax
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    mov	rbx, rdx
    mul	r10
    lea	rbp, Ldo_am2
    test	r13, r13
    jnz	Lm1e2
    add	rbx, rax
    adc	rdx, 0
    mov	QWORD PTR [rdi - 8], rbx
    mov	QWORD PTR [rdi], rdx
    jmp	Lret2

Lm1x1:
    test	r9b, 2
    jz	Lm1s3

Lm1s1:
    lea	r13, QWORD PTR [r9 + 1]
    mov	QWORD PTR [rdi + r9*8], rax
    test	r13, r13
    jz	L1
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    mov	rcx, rdx
    lea	rbp, Ldo_am1
    jmp	Lm1e1
L1:
    mov	QWORD PTR [rdi], rdx
    jmp	Lret2

Lm1s3:
    lea	r13, QWORD PTR [r9 - 1]
    mov	QWORD PTR [rdi + r9*8], rax
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    mov	rcx, rdx
    lea	rbp, Ldo_am3
    jmp	Lm1e3

ALIGN 16
Lm1top:
    mul	r10
    mov	QWORD PTR [rdi + r13*8 - 16], rcx
Lm1e2:
    xor	ecx, ecx
    add	rbx, rax
    mov	rax, QWORD PTR [rsi + r13*8]
    adc	rcx, rdx
    mov	QWORD PTR [rdi + r13*8 - 8], rbx
Lm1e1:
    xor	ebx, ebx
    mul	r10
    add	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    adc	rbx, rdx
    mov	QWORD PTR [rdi + r13*8], rcx
Lm1e0:
    xor	ecx, ecx
    mul	r10
    add	rbx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    adc	rcx, rdx
    mov	QWORD PTR [rdi + r13*8 + 8], rbx
Lm1e3:
    xor	ebx, ebx
    mul	r10
    add	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    adc	rbx, rdx
    add	r13, 4
    js	Lm1top

    mul	r10
    mov	QWORD PTR [rdi - 16], rcx
    add	rbx, rax
    adc	rdx, 0
    mov	QWORD PTR [rdi - 8], rbx
    mov	QWORD PTR [rdi], rdx

    dec	r8
    jz	Lret2
    lea	rdi, QWORD PTR [rdi - 8]
    jmp	rbp

Lm2:
    mov	r11, QWORD PTR [rcx + 8]
    lea	r14, QWORD PTR [rcx + 16]

    test	r9b, 1
    jnz	Lbx1

Lbx0:
    test	r9b, 2
    jnz	Lb10

Lb00:
    lea	r13, QWORD PTR [r9]
    mov	QWORD PTR [rdi + r9*8], rax
    mov	rcx, rdx
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	ebp, 0
    jmp	Lm2e0

Lb10:
    lea	r13, QWORD PTR [r9 - 2]
    mov	rbp, rax
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	r12, rdx
    mov	ebx, 0
    jmp	Lm2e2

Lbx1:
    test	r9b, 2
    jz	Lb11

Lb01:
    lea	r13, QWORD PTR [r9 + 1]
    mov	QWORD PTR [rdi + r9*8], rax
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	rbx, rdx
    mov	ecx, 0
    jmp	Lm2e1

Lb11:
    lea	r13, QWORD PTR [r9 - 1]
    mov	rcx, rax
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	rbp, rdx
    mov	r12d, 0
    jmp	Lm2e3

ALIGN 16
Lm2top0:
    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 - 8]
    mov	QWORD PTR [rdi + r13*8 - 8], r12
    adc	rbx, rdx
    adc	ecx, 0
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	ebp, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r10
    add	rbx, rax
    mov	QWORD PTR [rdi + r13*8], rbx
    adc	rcx, rdx
    mov	rax, QWORD PTR [rsi + r13*8]
    adc	ebp, 0
Lm2e0:
    mul	r11
    add	rcx, rax
    adc	rbp, rdx
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    mov	r12d, 0
    add	rcx, rax
    adc	rbp, rdx
    adc	r12d, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r11
    add	rbp, rax
    mov	QWORD PTR [rdi + r13*8 + 8], rcx
    adc	r12, rdx
    mov	ebx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    adc	r12, rdx
    adc	ebx, 0
    mul	r11
    mov	ecx, 0
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	QWORD PTR [rdi + r13*8 + 16], rbp
    adc	rbx, rdx
    add	r13, 4
    js	Lm2top0

    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi - 8]
    mov	QWORD PTR [rdi - 8], r12
    adc	rbx, rdx
    adc	ecx, ecx
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	QWORD PTR [rdi], rbx
    mov	QWORD PTR [rdi + 8], rcx

    add	r8, -2
    jz	Lret2

Ldo_am0:
    push	r15
    push	r8

Lolo0:
    mov	r10, QWORD PTR [r14]
    mov	r11, QWORD PTR [r14 + 8]
    lea	r14, QWORD PTR [r14 + 16]
    lea	rdi, QWORD PTR [rdi + 16]
    mov	rax, QWORD PTR [rsi + r9*8]

    mov	r13, r9
    mul	r10
    mov	r8, rax
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	r15, rdx
    mul	r11
    lea	rbx, QWORD PTR [rdx]
    mov	rbp, QWORD PTR [rdi + r9*8]
    mov	r12, rax
    jmp	Llo0

ALIGN 16
Lam2top0:
    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8]
    lea	r12, QWORD PTR [rdx]
    adc	r12, 0
    mul	r10
    add	r15, rcx
    mov	QWORD PTR [rdi + r13*8 - 8], r15
    adc	r8, rax
    mov	r15, rdx
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r11
    lea	rbx, QWORD PTR [rdx]
    mov	rcx, QWORD PTR [rdi + r13*8]
    add	rbp, rcx
    adc	r12, rax
    adc	rbx, 0
Llo0:
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    add	r8, rbp
    adc	r15, rax
    mov	QWORD PTR [rdi + r13*8], r8
    mov	r8, rdx
    adc	r8, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mov	rbp, QWORD PTR [rdi + r13*8 + 8]
    mul	r11
    add	r12, rbp
    adc	rbx, rax
    lea	rcx, QWORD PTR [rdx]
    adc	rcx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	r15, r12
    mov	QWORD PTR [rdi + r13*8 + 8], r15
    adc	r8, rax
    mov	r15, rdx
    mov	r12, QWORD PTR [rdi + r13*8 + 16]
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r11
    add	rbx, r12
    lea	rbp, QWORD PTR [rdx]
    adc	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    adc	rbp, 0
    mul	r10
    add	r8, rbx
    mov	QWORD PTR [rdi + r13*8 + 16], r8
    mov	r8, rdx
    adc	r15, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	rbx, QWORD PTR [rdi + r13*8 + 24]
    adc	r8, 0
    add	r13, 4
    jnc	Lam2top0

    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    adc	rdx, r13
    add	r15, rcx
    adc	r8, r13
    mov	QWORD PTR [rdi - 8], r15
    add	r8, rbp
    mov	QWORD PTR [rdi], r8
    adc	rdx, r13
    mov	QWORD PTR [rdi + 8], rdx

    add	DWORD PTR [rsp], -2
    jnz	Lolo0

Lret:
    pop	rax
    pop	r15
Lret2:
    pop	r14
    pop	r13
    pop	r12
    pop	rbp
    pop	rbx

    pop    rsi
    pop    rdi
    ret


ALIGN 16
Lm2top1:
    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 - 8]
    mov	QWORD PTR [rdi + r13*8 - 8], r12
    adc	rbx, rdx
    adc	ecx, 0
Lm2e1:
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	ebp, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r10
    add	rbx, rax
    mov	QWORD PTR [rdi + r13*8], rbx
    adc	rcx, rdx
    mov	rax, QWORD PTR [rsi + r13*8]
    adc	ebp, 0
    mul	r11
    add	rcx, rax
    adc	rbp, rdx
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    mov	r12d, 0
    add	rcx, rax
    adc	rbp, rdx
    adc	r12d, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r11
    add	rbp, rax
    mov	QWORD PTR [rdi + r13*8 + 8], rcx
    adc	r12, rdx
    mov	ebx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    adc	r12, rdx
    adc	ebx, 0
    mul	r11
    mov	ecx, 0
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	QWORD PTR [rdi + r13*8 + 16], rbp
    adc	rbx, rdx
    add	r13, 4
    js	Lm2top1

    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi - 8]
    mov	QWORD PTR [rdi - 8], r12
    adc	rbx, rdx
    adc	ecx, ecx
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	QWORD PTR [rdi], rbx
    mov	QWORD PTR [rdi + 8], rcx

    add	r8, -2
    jz	Lret2

Ldo_am1:
    push	r15
    push	r8

Lolo1:
    mov	r10, QWORD PTR [r14]
    mov	r11, QWORD PTR [r14 + 8]
    lea	r14, QWORD PTR [r14 + 16]
    lea	rdi, QWORD PTR [rdi + 16]
    mov	rax, QWORD PTR [rsi + r9*8]
    lea	r13, QWORD PTR [r9 + 1]
    mul	r10
    mov	r15, rax
    mov	r8, rdx
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	rcx, QWORD PTR [rdi + r9*8]
    mul	r11
    mov	rbp, rax
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    lea	r12, QWORD PTR [rdx]
    jmp	Llo1

ALIGN 16
Lam2top1:
    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8]
    lea	r12, QWORD PTR [rdx]
    adc	r12, 0
Llo1:
    mul	r10
    add	r15, rcx
    mov	QWORD PTR [rdi + r13*8 - 8], r15
    adc	r8, rax
    mov	r15, rdx
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r11
    lea	rbx, QWORD PTR [rdx]
    mov	rcx, QWORD PTR [rdi + r13*8]
    add	rbp, rcx
    adc	r12, rax
    adc	rbx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    add	r8, rbp
    adc	r15, rax
    mov	QWORD PTR [rdi + r13*8], r8
    mov	r8, rdx
    adc	r8, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mov	rbp, QWORD PTR [rdi + r13*8 + 8]
    mul	r11
    add	r12, rbp
    adc	rbx, rax
    lea	rcx, QWORD PTR [rdx]
    adc	rcx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	r15, r12
    mov	QWORD PTR [rdi + r13*8 + 8], r15
    adc	r8, rax
    mov	r15, rdx
    mov	r12, QWORD PTR [rdi + r13*8 + 16]
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r11
    add	rbx, r12
    lea	rbp, QWORD PTR [rdx]
    adc	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    adc	rbp, 0
    mul	r10
    add	r8, rbx
    mov	QWORD PTR [rdi + r13*8 + 16], r8
    mov	r8, rdx
    adc	r15, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	rbx, QWORD PTR [rdi + r13*8 + 24]
    adc	r8, 0
    add	r13, 4
    jnc	Lam2top1

    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    adc	rdx, r13
    add	r15, rcx
    adc	r8, r13
    mov	QWORD PTR [rdi - 8], r15
    add	r8, rbp
    mov	QWORD PTR [rdi], r8
    adc	rdx, r13
    mov	QWORD PTR [rdi + 8], rdx

    add	DWORD PTR [rsp], -2
    jnz	Lolo1

    pop	rax
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	rbp
    pop	rbx

    pop    rsi
    pop    rdi
    ret


ALIGN 16
Lm2top2:
    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 - 8]
    mov	QWORD PTR [rdi + r13*8 - 8], r12
    adc	rbx, rdx
    adc	ecx, 0
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	ebp, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r10
    add	rbx, rax
    mov	QWORD PTR [rdi + r13*8], rbx
    adc	rcx, rdx
    mov	rax, QWORD PTR [rsi + r13*8]
    adc	ebp, 0
    mul	r11
    add	rcx, rax
    adc	rbp, rdx
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    mov	r12d, 0
    add	rcx, rax
    adc	rbp, rdx
    adc	r12d, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r11
    add	rbp, rax
    mov	QWORD PTR [rdi + r13*8 + 8], rcx
    adc	r12, rdx
    mov	ebx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    adc	r12, rdx
    adc	ebx, 0
Lm2e2:
    mul	r11
    mov	ecx, 0
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	QWORD PTR [rdi + r13*8 + 16], rbp
    adc	rbx, rdx
    add	r13, 4
    js	Lm2top2

    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi - 8]
    mov	QWORD PTR [rdi - 8], r12
    adc	rbx, rdx
    adc	ecx, ecx
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	QWORD PTR [rdi], rbx
    mov	QWORD PTR [rdi + 8], rcx

    add	r8, -2
    jz	Lret2

Ldo_am2:
    push	r15
    push	r8

Lolo2:
    mov	r10, QWORD PTR [r14]
    mov	r11, QWORD PTR [r14 + 8]
    lea	r14, QWORD PTR [r14 + 16]
    lea	rdi, QWORD PTR [rdi + 16]
    mov	rax, QWORD PTR [rsi + r9*8]
    lea	r13, QWORD PTR [r9 - 2]
    mul	r10
    mov	r8, rax
    mov	r15, rdx
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	rbx, QWORD PTR [rdi + r9*8]
    mul	r11
    mov	rcx, rax
    lea	rbp, QWORD PTR [rdx]
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    jmp	Llo2

ALIGN 16
Lam2top2:
    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8]
    lea	r12, QWORD PTR [rdx]
    adc	r12, 0
    mul	r10
    add	r15, rcx
    mov	QWORD PTR [rdi + r13*8 - 8], r15
    adc	r8, rax
    mov	r15, rdx
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r11
    lea	rbx, QWORD PTR [rdx]
    mov	rcx, QWORD PTR [rdi + r13*8]
    add	rbp, rcx
    adc	r12, rax
    adc	rbx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    add	r8, rbp
    adc	r15, rax
    mov	QWORD PTR [rdi + r13*8], r8
    mov	r8, rdx
    adc	r8, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mov	rbp, QWORD PTR [rdi + r13*8 + 8]
    mul	r11
    add	r12, rbp
    adc	rbx, rax
    lea	rcx, QWORD PTR [rdx]
    adc	rcx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	r15, r12
    mov	QWORD PTR [rdi + r13*8 + 8], r15
    adc	r8, rax
    mov	r15, rdx
    mov	r12, QWORD PTR [rdi + r13*8 + 16]
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r11
    add	rbx, r12
    lea	rbp, QWORD PTR [rdx]
    adc	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    adc	rbp, 0
Llo2:
    mul	r10
    add	r8, rbx
    mov	QWORD PTR [rdi + r13*8 + 16], r8
    mov	r8, rdx
    adc	r15, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	rbx, QWORD PTR [rdi + r13*8 + 24]
    adc	r8, 0
    add	r13, 4
    jnc	Lam2top2

    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    adc	rdx, r13
    add	r15, rcx
    adc	r8, r13
    mov	QWORD PTR [rdi - 8], r15
    add	r8, rbp
    mov	QWORD PTR [rdi], r8
    adc	rdx, r13
    mov	QWORD PTR [rdi + 8], rdx

    add	DWORD PTR [rsp], -2
    jnz	Lolo2

    pop	rax
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	rbp
    pop	rbx

    pop    rsi
    pop    rdi
    ret


ALIGN 16
Lm2top3:
    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 - 8]
    mov	QWORD PTR [rdi + r13*8 - 8], r12
    adc	rbx, rdx
    adc	ecx, 0
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	ebp, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r10
    add	rbx, rax
    mov	QWORD PTR [rdi + r13*8], rbx
    adc	rcx, rdx
    mov	rax, QWORD PTR [rsi + r13*8]
    adc	ebp, 0
    mul	r11
    add	rcx, rax
    adc	rbp, rdx
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    mov	r12d, 0
    add	rcx, rax
    adc	rbp, rdx
    adc	r12d, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
Lm2e3:
    mul	r11
    add	rbp, rax
    mov	QWORD PTR [rdi + r13*8 + 8], rcx
    adc	r12, rdx
    mov	ebx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r10
    add	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    adc	r12, rdx
    adc	ebx, 0
    mul	r11
    mov	ecx, 0
    add	r12, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	QWORD PTR [rdi + r13*8 + 16], rbp
    adc	rbx, rdx
    add	r13, 4
    js	Lm2top3

    mul	r10
    add	r12, rax
    mov	rax, QWORD PTR [rsi - 8]
    mov	QWORD PTR [rdi - 8], r12
    adc	rbx, rdx
    adc	ecx, 0
    mul	r11
    add	rbx, rax
    adc	rcx, rdx
    mov	QWORD PTR [rdi], rbx
    mov	QWORD PTR [rdi + 8], rcx

    add	r8, -2
    jz	Lret2

Ldo_am3:
    push	r15
    push	r8

Lolo3:
    mov	r10, QWORD PTR [r14]
    mov	r11, QWORD PTR [r14 + 8]
    lea	r14, QWORD PTR [r14 + 16]
    lea	rdi, QWORD PTR [rdi + 16]
    mov	rax, QWORD PTR [rsi + r9*8]
    lea	r13, QWORD PTR [r9 - 1]
    mul	r10
    mov	r15, rax
    mov	r8, rdx
    mov	rax, QWORD PTR [rsi + r9*8]
    mov	r12, QWORD PTR [rdi + r9*8]
    mul	r11
    mov	rbx, rax
    lea	rcx, QWORD PTR [rdx]
    mov	rax, QWORD PTR [rsi + r9*8 + 8]
    jmp	Llo3

ALIGN 16
Lam2top3:
    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    mov	rax, QWORD PTR [rsi + r13*8]
    lea	r12, QWORD PTR [rdx]
    adc	r12, 0
    mul	r10
    add	r15, rcx
    mov	QWORD PTR [rdi + r13*8 - 8], r15
    adc	r8, rax
    mov	r15, rdx
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8]
    mul	r11
    lea	rbx, QWORD PTR [rdx]
    mov	rcx, QWORD PTR [rdi + r13*8]
    add	rbp, rcx
    adc	r12, rax
    adc	rbx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mul	r10
    add	r8, rbp
    adc	r15, rax
    mov	QWORD PTR [rdi + r13*8], r8
    mov	r8, rdx
    adc	r8, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 8]
    mov	rbp, QWORD PTR [rdi + r13*8 + 8]
    mul	r11
    add	r12, rbp
    adc	rbx, rax
    lea	rcx, QWORD PTR [rdx]
    adc	rcx, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
Llo3:
    mul	r10
    add	r15, r12
    mov	QWORD PTR [rdi + r13*8 + 8], r15
    adc	r8, rax
    mov	r15, rdx
    mov	r12, QWORD PTR [rdi + r13*8 + 16]
    adc	r15, 0
    mov	rax, QWORD PTR [rsi + r13*8 + 16]
    mul	r11
    add	rbx, r12
    lea	rbp, QWORD PTR [rdx]
    adc	rcx, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    adc	rbp, 0
    mul	r10
    add	r8, rbx
    mov	QWORD PTR [rdi + r13*8 + 16], r8
    mov	r8, rdx
    adc	r15, rax
    mov	rax, QWORD PTR [rsi + r13*8 + 24]
    mov	rbx, QWORD PTR [rdi + r13*8 + 24]
    adc	r8, 0
    add	r13, 4
    jnc	Lam2top3

    mul	r11
    add	rcx, rbx
    adc	rbp, rax
    adc	rdx, r13
    add	r15, rcx
    adc	r8, r13
    mov	QWORD PTR [rdi - 8], r15
    add	r8, rbp
    mov	QWORD PTR [rdi], r8
    adc	rdx, r13
    mov	QWORD PTR [rdi + 8], rdx

    add	DWORD PTR [rsp], -2
    jnz	Lolo3

    pop	rax
    pop	r15
    pop	r14
    pop	r13
    pop	r12
    pop	rbp
    pop	rbx

    pop     rsi
    pop     rdi
    ret
__core2_mul_basecase ENDP

END
