OPTION CASEMAP:NONE

PUBLIC __k8_mul_basecase

.code

ALIGN 16

__k8_mul_basecase PROC
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

	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15

	xor	r13d, r13d
	mov	rax, QWORD PTR [rsi]
	mov	r12, QWORD PTR [rcx]

	sub	r13, rdx
	mov	r11, r13
	mov	ebx, edx

	lea	rdi, QWORD PTR [rdi + rdx*8]
	lea	rsi, QWORD PTR [rsi + rdx*8]

	mul	r12

	test	r8b, 1
	jz	Lmul_2

Lmul_1:
	and	ebx, 3
	jz	Lmul_1_prologue_0
	cmp	ebx, 2
	jc	Lmul_1_prologue_1
	jz	Lmul_1_prologue_2

Lmul_1_prologue_3:
	add	r11, -1
	lea	r14, Laddmul_outer_3
	mov	r10, rax
	mov	rbx, rdx
	jmp	Lmul_1_entry_3

Lmul_1_prologue_0:
	mov	rbp, rax
	mov	r10, rdx
	lea	r14, Laddmul_outer_0
	jmp	Lmul_1_entry_0

Lmul_1_prologue_1:
	cmp	r13, -1
	jne	L2_1
	mov	QWORD PTR [rdi - 8], rax
	mov	QWORD PTR [rdi], rdx
	jmp	Lret
L2_1:
	add	r11, 1
	lea	r14, Laddmul_outer_1
	mov	r15, rax
	mov	rbp, rdx
	xor	r10d, r10d
	mov	rax, QWORD PTR [rsi + r11*8]
	jmp	Lmul_1_entry_1

Lmul_1_prologue_2:
	add	r11, -2
	lea	r14, Laddmul_outer_2
	mov	rbx, rax
	mov	r15, rdx
	mov	rax, QWORD PTR [rsi + r11*8 + 24]
	xor	ebp, ebp
	xor	r10d, r10d
	jmp	Lmul_1_entry_2




ALIGN 16
Lmul_1_top:
	mov	QWORD PTR [rdi + r11*8 - 16], rbx
	add	r15, rax
	mov	rax, QWORD PTR [rsi + r11*8]
	adc	rbp, rdx
Lmul_1_entry_1:
	xor	ebx, ebx
	mul	r12
	mov	QWORD PTR [rdi + r11*8 - 8], r15
	add	rbp, rax
	adc	r10, rdx
Lmul_1_entry_0:
	mov	rax, QWORD PTR [rsi + r11*8 + 8]
	mul	r12
	mov	QWORD PTR [rdi + r11*8], rbp
	add	r10, rax
	adc	rbx, rdx
Lmul_1_entry_3:
	mov	rax, QWORD PTR [rsi + r11*8 + 16]
	mul	r12
	mov	QWORD PTR [rdi + r11*8 + 8], r10
	xor	ebp, ebp
	mov	r10, rbp
	add	rbx, rax
	mov	rax, QWORD PTR [rsi + r11*8 + 24]
	mov	r15, rbp
	adc	r15, rdx
Lmul_1_entry_2:
	mul	r12
	add	r11, 4
	js	Lmul_1_top

	mov	QWORD PTR [rdi - 16], rbx
	add	r15, rax
	mov	QWORD PTR [rdi - 8], r15
	adc	rbp, rdx
	mov	QWORD PTR [rdi], rbp

	add	r8, -1
	jz	Lret

	mov	r12, QWORD PTR [rcx + 8]
	mov	r9, QWORD PTR [rcx + 16]

	lea	rcx, QWORD PTR [rcx + 8]
	lea	rdi, QWORD PTR [rdi + 8]

	jmp	r14




ALIGN 16
Lmul_2:
	mov	r9, QWORD PTR [rcx + 8]

	and	ebx, 3
	jz	Lmul_2_prologue_0
	cmp	ebx, 2
	jz	Lmul_2_prologue_2
	jc	Lmul_2_prologue_1

Lmul_2_prologue_3:
	lea	r14, Laddmul_outer_3
	add	r11, 2
	mov	QWORD PTR [rdi + r11*8 - 16], rax
	mov	rbp, rdx
	xor	r10d, r10d
	xor	ebx, ebx
	mov	rax, QWORD PTR [rsi + r11*8 - 16]
	jmp	Lmul_2_entry_3

ALIGN 16
Lmul_2_prologue_0:
	add	r11, 3
	mov	rbx, rax
	mov	r15, rdx
	xor	ebp, ebp
	mov	rax, QWORD PTR [rsi + r11*8 - 24]
	lea	r14, Laddmul_outer_0
	jmp	Lmul_2_entry_0

ALIGN 16
Lmul_2_prologue_1:
	mov	r10, rax
	mov	rbx, rdx
	xor	r15d, r15d
	lea	r14, Laddmul_outer_1
	jmp	Lmul_2_entry_1

ALIGN 16
Lmul_2_prologue_2:
	add	r11, 1
	lea	r14, Laddmul_outer_2
	mov	ebx, 0
	mov	r15d, 0
	mov	rbp, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 8]
	mov	r10, rdx
	jmp	Lmul_2_entry_2



ALIGN 16
Lmul_2_top:
	mov	rax, QWORD PTR [rsi + r11*8 - 32]
	mul	r9
	add	rbx, rax
	adc	r15, rdx
	mov	rax, QWORD PTR [rsi + r11*8 - 24]
	xor	ebp, ebp
	mul	r12
	add	rbx, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 24]
	adc	r15, rdx
	adc	ebp, 0
Lmul_2_entry_0:
	mul	r9
	add	r15, rax
	mov	QWORD PTR [rdi + r11*8 - 24], rbx
	adc	rbp, rdx
	mov	rax, QWORD PTR [rsi + r11*8 - 16]
	mul	r12
	mov	r10d, 0
	add	r15, rax
	adc	rbp, rdx
	mov	rax, QWORD PTR [rsi + r11*8 - 16]
	adc	r10d, 0
	mov	ebx, 0
	mov	QWORD PTR [rdi + r11*8 - 16], r15
Lmul_2_entry_3:
	mul	r9
	add	rbp, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 8]
	adc	r10, rdx
	mov	r15d, 0
	mul	r12
	add	rbp, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 8]
	adc	r10, rdx
	adc	ebx, r15d
Lmul_2_entry_2:
	mul	r9
	add	r10, rax
	mov	QWORD PTR [rdi + r11*8 - 8], rbp
	adc	rbx, rdx
	mov	rax, QWORD PTR [rsi + r11*8]
	mul	r12
	add	r10, rax
	adc	rbx, rdx
	adc	r15d, 0
Lmul_2_entry_1:
	add	r11, 4
	mov	QWORD PTR [rdi + r11*8 - 32], r10
	js	Lmul_2_top

	mov	rax, QWORD PTR [rsi + r11*8 - 32]
	mul	r9
	add	rbx, rax
	mov	QWORD PTR [rdi], rbx
	adc	r15, rdx
	mov	QWORD PTR [rdi + 8], r15

	add	r8, -2
	jz	Lret

	mov	r12, QWORD PTR [rcx + 16]
	mov	r9, QWORD PTR [rcx + 24]

	lea	rcx, QWORD PTR [rcx + 16]
	lea	rdi, QWORD PTR [rdi + 16]

	jmp	r14








Laddmul_outer_0:
	add	r13, 3
	lea	r14, $

	mov	r11, r13
	mov	rax, QWORD PTR [rsi + r13*8 - 24]
	mul	r12
	mov	rbx, rax
	mov	rax, QWORD PTR [rsi + r13*8 - 24]
	mov	r15, rdx
	xor	ebp, ebp
	jmp	Laddmul_entry_0

Laddmul_outer_1:
	mov	r11, r13
	mov	rax, QWORD PTR [rsi + r13*8]
	mul	r12
	mov	r10, rax
	mov	rax, QWORD PTR [rsi + r13*8]
	mov	rbx, rdx
	xor	r15d, r15d
	jmp	Laddmul_entry_1

Laddmul_outer_2:
	add	r13, 1
	lea	r14, $

	mov	r11, r13
	mov	rax, QWORD PTR [rsi + r13*8 - 8]
	mul	r12
	xor	ebx, ebx
	mov	rbp, rax
	xor	r15d, r15d
	mov	r10, rdx
	mov	rax, QWORD PTR [rsi + r13*8 - 8]
	jmp	Laddmul_entry_2

Laddmul_outer_3:
	add	r13, 2
	lea	r14, $

	mov	r11, r13
	mov	rax, QWORD PTR [rsi + r13*8 - 16]
	xor	r10d, r10d
	mul	r12
	mov	r15, rax
	mov	rax, QWORD PTR [rsi + r13*8 - 16]
	mov	rbp, rdx
	jmp	Laddmul_entry_3



ALIGN 16
Laddmul_top:
	add	QWORD PTR [rdi + r11*8 - 32], r10
	adc	rbx, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 24]
	adc	r15, rdx
	xor	ebp, ebp
	mul	r12
	add	rbx, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 24]
	adc	r15, rdx
	adc	ebp, ebp
Laddmul_entry_0:
	mul	r9
	xor	r10d, r10d
	add	QWORD PTR [rdi + r11*8 - 24], rbx
	adc	r15, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 16]
	adc	rbp, rdx
	mul	r12
	add	r15, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 16]
	adc	rbp, rdx
	adc	r10d, 0
Laddmul_entry_3:
	mul	r9
	add	QWORD PTR [rdi + r11*8 - 16], r15
	adc	rbp, rax
	mov	rax, QWORD PTR [rsi + r11*8 - 8]
	adc	r10, rdx
	mul	r12
	xor	ebx, ebx
	add	rbp, rax
	adc	r10, rdx
	mov	r15d, 0
	mov	rax, QWORD PTR [rsi + r11*8 - 8]
	adc	ebx, r15d
Laddmul_entry_2:
	mul	r9
	add	QWORD PTR [rdi + r11*8 - 8], rbp
	adc	r10, rax
	adc	rbx, rdx
	mov	rax, QWORD PTR [rsi + r11*8]
	mul	r12
	add	r10, rax
	mov	rax, QWORD PTR [rsi + r11*8]
	adc	rbx, rdx
	adc	r15d, 0
Laddmul_entry_1:
	mul	r9
	add	r11, 4
	js	Laddmul_top

	add	QWORD PTR [rdi - 8], r10
	adc	rbx, rax
	mov	QWORD PTR [rdi], rbx
	adc	r15, rdx
	mov	QWORD PTR [rdi + 8], r15

	add	r8, -2
	jz	Lret

	lea	rdi, QWORD PTR [rdi + 16]
	lea	rcx, QWORD PTR [rcx + 16]

	mov	r12, QWORD PTR [rcx]
	mov	r9, QWORD PTR [rcx + 8]

	jmp	r14

ALIGN 16
Lret:
	pop	r15
	pop	r14
	pop	r13
	pop	r12
	pop	rbp
	pop	rbx

	pop     rsi
    pop     rdi
    ret

__k8_mul_basecase ENDP

END
