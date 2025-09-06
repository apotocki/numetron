OPTION CASEMAP:NONE

PUBLIC __alderlake_mul_basecase

.code

ALIGN 16

; extern "C" void __alderlake_mul_basecase(uint64_t* rp, uint64_t* up, size_t un, uint64_t* vp, uint64_t vn);
; rp in rcx, up in rdx, un in r8, vp in r9, vn on stack

__alderlake_mul_basecase PROC
    ; Save non-volatile we clobber in the small paths
    push    rdi
    push    rsi

	; Map Win64 args to the original register convention:
    ; rdi=rp, rsi=up, rdx=un, rcx=vp, vn=r8
    mov     rdi, rcx          ; rp
    mov     rsi, rdx          ; up
    mov     rdx, r8           ; un
    mov     rcx, r9           ; vp
	mov     r8, QWORD PTR  [rsp+56] ; vn

	cmp	rdx, 2
	ja	Lgen
	mov	rdx, QWORD PTR [rcx]
	mulx r9, rax, QWORD PTR [rsi]
	mov	QWORD PTR [rdi], rax
	je	Ls2x

	mov	QWORD PTR [rdi + 8], r9
	pop     rsi
    pop     rdi
    ret

Ls2x:
	mulx r10, rax, QWORD PTR [rsi + 8]
	add	rax, r9
	adc	r10, 0
	cmp	r8d, 2
	je	Ls22

Ls21:
	mov	QWORD PTR [rdi + 8], rax
	mov	QWORD PTR [rdi + 16], r10
	pop     rsi
    pop     rdi
    ret

Ls22:
	mov	rdx, QWORD PTR [rcx + 8]
	mulx r9, r8, QWORD PTR [rsi]
	add	rax, r8
	adc	r9, r10
	mov	QWORD PTR [rdi + 8], rax
	mulx r10, rax, QWORD PTR [rsi + 8]
	adc	r10, 0
	adc	rax, r9
	mov	QWORD PTR [rdi + 16], rax
	adc	r10, 0
	mov	QWORD PTR [rdi + 24], r10
	pop rsi
    pop rdi
    ret

Lgen:
	push rbx
	push rbp
	push r12
	push r13
	push r14
	push r15

	mov	r14, rdx
	neg	r14
	shl	r14, 3
	mov	r15, rcx
	mov	rbp, r8

	test dl, 1
	mov	rdx, QWORD PTR [r15]
	jz	Lbx0

Lbx1:
	test r14b, 16
	jnz	Lb01

Lb11:
	lea	rcx, QWORD PTR [r14 + 24]
	mulx	r10, r11, QWORD PTR [rsi]
	mulx	r12, r13, QWORD PTR [rsi + 8]
	mulx	rax, rbx, QWORD PTR [rsi + 16]
	lea	rdi, QWORD PTR [rdi + 8]
	lea	rsi, QWORD PTR [rsi + 24]
	jrcxz	Lmed3

Lmtp3:
	mulx r8, r9, QWORD PTR [rsi]
	adcx r13, r10
	mov	QWORD PTR [rdi - 8], r11
	mulx r10, r11, QWORD PTR [rsi + 8]
	adcx rbx, r12
	mov	QWORD PTR [rdi], r13
	mulx r12, r13, QWORD PTR [rsi + 16]
	adcx r9, rax
	mov	QWORD PTR [rdi + 8], rbx
	mulx rax, rbx, QWORD PTR [rsi + 24]
	adcx r11, r8
	mov	QWORD PTR [rdi + 16], r9
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz Lmed3
	jmp	Lmtp3

Lmed3:
	adcx r13, r10
	mov	QWORD PTR [rdi - 8], r11
	adcx rbx, r12
	mov	QWORD PTR [rdi], r13
	adcx rax, rcx
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jz Lret

Lout3:
	lea	rdi, QWORD PTR [rdi + r14 + 32]
	lea	rsi, QWORD PTR [rsi + r14 + 24]
	lea	r15, QWORD PTR [r15 + 8]
; xor	%edx, %edx
	mov	rdx, QWORD PTR [r15]

	mulx r10, r11, QWORD PTR [rsi - 24]
	mulx r12, r13, QWORD PTR [rsi - 16]
	mulx rax, rbx, QWORD PTR [rsi - 8]

	lea	rcx, QWORD PTR [r14 + 24]
	adox r11, QWORD PTR [rdi - 8]
	jrcxz Led3

Ltp3:
	mulx r8, r9, QWORD PTR [rsi]
	adcx r13, r10
	mov	QWORD PTR [rdi - 8], r11
	adox r13, QWORD PTR [rdi]
	mulx r10, r11, QWORD PTR [rsi + 8]
	adcx rbx, r12
	mov	QWORD PTR [rdi], r13
	adox rbx, QWORD PTR [rdi + 8]
	mulx r12, r13, QWORD PTR [rsi + 16]
	adcx r9, rax
	mov	QWORD PTR [rdi + 8], rbx
	adox r9, QWORD PTR [rdi + 16]
	mulx rax, rbx, QWORD PTR [rsi + 24]
	adcx r11, r8
	mov	QWORD PTR [rdi + 16], r9
	adox r11, QWORD PTR [rdi + 24]
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz Led3
	jmp	Ltp3

Led3:
	adcx r13, r10
	mov	QWORD PTR [rdi - 8], r11
	adox r13, QWORD PTR [rdi]
	adcx rbx, r12
	mov	QWORD PTR [rdi], r13
	adox rbx, QWORD PTR [rdi + 8]
	adcx rax, rcx
	adox rax, rcx
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp ; vn--
	jnz	Lout3
	jmp	Lret


Lb01:
DB	196, 226, 227, 246, 6
	lea	rcx, QWORD PTR [r14 + 8]
DB	196, 98, 179, 246, 70, 8
DB	196, 98, 163, 246, 86, 16
	lea	rsi, QWORD PTR [rsi + 8]
	lea	rdi, QWORD PTR [rdi - 8]
	jmp	Lml1
Lmtp1:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
Lml1:
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz Lmed1
	jmp	Lmtp1
Lmed1:
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	102, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jz	Lret
Lout1:
	lea	rdi, QWORD PTR [rdi + r14 + 16]
	lea	rsi, QWORD PTR [rsi + r14 + 8]
	lea	r15, QWORD PTR [r15 + 8]
	xor	edx, edx
	mov	rdx, QWORD PTR [r15]
	lea	rcx, QWORD PTR [r14 + 8]
DB	196, 226, 227, 246, 70, 248
DB	196, 98, 179, 246, 6
DB	196, 98, 163, 246, 86, 8
	jmp	Llo1
Ltp1:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	243, 76, 15, 56, 246, 47
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
Llo1:
DB	243, 72, 15, 56, 246, 95, 8
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	243, 76, 15, 56, 246, 79, 16
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
DB	243, 76, 15, 56, 246, 95, 24
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Led1
	jmp	Ltp1
Led1:
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	243, 76, 15, 56, 246, 47
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	243, 72, 15, 56, 246, 95, 8
DB	102, 72, 15, 56, 246, 193
DB	243, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jnz	Lout1
	jmp	Lret

Lbx0:
	test	r14b, 16
	jz	Lb00

Lb10:
DB	196, 98, 147, 246, 38
DB	196, 226, 227, 246, 70, 8
	lea	rcx, QWORD PTR [r14 + 16]
DB	196, 98, 179, 246, 70, 16
	lea	rsi, QWORD PTR [rsi + 16]
	jmp	Lml2
Lmtp2:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
Lml2:
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Lmed2
	jmp	Lmtp2
Lmed2:
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	102, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jz	Lret
Lout2:
	lea	rdi, QWORD PTR [rdi + r14 + 24]
	lea	rsi, QWORD PTR [rsi + r14 + 16]
	lea	r15, QWORD PTR [r15 + 8]
	xor	edx, edx
	mov	rdx, QWORD PTR [r15]
DB	196, 98, 147, 246, 102, 240
DB	196, 226, 227, 246, 70, 248
	lea	rcx, QWORD PTR [r14 + 16]
DB	196, 98, 179, 246, 6
	jmp	Llo2
Ltp2:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
Llo2:
DB	243, 76, 15, 56, 246, 47
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	243, 72, 15, 56, 246, 95, 8
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	243, 76, 15, 56, 246, 79, 16
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
DB	243, 76, 15, 56, 246, 95, 24
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Led2
	jmp	Ltp2
Led2:
	adcx	r13, r10
	mov	QWORD PTR [rdi - 8], r11
	adox	r13, QWORD PTR [rdi]
	adcx	rbx, r12
	mov	QWORD PTR [rdi], r13
DB	243, 72, 15, 56, 246, 95, 8
DB	102, 72, 15, 56, 246, 193
DB	243, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jnz	Lout2
	jmp	Lret


Lb00:
	lea	rcx, QWORD PTR [r14 + 32]
DB	196, 98, 179, 246, 6
DB	196, 98, 163, 246, 86, 8
DB	196, 98, 147, 246, 102, 16
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi], r9
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 16]
	jrcxz	Lmed0
Lmtp0:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Lmed0
	jmp	Lmtp0
Lmed0:
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	102, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jz	Lret
Lout0:
	lea	rdi, QWORD PTR [rdi + r14 + 40]
	lea	rsi, QWORD PTR [rsi + r14 + 32]
	lea	r15, QWORD PTR [r15 + 8]
	xor	edx, edx
	mov	rdx, QWORD PTR [r15]
	lea	rcx, QWORD PTR [r14 + 32]
DB	196, 98, 179, 246, 70, 224
DB	196, 98, 163, 246, 86, 232
DB	196, 98, 147, 246, 102, 240
DB	243, 76, 15, 56, 246, 79, 240
DB	196, 226, 227, 246, 70, 248
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi - 16], r9
DB	243, 76, 15, 56, 246, 95, 248
	jrcxz	Led0
Ltp0:
DB	196, 98, 179, 246, 6
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	243, 76, 15, 56, 246, 47
DB	196, 98, 163, 246, 86, 8
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	243, 72, 15, 56, 246, 95, 8
DB	196, 98, 147, 246, 102, 16
DB	102, 76, 15, 56, 246, 200
	mov	QWORD PTR [rdi + 8], rbx
DB	243, 76, 15, 56, 246, 79, 16
DB	196, 226, 227, 246, 70, 24
DB	102, 77, 15, 56, 246, 216
	mov	QWORD PTR [rdi + 16], r9
DB	243, 76, 15, 56, 246, 95, 24
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Led0
	jmp	Ltp0
Led0:
DB	102, 77, 15, 56, 246, 234
	mov	QWORD PTR [rdi - 8], r11
DB	243, 76, 15, 56, 246, 47
DB	102, 73, 15, 56, 246, 220
	mov	QWORD PTR [rdi], r13
DB	243, 72, 15, 56, 246, 95, 8
DB	102, 72, 15, 56, 246, 193
DB	243, 72, 15, 56, 246, 193
	mov	QWORD PTR [rdi + 8], rbx
	mov	QWORD PTR [rdi + 16], rax
	dec	rbp
	jnz	Lout0

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

__alderlake_mul_basecase ENDP

END
