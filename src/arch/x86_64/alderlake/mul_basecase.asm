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
	ja	Lgen ; if un > 2 goto .Lgen
	mov	rdx, QWORD PTR [rcx] ; load v[0] ->%rdx // before that %rdx could be 1 or 2, but flags are preserved and we can use them to know exactly un
	mulx	r9, rax, QWORD PTR [rsi] ; v[0](%rdx) * u[0](%rsi) -> %r9:%rax
	mov	QWORD PTR [rdi], rax ; %rax -> r[0](%rdi)
	je	Ls2x ; if un == 2 goto .Ls2x

	mov	QWORD PTR [rdi + 8], r9 ; %r9 -> r[1](%rdi + 8)
	pop     rsi
    pop     rdi
    ret

Ls2x: ; case un == 2
	mulx	r10, rax, QWORD PTR [rsi + 8] ; v[0](%rdx) * u[1](%rsi + 8) -> %r10:%rax
	add	rax, r9 ; low(v0*u1) + hight(v0*u0) -> carry, %rax
	adc	r10, 0 ; %r10 + carry -> carry, %r10
	cmp	r8d, 2
	je	Ls22 ; if vn == 2 goto .Ls22

Ls21: ; case un == 2, vn == 1
	mov	QWORD PTR [rdi + 8], rax ; %rax -> r[1](%rdi + 8)
	mov	QWORD PTR [rdi + 16], r10 ; %r10 -> r[2](%rdi + 16)
	pop     rsi
    pop     rdi
    ret

Ls22: ; case un == 2, vn == 2
	mov	rdx, QWORD PTR [rcx + 8] ; load v[1] ->%rdx
	mulx	r9, r8, QWORD PTR [rsi] ; v[1](%rdx) * u[0](%rsi) -> %r9:%r8
	add	rax, r8 ; low(v0*u1) + hight(v0*u0) + low(v1*u0) -> carry, %rax
	adc	r9, r10 ; %r9 + carry -> carry, %r9
	mov	QWORD PTR [rdi + 8], rax ; %rax -> r[1](%rdi + 8)
	mulx	r10, rax, QWORD PTR [rsi + 8] ; v[1](%rdx) * u[1](%rsi + 8) -> %r10:%rax
	adc	r10, 0 ; %r10 + carry -> carry, %r10
	adc	rax, r9 ; %rax + carry -> carry, %rax
	mov	QWORD PTR [rdi + 16], rax ; %rax -> r[2](%rdi + 16)
	adc	r10, 0 ; %r10 + carry -> carry, %r10
	mov	QWORD PTR [rdi + 24], r10 ; %r10 -> r[3](%rdi + 24)
	pop     rsi
    pop     rdi
    ret

Lgen:
	push	rbx
	push	rbp
	push	r12
	push	r13
	push	r14
	push	r15
; will be used as temporaries %rbx, %rbp, %r12, %r13, %r14, %r15
	mov	r14, rdx ; un -> %r14
	neg	r14 ; -un -> %r14
	shl	r14, 3 ; -un * 8 -> %r14
	mov	r15, rcx ; vp -> %r15
	mov	rbp, r8 ; vn -> %rbp

	test	dl, 1 ; test if un is odd
	mov	rdx, QWORD PTR [r15] ; load v[0] ->%rdx
	jz	Lbx0 ; if un is even goto .Lbx0

Lbx1: ; un is odd
	test	r14b, 16 ; test second bit of ~un
	jnz	Lb01 ; if second bit of un is 0 goto .Lb01 (un = 5, 9, 13, ...)

Lb11: ; un = bxxx11 (u is odd, second bit of un is 1) (un = 3, 7, 11, 15, ...)
	lea	rcx, QWORD PTR [r14 + 24] ; -un * 8 + 24 -> %rcx
	mulx	r10, r11, QWORD PTR [rsi] ; v[0](%rdx) * u[0](%rsi) -> %r10:%r11
	mulx	r12, r13, QWORD PTR [rsi + 8] ; v[0](%rdx) * u[1](%rsi + 8) -> %r12:%r13
	mulx	rax, rbx, QWORD PTR [rsi + 16] ; v[0](%rdx) * u[2](%rsi + 16) -> %rax:%rbx
	lea	rdi, QWORD PTR [rdi + 8] ; ++rp
	lea	rsi, QWORD PTR [rsi + 24] ; up += 3
	jrcxz	Lmed3 ; if (24 - 8*un == 0) <=> un == 3 goto .Lmed3

Lmtp3: ; un >= 4, but un = bxxx11 => un = 7, 11, 15, ...
	mulx	r8, r9, QWORD PTR [rsi] ; v[0](%rdx) * u[3](%rsi) -> %r8:%r9
	adcx	r13, r10 ; high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	QWORD PTR [rdi - 8], r11 ; low(u0*v0) -> r[0](%rdi - 8)
	mulx	r10, r11, QWORD PTR [rsi + 8] ; v[0](%rdx) * u[4](%rsi + 8) -> %r10:%r11
	adcx	rbx, r12 ; high(u1*v0) + low(u2*v0) + carry -> carry, %rbx
	mov	QWORD PTR [rdi], r13 ; high(u0*v0) + low(u1*v0) -> r[1](%rdi)
	mulx	r12, r13, QWORD PTR [rsi + 16] ; v[0](%rdx) * u[5](%rsi + 16) -> %r12:%r13
	adcx	r9, rax ; high(u2*v0) + carry -> carry, %r9
	mov	QWORD PTR [rdi + 8], rbx ; high(u1*v0) + low(u2*v0) + carry -> r[2](%rdi + 8)
	mulx	rax, rbx, QWORD PTR [rsi + 24] ; v[0](%rdx) * u[6](%rsi + 24) -> %rax:%rbx
	adcx	r11, r8 ; high(u3*v0) + low(u4*v0) + carry -> carry, %r11
	mov	QWORD PTR [rdi + 16], r9 ; high(u2*v0) + carry -> r[3](%rdi + 16)
	lea	rsi, QWORD PTR [rsi + 32] ; up += 4
	lea	rdi, QWORD PTR [rdi + 32] ; rp += 4
	lea	rcx, QWORD PTR [rcx + 32] ; -un * 8 + 24 + 32 -> %rcx
	jrcxz	Lmed3 ; if (24 - 8*un + 32 == 0) <=> un == 7 goto .Lmed3
	jmp	Lmtp3 ; else goto .Lmtp3 // un >= 8, un = bxxx11 => un >= 11

Lmed3: ; un = 3 || un = 7
	adcx	r13, r10 ; high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	QWORD PTR [rdi - 8], r11 ; low(u0*v0) -> r[0](%rdi - 8)
	adcx	rbx, r12 ; high(u1*v0) + low(u2*v0) + carry -> carry, %rbx
	mov	QWORD PTR [rdi], r13 ; high(u0*v0) + low(u1*v0) -> r[1](%rdi)
	adcx	rax, rcx ; high(u2*v0) + carry -> carry, %rax
	mov	QWORD PTR [rdi + 8], rbx ; high(u1*v0) + low(u2*v0) + carry -> r[2](%rdi + 8)
	mov	QWORD PTR [rdi + 16], rax ; high(u2*v0) + carry -> r[3](%rdi + 16)
	dec	rbp ; vn--
	jz	Lret ; if vn == 0 goto .Lret

Lout3: ; vn > 1, un = 3 || un = 7
	lea	rdi, QWORD PTR [rdi + r14 + 32] ; rp += -un * 8 + 32
	lea	rsi, QWORD PTR [rsi + r14 + 24] ; up += -un * 8 + 24
	lea	r15, QWORD PTR [r15 + 8] ; vp += 1
; xor	%edx, %edx				# un(%rdx) = 0
	mov	rdx, QWORD PTR [r15] ; load v[0] ->%rdx

	mulx	r10, r11, QWORD PTR [rsi - 24] ; v[0](%rdx) * u[0](%rsi - 24) -> %r10:%r11
	mulx	r12, r13, QWORD PTR [rsi - 16] ; v[0](%rdx) * u[1](%rsi - 16) -> %r12:%r13
	mulx	rax, rbx, QWORD PTR [rsi - 8] ; v[0](%rdx) * u[2](%rsi - 8) -> %rax:%rbx

	lea	rcx, QWORD PTR [r14 + 24] ; -un * 8 + 24 -> %rcx
	adox	r11, QWORD PTR [rdi - 8] ; low(u0*v0) + r[0](%rdi - 8) -> carry, %r11
	jrcxz	Led3 ; if (24 - 8*un == 0) <=> un == 3 goto .Led3

Ltp3: ; un >= 4, but un = bxxx11 => un = 7, 11, 15, ...
	mulx	r8, r9, QWORD PTR [rsi] ; v[0](%rdx) * u[3](%rsi) -> %r8:%r9
	adcx	r13, r10 ; high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	QWORD PTR [rdi - 8], r11 ; low(u0*v0) -> r[0](%rdi - 8)
	adox	r13, QWORD PTR [rdi] ; high(u0*v0) + low(u1*v0) + r[1](%rdi) + ovcarry -> ovcarry, %r13
	mulx	r10, r11, QWORD PTR [rsi + 8] ; v[0](%rdx) * u[4](%rsi + 8) -> %r10:%r11
	adcx	rbx, r12
	mov	QWORD PTR [rdi], r13
	adox	rbx, QWORD PTR [rdi + 8]
	mulx	r12, r13, QWORD PTR [rsi + 16]
	adcx	r9, rax
	mov	QWORD PTR [rdi + 8], rbx
	adox	r9, QWORD PTR [rdi + 16]
	mulx	rax, rbx, QWORD PTR [rsi + 24]
	adcx	r11, r8
	mov	QWORD PTR [rdi + 16], r9
	adox	r11, QWORD PTR [rdi + 24]
	lea	rsi, QWORD PTR [rsi + 32]
	lea	rdi, QWORD PTR [rdi + 32]
	lea	rcx, QWORD PTR [rcx + 32]
	jrcxz	Led3
	jmp	Ltp3

Led3:
	adcx	r13, r10
	mov	QWORD PTR [rdi - 8], r11
	adox	r13, QWORD PTR [rdi]
	adcx	rbx, r12
	mov	QWORD PTR [rdi], r13
	adox	rbx, QWORD PTR [rdi + 8]
	adcx	rax, rcx
	adox	rax, rcx
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
	jrcxz	Lmed1
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
