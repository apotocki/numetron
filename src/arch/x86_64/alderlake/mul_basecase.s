	.text
	.align	16, 0x90
	.globl	__alderlake_mul_basecase
	.type	__alderlake_mul_basecase,@function

# extern "C" void __alderlake_mul_basecase(uint64_t* rp, uint64_t* up, size_t un, uint64_t* vp, uint64_t vn);
# rp in %rdi, up in %rsi, un in %rdx, vp in %rcx, vn in %r8

__alderlake_mul_basecase:

	cmp	$2, %rdx
	ja	.Lgen					# if un > 2 goto .Lgen
	mov	(%rcx), %rdx			# load v[0] ->%rdx // before that %rdx could be 1 or 2, but flags are preserved and we can use them to know exactly un
	mulx (%rsi), %rax, %r9		# v[0](%rdx) * u[0](%rsi) -> %r9:%rax
	mov	%rax, (%rdi)			# %rax -> r[0](%rdi)
	je	.Ls2x					# if un == 2 goto .Ls2x

	mov	%r9, 8(%rdi)			# %r9 -> r[1](%rdi + 8)
	ret

.Ls2x:							# case un == 2
	mulx 8(%rsi), %rax, %r10	# v[0](%rdx) * u[1](%rsi + 8) -> %r10:%rax
	add	%r9, %rax				# low(v0*u1) + hight(v0*u0) -> carry, %rax
	adc	$0, %r10				# %r10 + carry -> carry, %r10
	cmp	$2, %r8d				
	je	.Ls22					# if vn == 2 goto .Ls22

.Ls21:							# case un == 2, vn == 1
	mov	%rax, 8(%rdi)			# %rax -> r[1](%rdi + 8)
	mov	%r10, 16(%rdi)			# %r10 -> r[2](%rdi + 16)
	ret

.Ls22:							# case un == 2, vn == 2
	mov	8(%rcx), %rdx			# load v[1] ->%rdx
	mulx (%rsi), %r8, %r9		# v[1](%rdx) * u[0](%rsi) -> %r9:%r8
	add	%r8, %rax				# low(v0*u1) + hight(v0*u0) + low(v1*u0) -> carry, %rax
	adc	%r10, %r9				# %r9 + carry -> carry, %r9
	mov	%rax, 8(%rdi)			# %rax -> r[1](%rdi + 8)
	mulx 8(%rsi), %rax, %r10	# v[1](%rdx) * u[1](%rsi + 8) -> %r10:%rax
	adc	$0, %r10				# %r10 + carry -> carry, %r10
	adc	%r9, %rax				# %rax + carry -> carry, %rax
	mov	%rax, 16(%rdi)			# %rax -> r[2](%rdi + 16)
	adc	$0, %r10				# %r10 + carry -> carry, %r10
	mov	%r10, 24(%rdi)			# %r10 -> r[3](%rdi + 24)
	ret

.Lgen:	
	push	%rbx	
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15
								# will be used as temporaries %rbx, %rbp, %r12, %r13, %r14, %r15
	mov	%rdx, %r14				# un -> %r14
	neg	%r14					# -un -> %r14
	shl	$3, %r14				# -un * 8 -> %r14
	mov	%rcx, %r15				# vp -> %r15
	mov	%r8, %rbp				# vn -> %rbp

	test	$1, %dl				# test if un is odd
	mov	(%r15), %rdx			# load v[0] ->%rdx
	jz	.Lbx0					# if un is even goto .Lbx0

.Lbx1:							# un is odd
	test	$16, %r14b			# test second bit of ~un
	jnz	.Lb01					# if second bit of un is 0 goto .Lb01 (un = 5, 9, 13, ...)

.Lb11:							# un = bxxx11 (u is odd, second bit of un is 1) (un = 3, 7, 11, 15, ...)
	lea	24(%r14), %rcx			# -un * 8 + 24 -> %rcx
	mulx (%rsi), %r11, %r10		# v[0](%rdx) * u[0](%rsi) -> %r10:%r11
	mulx 8(%rsi), %r13, %r12	# v[0](%rdx) * u[1](%rsi + 8) -> %r12:%r13
	mulx 16(%rsi), %rbx, %rax	# v[0](%rdx) * u[2](%rsi + 16) -> %rax:%rbx
	lea	8(%rdi), %rdi			# ++rp
	lea	24(%rsi), %rsi			# up += 3
	jrcxz	.Lmed3				# if (24 - 8*un == 0) <=> un == 3 goto .Lmed3

.Lmtp3:							# un >= 4, but un = bxxx11 => un = 7, 11, 15, ...
	mulx (%rsi), %r9, %r8		# v[0](%rdx) * u[3](%rsi) -> %r8:%r9
	adcx %r10, %r13				# high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	%r11, -8(%rdi)			# low(u0*v0) -> r[0](%rdi - 8)
	mulx 8(%rsi), %r11, %r10	# v[0](%rdx) * u[4](%rsi + 8) -> %r10:%r11
	adcx %r12, %rbx				# high(u1*v0) + low(u2*v0) + carry -> carry, %rbx
	mov	%r13, (%rdi)			# high(u0*v0) + low(u1*v0) -> r[1](%rdi)
	mulx 16(%rsi), %r13, %r12	# v[0](%rdx) * u[5](%rsi + 16) -> %r12:%r13
	adcx %rax, %r9				# high(u2*v0) + carry -> carry, %r9
	mov	%rbx, 8(%rdi)			# high(u1*v0) + low(u2*v0) + carry -> r[2](%rdi + 8)
	mulx 24(%rsi), %rbx, %rax	# v[0](%rdx) * u[6](%rsi + 24) -> %rax:%rbx
	adcx %r8, %r11				# high(u3*v0) + low(u4*v0) + carry -> carry, %r11
	mov	%r9, 16(%rdi)			# high(u2*v0) + carry -> r[3](%rdi + 16)
	lea	32(%rsi), %rsi			# up += 4
	lea	32(%rdi), %rdi			# rp += 4
	lea	32(%rcx), %rcx			# -un * 8 + 24 + 32 -> %rcx
	jrcxz	.Lmed3				# if (24 - 8*un + 32 == 0) <=> un == 7 goto .Lmed3
	jmp	.Lmtp3					# else goto .Lmtp3 // un >= 8, un = bxxx11 => un >= 11

.Lmed3:							# un = 3 || un = 7
	adcx %r10, %r13				# high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	%r11, -8(%rdi)			# low(u0*v0) -> r[0](%rdi - 8)
	adcx %r12, %rbx				# high(u1*v0) + low(u2*v0) + carry -> carry, %rbx
	mov	%r13, (%rdi)			# high(u0*v0) + low(u1*v0) -> r[1](%rdi)
	adcx %rcx, %rax				# high(u2*v0) + carry -> carry, %rax
	mov	%rbx, 8(%rdi)			# high(u1*v0) + low(u2*v0) + carry -> r[2](%rdi + 8)
	mov	%rax, 16(%rdi)			# high(u2*v0) + carry -> r[3](%rdi + 16)
	dec	%rbp					# vn--
	jz	.Lret					# if vn == 0 goto .Lret

.Lout3:							# vn > 1, un = 3 || un = 7
	lea	32(%rdi,%r14), %rdi		# rp += -un * 8 + 32
	lea	24(%rsi,%r14), %rsi		# up += -un * 8 + 24
	lea	8(%r15), %r15			# vp += 1
	#xor	%edx, %edx				# un(%rdx) = 0
	mov	(%r15), %rdx			# load v[0] ->%rdx

	mulx -24(%rsi), %r11, %r10	# v[0](%rdx) * u[0](%rsi - 24) -> %r10:%r11
	mulx -16(%rsi), %r13, %r12	# v[0](%rdx) * u[1](%rsi - 16) -> %r12:%r13
	mulx -8(%rsi), %rbx, %rax	# v[0](%rdx) * u[2](%rsi - 8) -> %rax:%rbx

	lea	24(%r14), %rcx			# -un * 8 + 24 -> %rcx
	adox -8(%rdi), %r11			# low(u0*v0) + r[0](%rdi - 8) -> carry, %r11
	jrcxz	.Led3				# if (24 - 8*un == 0) <=> un == 3 goto .Led3

.Ltp3:							# un >= 4, but un = bxxx11 => un = 7, 11, 15, ...
	mulx (%rsi), %r9, %r8		# v[0](%rdx) * u[3](%rsi) -> %r8:%r9
	adcx %r10, %r13				# high(u0*v0) + low(u1*v0) -> carry, %r13
	mov	%r11, -8(%rdi)			# low(u0*v0) -> r[0](%rdi - 8)
	adox (%rdi), %r13			# high(u0*v0) + low(u1*v0) + r[1](%rdi) + ovcarry -> ovcarry, %r13
	mulx 8(%rsi), %r11, %r10	# v[0](%rdx) * u[4](%rsi + 8) -> %r10:%r11
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adox 8(%rdi), %rbx
	mulx 16(%rsi), %r13, %r12
	adcx %rax, %r9
	mov	%rbx, 8(%rdi)
	adox 16(%rdi), %r9
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, 16(%rdi)
	adox 24(%rdi), %r11
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Led3
	jmp	.Ltp3

.Led3:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adox (%rdi), %r13
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adox 8(%rdi), %rbx
	adcx %rcx, %rax
	adox %rcx, %rax
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp					# vn--
	jnz	.Lout3
	jmp	.Lret


.Lb01:	.byte	0xc4,226,227,0xf6,6
	lea	8(%r14), %rcx
	.byte	0xc4,98,179,0xf6,70,8
	.byte	0xc4,98,163,0xf6,86,16
	lea	8(%rsi), %rsi
	lea	-8(%rdi), %rdi
	jmp	.Lml1
.Lmtp1:.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
.Lml1:	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed1
	jmp	.Lmtp1
.Lmed1:.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0x66,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout1:lea	16(%rdi,%r14), %rdi
	lea	8(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	lea	8(%r14), %rcx
	.byte	0xc4,226,227,0xf6,70,248
	.byte	0xc4,98,179,0xf6,6
	.byte	0xc4,98,163,0xf6,86,8
	jmp	.Llo1
.Ltp1:	.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,47
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
.Llo1:	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,79,16
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,95,24
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Led1
	jmp	.Ltp1
.Led1:	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,47
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0x66,72,0x0f,0x38,0xf6,193
	.byte	0xf3,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jnz	.Lout1
	jmp	.Lret


.Lbx0:	test	$16, %r14b
	jz	.Lb00

.Lb10:	.byte	0xc4,98,147,0xf6,38
	.byte	0xc4,226,227,0xf6,70,8
	lea	16(%r14), %rcx
	.byte	0xc4,98,179,0xf6,70,16
	lea	16(%rsi), %rsi
	jmp	.Lml2
.Lmtp2:.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
.Lml2:	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed2
	jmp	.Lmtp2
.Lmed2:.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0x66,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout2:lea	24(%rdi,%r14), %rdi
	lea	16(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	.byte	0xc4,98,147,0xf6,102,240
	.byte	0xc4,226,227,0xf6,70,248
	lea	16(%r14), %rcx
	.byte	0xc4,98,179,0xf6,6
	jmp	.Llo2
.Ltp2:	.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
.Llo2:	.byte	0xf3,76,0x0f,0x38,0xf6,47
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,79,16
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,95,24
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Led2
	jmp	.Ltp2
.Led2:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adox (%rdi), %r13
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0x66,72,0x0f,0x38,0xf6,193
	.byte	0xf3,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jnz	.Lout2
	jmp	.Lret


.Lb00:	lea	32(%r14), %rcx
	.byte	0xc4,98,179,0xf6,6
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, (%rdi)
	lea	32(%rsi), %rsi
	lea	16(%rdi), %rdi
	jrcxz	.Lmed0
.Lmtp0:.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed0
	jmp	.Lmtp0
.Lmed0:.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0x66,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout0:lea	40(%rdi,%r14), %rdi
	lea	32(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	lea	32(%r14), %rcx
	.byte	0xc4,98,179,0xf6,70,224
	.byte	0xc4,98,163,0xf6,86,232
	.byte	0xc4,98,147,0xf6,102,240
	.byte	0xf3,76,0x0f,0x38,0xf6,79,240
	.byte	0xc4,226,227,0xf6,70,248
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, -16(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,95,248
	jrcxz	.Led0
.Ltp0:	.byte	0xc4,98,179,0xf6,6
	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,47
	.byte	0xc4,98,163,0xf6,86,8
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0xc4,98,147,0xf6,102,16
	.byte	0x66,76,0x0f,0x38,0xf6,200
	mov	%rbx, 8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,79,16
	.byte	0xc4,226,227,0xf6,70,24
	.byte	0x66,77,0x0f,0x38,0xf6,216
	mov	%r9, 16(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,95,24
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Led0
	jmp	.Ltp0
.Led0:	.byte	0x66,77,0x0f,0x38,0xf6,234
	mov	%r11, -8(%rdi)
	.byte	0xf3,76,0x0f,0x38,0xf6,47
	.byte	0x66,73,0x0f,0x38,0xf6,220
	mov	%r13, (%rdi)
	.byte	0xf3,72,0x0f,0x38,0xf6,95,8
	.byte	0x66,72,0x0f,0x38,0xf6,193
	.byte	0xf3,72,0x0f,0x38,0xf6,193
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jnz	.Lout0

.Lret:	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
	
	.size	__alderlake_mul_basecase,.-__alderlake_mul_basecase
	.section .note.GNU-stack,"",@progbits
