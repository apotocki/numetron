	.text
	.align	16, 0x90
	.globl	__alderlake_mul_basecase
	.hidden	__alderlake_mul_basecase
	.type	__alderlake_mul_basecase,@function

# extern "C" void __alderlake_mul_basecase(uint64_t* rp, uint64_t* up, size_t un, uint64_t* vp, uint64_t vn);
# rp in %rdi, up in %rsi, un in %rdx, vp in %rcx, vn in %r8

__alderlake_mul_basecase:

	cmp	$2, %rdx
	ja	.Lgen
	mov	(%rcx), %rdx
	mulx (%rsi), %rax, %r9
	mov	%rax, (%rdi)
	je	.Ls2x

	mov	%r9, 8(%rdi)
	ret

.Ls2x:
	mulx 8(%rsi), %rax, %r10
	add	%r9, %rax
	adc	$0, %r10
	cmp	$2, %r8d
	je	.Ls22

.Ls21:
	mov	%rax, 8(%rdi)
	mov	%r10, 16(%rdi)
	ret

.Ls22:
	mov	8(%rcx), %rdx
	mulx (%rsi), %r8, %r9
	add	%r8, %rax
	adc	%r10, %r9
	mov	%rax, 8(%rdi)
	mulx 8(%rsi), %rax, %r10
	adc	$0, %r10
	adc	%r9, %rax
	mov	%rax, 16(%rdi)
	adc	$0, %r10
	mov	%r10, 24(%rdi)
	ret

.Lgen:	
	push	%rbx	
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	%rdx, %r14
	neg	%r14
	shl	$3, %r14
	mov	%rcx, %r15
	mov	%r8, %rbp

	test $1, %dl
	mov	(%r15), %rdx
	jz	.Lbx0

.Lbx1:
	test $16, %r14b
	jnz	.Lb01

.Lb11:
	lea	24(%r14), %rcx
	mulx (%rsi), %r11, %r10
	mulx 8(%rsi), %r13, %r12
	mulx 16(%rsi), %rbx, %rax
	lea	8(%rdi), %rdi
	lea	24(%rsi), %rsi
	jrcxz	.Lmed3

.Lmtp3:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	mulx 8(%rsi), %r11, %r10
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	mulx 16(%rsi), %r13, %r12
	adcx %rax, %r9
	mov	%rbx, 8(%rdi)
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz .Lmed3
	jmp	.Lmtp3

.Lmed3:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adcx %rcx, %rax
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret

.Lout3:
	lea	32(%rdi,%r14), %rdi
	lea	24(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	#xor	%edx, %edx
	mov	(%r15), %rdx

	mulx -24(%rsi), %r11, %r10
	mulx -16(%rsi), %r13, %r12
	mulx -8(%rsi), %rbx, %rax

	lea	24(%r14), %rcx
	adox -8(%rdi), %r11
	jrcxz	.Led3

.Ltp3:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adox (%rdi), %r13
	mulx 8(%rsi), %r11, %r10
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
	dec	%rbp
	jnz	.Lout3
	jmp	.Lret


.Lb01:
	mulx (%rsi), %rbx, %rax
	lea	8(%r14), %rcx
	mulx 8(%rsi), %r9, %r8
	mulx 16(%rsi), %r11, %r10
	lea	8(%rsi), %rsi
	lea	-8(%rdi), %rdi
	jmp	.Lml1
.Lmtp1:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	mulx 8(%rsi), %r11, %r10
	adcx %r12, %rbx
	mov	%r13, (%rdi)
.Lml1:
	mulx 16(%rsi), %r13, %r12
	adcx %rax, %r9
	mov	%rbx, 8(%rdi)
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed1
	jmp	.Lmtp1
.Lmed1:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adcx %rcx, %rax
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout1:
	lea	16(%rdi,%r14), %rdi
	lea	8(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	lea	8(%r14), %rcx
	mulx -8(%rsi), %rbx, %rax
	mulx (%rsi), %r9, %r8
	mulx 8(%rsi), %r11, %r10
	jmp	.Llo1
.Ltp1:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adox (%rdi), %r13
	mulx 8(%rsi), %r11, %r10
	adcx %r12, %rbx
	mov	%r13, (%rdi)
.Llo1:
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
	jrcxz	.Led1
	jmp	.Ltp1
.Led1:
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
	dec	%rbp
	jnz	.Lout1
	jmp	.Lret


.Lbx0:
	test $16, %r14b
	jz .Lb00

.Lb10:
	mulx (%rsi), %r13, %r12
	mulx 8(%rsi), %rbx, %rax
	lea	16(%r14), %rcx
	mulx 16(%rsi), %r9, %r8
	lea	16(%rsi), %rsi
	jmp	.Lml2
.Lmtp2:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
.Lml2:
	mulx 8(%rsi), %r11, %r10
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	mulx 16(%rsi), %r13, %r12
	adcx %rax, %r9
	mov	%rbx, 8(%rdi)
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed2
	jmp	.Lmtp2
.Lmed2:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adcx %rcx, %rax
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout2:
	lea	24(%rdi,%r14), %rdi
	lea	16(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	mulx -16(%rsi), %r13, %r12
	mulx -8(%rsi), %rbx, %rax
	lea	16(%r14), %rcx
	mulx (%rsi), %r9, %r8
	jmp	.Llo2
.Ltp2:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
.Llo2:
	adox (%rdi), %r13
	mulx 8(%rsi), %r11, %r10
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
	jrcxz	.Led2
	jmp	.Ltp2
.Led2:
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
	dec	%rbp
	jnz	.Lout2
	jmp	.Lret


.Lb00:
	lea	32(%r14), %rcx
	mulx (%rsi), %r9, %r8
	mulx 8(%rsi), %r11, %r10
	mulx 16(%rsi), %r13, %r12
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, (%rdi)
	lea	32(%rsi), %rsi
	lea	16(%rdi), %rdi
	jrcxz	.Lmed0
.Lmtp0:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	mulx 8(%rsi), %r11, %r10
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	mulx 16(%rsi), %r13, %r12
	adcx %rax, %r9
	mov	%rbx, 8(%rdi)
	mulx 24(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, 16(%rdi)
	lea	32(%rsi), %rsi
	lea	32(%rdi), %rdi
	lea	32(%rcx), %rcx
	jrcxz	.Lmed0
	jmp	.Lmtp0
.Lmed0:
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adcx %r12, %rbx
	mov	%r13, (%rdi)
	adcx %rcx, %rax
	mov	%rbx, 8(%rdi)
	mov	%rax, 16(%rdi)
	dec	%rbp
	jz	.Lret
.Lout0:
	lea	40(%rdi,%r14), %rdi
	lea	32(%rsi,%r14), %rsi
	lea	8(%r15), %r15
	xor	%edx, %edx
	mov	(%r15), %rdx
	lea	32(%r14), %rcx
	mulx -32(%rsi), %r9, %r8
	mulx -24(%rsi), %r11, %r10
	mulx -16(%rsi), %r13, %r12
	adox -16(%rdi), %r9
	mulx -8(%rsi), %rbx, %rax
	adcx %r8, %r11
	mov	%r9, -16(%rdi)
	adox -8(%rdi), %r11
	jrcxz	.Led0
.Ltp0:
	mulx (%rsi), %r9, %r8
	adcx %r10, %r13
	mov	%r11, -8(%rdi)
	adox (%rdi), %r13
	mulx 8(%rsi), %r11, %r10
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
	jrcxz	.Led0
	jmp	.Ltp0
.Led0:
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
