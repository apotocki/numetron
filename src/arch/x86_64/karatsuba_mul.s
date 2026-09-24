# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2026 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

# karatsuba_mul.s -- the whole recursive Karatsuba multiplication in assembly (SysV x64; the
# Microsoft x64 version is karatsuba_mul.asm), selected by NUMETRON_KARATSUBA_IMPL_ASM, see
# umul_karatsuba_asm.hpp.
#
#   struct numetron_karatsuba_ctx { void (*mul_basecase)(rp, up, un, vp, vn); size_t threshold;
#                                   uint64_t* scratch; };
#   void numetron_karatsuba_mul(uint64_t* rp, const uint64_t* up, size_t un,
#                               const uint64_t* vp, size_t vn, const numetron_karatsuba_ctx* ctx);
#
# rp[0..un+vn) = u * v, for threshold <= vn <= un < 2*vn, threshold >= 4, rp not overlapping u, v.
# The same algorithm as umul_karatsuba_impl (split by un: s = un/2, n = un - s, t = vn - n,
# h = s + t - n; v0 = L0 + H0*B^n, vinf = Li + Hi*B^n, vm1 = |a0-a1|*|b0-b1|), recursing into
# itself while a product is Karatsuba-applicable (vn >= threshold, un < 2*vn) and calling
# ctx->mul_basecase otherwise. Differences from the C++ code:
#  - no allocator: all temporaries come from ctx->scratch, which the caller sizes (a node uses 2n
#    limbs and hands the rest to its children; see detail::karatsuba_asm_scratch);
#  - every product writes exactly un+vn limbs, so no zero padding, and no leading-zero stripping;
#  - |a0-a1| only subtracts up to the highest differing limb and zero-fills the rest;
#  - vm1 is folded into the middle-column pass as two more carry chains (one per column), so the
#    interpolation is one pass over n limbs instead of four passes.
#
# Internal routines use their own conventions (documented at each); r15 holds ctx for the whole
# recursion and is never changed below the entry point (the basecase preserves it, being a
# callee-saved register).

    .text

# ---------------------------------------------------------------------------------------------
# Entry point.
    .align  16, 0x90
    .globl  numetron_karatsuba_mul
    .hidden numetron_karatsuba_mul
    .type   numetron_karatsuba_mul, @function
numetron_karatsuba_mul:
    push    %r15
    mov     %r9, %r15                   # ctx
    mov     16(%r15), %r9               # scratch
    call    .Lmul
    pop     %r15
    ret
    .size   numetron_karatsuba_mul, .-numetron_karatsuba_mul

# ---------------------------------------------------------------------------------------------
# .Lmul: rp[0..un+vn) = u * v for un >= vn >= 1.
#   rdi rp, rsi up, rdx un, rcx vp, r8 vn, r9 scratch, r15 ctx.
# Preserves rbx, rbp, r12..r15 (the C ABI's callee-saved set); clobbers the rest.
# Must be entered by `call` (the basecase is reached by a tail jump with the caller's frame).
    .align  16, 0x90
.Lmul:
    cmp     8(%r15), %r8                # vn < threshold -> basecase
    jb      .Lmul_base
    lea     (%r8,%r8), %rax
    cmp     %rdx, %rax                  # 2*vn > un -> Karatsuba
    ja      .Lnode
.Lmul_base:
    jmp     *(%r15)

# ---------------------------------------------------------------------------------------------
# .Lnode: one Karatsuba node, same registers and contract as .Lmul, Karatsuba-applicable sizes.
# Frame: rbx rp, rbp scratch, r12 n, r13 up, r14 vp; on the stack
#   0 s, 8 t, 16 re = rp + un + vn, 24 sign of vm1, 32 carry into limb 3n, 40 scratch + 2n.
# 5 pushes + 48 bytes keep rsp 16-aligned at the calls (it is 8 mod 16 on entry).
    .align  16, 0x90
.Lnode:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    sub     $48, %rsp
    mov     %rdi, %rbx
    mov     %r9, %rbp
    mov     %rsi, %r13
    mov     %rcx, %r14
    mov     %rdx, %rax
    shr     %rax                        # s = un / 2
    mov     %rdx, %r12
    sub     %rax, %r12                  # n = un - s
    mov     %rax, 0(%rsp)
    lea     (%rdx,%r8), %r9
    lea     (%rdi,%r9,8), %r9
    mov     %r9, 16(%rsp)               # re
    lea     (%rbp,%r12,8), %r9
    lea     (%r9,%r12,8), %r9
    mov     %r9, 40(%rsp)               # scratch + 2n
    sub     %r12, %r8                   # t = vn - n
    mov     %r8, 8(%rsp)
    jz      .Lnode_t0

    # |a0 - a1| -> rp[0..n)
    mov     %rbx, %rdi
    mov     %r13, %rsi
    mov     %r12, %rdx
    lea     (%r13,%r12,8), %rcx
    mov     0(%rsp), %r8
    call    .Labsdiff
    mov     %rax, 24(%rsp)
    # |b0 - b1| -> rp[n..2n)
    lea     (%rbx,%r12,8), %rdi
    mov     %r14, %rsi
    mov     %r12, %rdx
    lea     (%r14,%r12,8), %rcx
    mov     8(%rsp), %r8
    call    .Labsdiff
    imul    24(%rsp), %rax
    mov     %rax, 24(%rsp)              # sign: +1 subtract vm1, -1 add it, 0 no vm1
    test    %rax, %rax
    jz      .Lnode_vinf

    # vm1 = |a0-a1| * |b0-b1| -> scratch[0..2n)
    mov     %rbp, %rdi
    mov     %rbx, %rsi
    mov     %r12, %rdx
    lea     (%rbx,%r12,8), %rcx
    mov     %r12, %r8
    mov     40(%rsp), %r9
    call    .Lmul

.Lnode_vinf:
    # vinf = a1 * b1 -> rp[2n..re)   (s >= t >= 1)
    lea     (%rbx,%r12,8), %rdi
    lea     (%rdi,%r12,8), %rdi
    lea     (%r13,%r12,8), %rsi
    mov     0(%rsp), %rdx
    lea     (%r14,%r12,8), %rcx
    mov     8(%rsp), %r8
    mov     40(%rsp), %r9
    call    .Lmul

    # v0 = a0 * b0 -> rp[0..2n), overwriting |a0-a1|, |b0-b1|
    mov     %rbx, %rdi
    mov     %r13, %rsi
    mov     %r12, %rdx
    mov     %r14, %rcx
    mov     %r12, %r8
    mov     40(%rsp), %r9
    call    .Lmul

    # middle columns (-/+ vm1): rax = carry into limb 2n, rdx = carry into limb 3n (signed)
    mov     0(%rsp), %rdx
    add     8(%rsp), %rdx
    sub     %r12, %rdx                  # h = s + t - n
    mov     %rbx, %rdi
    mov     %r12, %rsi
    mov     %rbp, %rcx
    mov     24(%rsp), %rax
    test    %rax, %rax
    jz      .Lnode_interp0
    js      .Lnode_interp_add
    call    .Linterp_sub
    jmp     .Lnode_carry
.Lnode_interp_add:
    call    .Linterp_add
    jmp     .Lnode_carry
.Lnode_interp0:
    call    numetron_karatsuba_interp
    mov     %rax, %rdx
    shr     $32, %rdx
    mov     %eax, %eax

.Lnode_carry:
    mov     %rdx, 32(%rsp)
    lea     (%rbx,%r12,8), %rdi
    lea     (%rdi,%r12,8), %rdi         # rp + 2n
    mov     16(%rsp), %rsi
    call    .Lprop
    mov     32(%rsp), %rax
    lea     (%rbx,%r12,8), %rdi
    lea     (%rdi,%r12,8), %rdi
    lea     (%rdi,%r12,8), %rdi         # rp + 3n
    mov     16(%rsp), %rsi
    call    .Lprop

.Lnode_ret:
    add     $48, %rsp
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret

.Lnode_t0:
    # t == 0 (vn == n): u*v = a0*v + a1*v*B^n.
    # v0 = a0 * b0 -> rp[0..2n); the scratch is still free, its children may use all of it
    mov     %rbx, %rdi
    mov     %r13, %rsi
    mov     %r12, %rdx
    mov     %r14, %rcx
    mov     %r12, %r8
    mov     %rbp, %r9
    call    .Lmul
    # b0 * a1 -> scratch[0..n+s)   (n >= s)
    mov     %rbp, %rdi
    mov     %r14, %rsi
    mov     %r12, %rdx
    lea     (%r13,%r12,8), %rcx
    mov     0(%rsp), %r8
    lea     (%rbp,%r12,8), %r9
    lea     (%r9,%r8,8), %r9
    call    .Lmul
    # rp[2n..re) = 0, then rp[n..re) += scratch[0..n+s) (no carry out: the product fits)
    lea     (%rbx,%r12,8), %rdi
    lea     (%rdi,%r12,8), %rdi
    mov     16(%rsp), %rsi
    xor     %eax, %eax
.Lnode_t0_zero:
    mov     %rax, (%rdi)
    lea     8(%rdi), %rdi
    cmp     %rsi, %rdi
    jne     .Lnode_t0_zero
    lea     (%rbx,%r12,8), %rdi
    mov     %rdi, %rsi
    mov     %rbp, %rdx
    mov     %r12, %rcx
    add     0(%rsp), %rcx
    call    numetron_add_n
    jmp     .Lnode_ret

# ---------------------------------------------------------------------------------------------
# .Labsdiff: r[0..xn) = |x - y|, rax = sign of x - y (1, -1, 0), for 1 <= yn <= xn
# (y zero-extended). rdi r, rsi x, rdx xn, rcx y, r8 yn. Clobbers the caller-saved registers.
# Only the limbs up to the highest differing one are subtracted; the ones above are zero.
    .align  16, 0x90
.Labsdiff:
    mov     %rdx, %r10
.Lad_top:                               # a non-zero x limb at or above yn: x > y
    cmp     %r8, %r10
    je      .Lad_cmp
    cmpq    $0, -8(%rsi,%r10,8)
    jne     .Lad_long
    dec     %r10
    jmp     .Lad_top
.Lad_cmp:                               # r10 = yn: find the highest differing limb
    mov     -8(%rsi,%r10,8), %rax
    cmp     -8(%rcx,%r10,8), %rax
    jne     .Lad_ne
    dec     %r10
    jnz     .Lad_cmp
    xor     %eax, %eax                  # x == y: r = 0, sign 0
.Lad_zero_all:
    mov     %rax, (%rdi)
    lea     8(%rdi), %rdi
    dec     %rdx
    jnz     .Lad_zero_all
    ret
.Lad_ne:                                # limb r10-1 is the highest differing one
    mov     $1, %r11d                   # (mov keeps the flags of the cmp)
    ja      .Lad_sub
    xchg    %rsi, %rcx
    mov     $-1, %r11
.Lad_sub:                               # r[0..r10) = rsi - rcx, r[r10..xn) = 0
    push    %r11
    push    %rdi
    push    %rdx
    push    %r10
    mov     %rcx, %rdx
    mov     %r10, %rcx
    call    numetron_sub_n
    pop     %r10
    pop     %rdx
    pop     %rdi
    pop     %rax
    xor     %ecx, %ecx
    jmp     .Lad_zc
.Lad_z:
    mov     %rcx, (%rdi,%r10,8)
    inc     %r10
.Lad_zc:
    cmp     %rdx, %r10
    jne     .Lad_z
    ret
.Lad_long:                              # r[0..yn) = x - y, r[yn..xn) = x[yn..xn) - borrow
    push    %rdi
    push    %rsi
    push    %rdx
    push    %r8
    mov     %rcx, %rdx
    mov     %r8, %rcx
    call    numetron_sub_n
    pop     %r8
    pop     %rdx
    pop     %rsi
    pop     %rdi
    mov     %rax, %rcx                  # borrow, 0 or 1
    jmp     .Lad_lc
.Lad_l:
    mov     (%rsi,%r8,8), %r9
    sub     %rcx, %r9
    setc    %cl                         # rcx stays 0 or 1
    mov     %r9, (%rdi,%r8,8)
    inc     %r8
.Lad_lc:
    cmp     %rdx, %r8
    jne     .Lad_l
    mov     $1, %eax
    ret

# ---------------------------------------------------------------------------------------------
# .Lprop: add the signed value rax at limb rdi, propagating the carry/borrow up to (not
# including) rsi; wraps silently past it. Clobbers rax, rdi.
    .align  16, 0x90
.Lprop:
    cmp     %rsi, %rdi
    je      .Lp_ret
    test    %rax, %rax
    jz      .Lp_ret
    js      .Lp_neg
    add     %rax, (%rdi)
    jnc     .Lp_ret
.Lp_inc:
    lea     8(%rdi), %rdi
    cmp     %rsi, %rdi
    je      .Lp_ret
    addq    $1, (%rdi)
    jc      .Lp_inc
.Lp_ret:
    ret
.Lp_neg:
    neg     %rax
    sub     %rax, (%rdi)
    jnc     .Lp_ret
.Lp_dec:
    lea     8(%rdi), %rdi
    cmp     %rsi, %rdi
    je      .Lp_ret
    subq    $1, (%rdi)
    jc      .Lp_dec
    ret

# ---------------------------------------------------------------------------------------------
# .Linterp_sub / .Linterp_add: the middle-column pass with vm1 folded in.
#   rdi rp, rsi n, rdx h, rcx vm1 (2n limbs). On entry rp[0..2n) = v0 = L0 + H0*B^n,
#   rp[2n..3n+h) = vinf = Li + Hi*B^n. With X = H0 + Li, for i in [0, n):
#     rp[n+i]  = X[i] + L0[i] -/+ vm1[i]
#     rp[2n+i] = X[i] + Hi[i] -/+ vm1[n+i]        (Hi[i] = 0 for i >= h)
#   Returns rax = cX + cL -/+ cV1 (carry into limb 2n), rdx = cX + cH -/+ cV2 (into limb 3n),
#   both signed. Preserves rbx, rbp, r12..r15; clobbers the rest.
# Five carry chains, each parked as 0/1 in a byte register between uses (`add $0xff, reg8` puts
# it back in CF, `setc` takes it out; for sbb chains CF is the borrow):
#   al cX, ah cL, bl cH, bh cV1, cl cV2
# Registers: rdi p = rp + n + i (H0 at (p), Li at (p,n8), Hi at (p,n8,2), L0 at (p,m)),
#   rdx q = vm1 + i (vm1 low at (q), high at (q,n8)), rsi n8, rbp m = -n8,
#   r8..r11 column-n limbs, r12..r15 column-2n limbs.
# The loop bounds are pointers in the red zone:
#   -8 end of phase-A singles, -16 end of phase A (p0 + h), -24 end of phase-B singles, -32 p0 + n.

.macro KSTEP op, hi, k
    # one limb at offset k (bytes) of the current position
    mov     \k(%rdi), %r8
    add     $0xff, %al
    adc     \k(%rdi,%rsi), %r8          # X = H0 + Li
    setc    %al
.if \hi
    mov     \k(%rdi,%rsi,2), %r12
    add     $0xff, %bl
    adc     %r8, %r12                   # X + Hi
.else
    mov     %r8, %r12
    add     $0xff, %bl
    adc     $0, %r12
.endif
    setc    %bl
    add     $0xff, %ah
    adc     \k(%rdi,%rbp), %r8          # X + L0
    setc    %ah
    add     $0xff, %bh
    \op     \k(%rdx), %r8               # -/+ vm1 low
    setc    %bh
    add     $0xff, %cl
    \op     \k(%rdx,%rsi), %r12         # -/+ vm1 high
    setc    %cl
    mov     %r8, \k(%rdi)
    mov     %r12, \k(%rdi,%rsi)
.endm

.macro KBLOCK op, hi
    mov     (%rdi), %r8
    mov     8(%rdi), %r9
    mov     16(%rdi), %r10
    mov     24(%rdi), %r11
    add     $0xff, %al
    adc     (%rdi,%rsi), %r8
    adc     8(%rdi,%rsi), %r9
    adc     16(%rdi,%rsi), %r10
    adc     24(%rdi,%rsi), %r11
    setc    %al
.if \hi
    mov     (%rdi,%rsi,2), %r12
    mov     8(%rdi,%rsi,2), %r13
    mov     16(%rdi,%rsi,2), %r14
    mov     24(%rdi,%rsi,2), %r15
    add     $0xff, %bl
    adc     %r8, %r12
    adc     %r9, %r13
    adc     %r10, %r14
    adc     %r11, %r15
.else
    mov     %r8, %r12
    mov     %r9, %r13
    mov     %r10, %r14
    mov     %r11, %r15
    add     $0xff, %bl
    adc     $0, %r12
    adc     $0, %r13
    adc     $0, %r14
    adc     $0, %r15
.endif
    setc    %bl
    add     $0xff, %ah
    adc     (%rdi,%rbp), %r8
    adc     8(%rdi,%rbp), %r9
    adc     16(%rdi,%rbp), %r10
    adc     24(%rdi,%rbp), %r11
    setc    %ah
    add     $0xff, %bh
    \op     (%rdx), %r8
    \op     8(%rdx), %r9
    \op     16(%rdx), %r10
    \op     24(%rdx), %r11
    setc    %bh
    add     $0xff, %cl
    \op     (%rdx,%rsi), %r12
    \op     8(%rdx,%rsi), %r13
    \op     16(%rdx,%rsi), %r14
    \op     24(%rdx,%rsi), %r15
    setc    %cl
    mov     %r8, (%rdi)
    mov     %r9, 8(%rdi)
    mov     %r10, 16(%rdi)
    mov     %r11, 24(%rdi)
    mov     %r12, (%rdi,%rsi)
    mov     %r13, 8(%rdi,%rsi)
    mov     %r14, 16(%rdi,%rsi)
    mov     %r15, 24(%rdi,%rsi)
.endm

.macro KINTERP name, op, fin
    .align  16, 0x90
\name:
    push    %rbx
    push    %rbp
    push    %r12
    push    %r13
    push    %r14
    push    %r15
    lea     (%rdi,%rsi,8), %rdi         # p0 = rp + n
    mov     %rdx, %rax
    and     $3, %eax
    lea     (%rdi,%rax,8), %rax
    mov     %rax, -8(%rsp)              # p0 + h % 4
    lea     (%rdi,%rdx,8), %rax
    mov     %rax, -16(%rsp)             # p0 + h
    mov     %rsi, %r8
    sub     %rdx, %r8
    and     $3, %r8d
    lea     (%rax,%r8,8), %r8
    mov     %r8, -24(%rsp)              # p0 + h + (n - h) % 4
    lea     (%rdi,%rsi,8), %rax
    mov     %rax, -32(%rsp)             # p0 + n
    shl     $3, %rsi                    # n8
    mov     %rsi, %rbp
    neg     %rbp                        # m
    mov     %rcx, %rdx                  # q
    xor     %eax, %eax
    xor     %ebx, %ebx
    xor     %ecx, %ecx

    jmp     \name\()_a1c
\name\()_a1:
    KSTEP   \op, 1, 0
    lea     8(%rdi), %rdi
    lea     8(%rdx), %rdx
\name\()_a1c:
    cmp     -8(%rsp), %rdi
    jne     \name\()_a1
    jmp     \name\()_a4c
\name\()_a4:
    KBLOCK  \op, 1
    lea     32(%rdi), %rdi
    lea     32(%rdx), %rdx
\name\()_a4c:
    cmp     -16(%rsp), %rdi
    jne     \name\()_a4

    jmp     \name\()_b1c
\name\()_b1:
    KSTEP   \op, 0, 0
    lea     8(%rdi), %rdi
    lea     8(%rdx), %rdx
\name\()_b1c:
    cmp     -24(%rsp), %rdi
    jne     \name\()_b1
    jmp     \name\()_b4c
\name\()_b4:
    KBLOCK  \op, 0
    lea     32(%rdi), %rdi
    lea     32(%rdx), %rdx
\name\()_b4c:
    cmp     -32(%rsp), %rdi
    jne     \name\()_b4

    movzbl  %ah, %edx                   # cL   (high-byte sources need non-REX destinations)
    movzbl  %bh, %esi                   # cV1
    movzbl  %al, %eax                   # cX
    movzbl  %bl, %ebx                   # cH
    movzbl  %cl, %ecx                   # cV2
    lea     (%rax,%rbx), %rdi           # cX + cH
    add     %rdx, %rax                  # cX + cL
    \fin    %rsi, %rax
    \fin    %rcx, %rdi
    mov     %rdi, %rdx
    pop     %r15
    pop     %r14
    pop     %r13
    pop     %r12
    pop     %rbp
    pop     %rbx
    ret
.endm

    KINTERP .Linterp_sub, sbb, sub
    KINTERP .Linterp_add, adc, add

    .section .note.GNU-stack,"",@progbits
