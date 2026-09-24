; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2026 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; karatsuba_mul.asm -- the whole recursive Karatsuba multiplication in assembly, Microsoft x64
; version of karatsuba_mul.s (see there for the algorithm; selected by
; NUMETRON_KARATSUBA_IMPL_ASM, see umul_karatsuba_asm.hpp).
;
;   struct numetron_karatsuba_ctx { void (*mul_basecase)(rp, up, un, vp, vn); size_t threshold;
;                                   uint64_t* scratch; };
;   void numetron_karatsuba_mul(uint64_t* rp, const uint64_t* up, size_t un,
;                               const uint64_t* vp, size_t vn, const numetron_karatsuba_ctx* ctx);
;
; Microsoft x64 calling convention at the entry point: rcx = rp, rdx = up, r8 = un, r9 = vp,
; [rsp+40] = vn, [rsp+48] = ctx. The private routines keep the register conventions of the GAS
; version (rdi rp, rsi up, rdx un, rcx vp, r8 vn, r9 scratch, r15 ctx), so the bodies match it
; line for line. What differs:
;  - calls out (mul_basecase, numetron_sub_n / numetron_add_n, numetron_karatsuba_interp) follow the
;    Microsoft convention: arguments in rcx, rdx, r8, r9, the fifth on the stack, 32 bytes of home
;    space. Every frame that calls Lmul keeps a 48-byte outgoing area at its bottom (home space
;    + the fifth argument), so Lmul can store vn there and tail-jump into mul_basecase;
;  - there is no red zone: the interpolation keeps its loop bounds in its own frame;
;  - rdi and rsi are non-volatile here, so the entry point saves them;
;  - every routine that moves rsp has unwind info (PROC FRAME).

OPTION CASEMAP:NONE
OPTION PROC:PRIVATE

PUBLIC numetron_karatsuba_mul

EXTERN numetron_sub_n:PROC
EXTERN numetron_add_n:PROC
EXTERN numetron_karatsuba_interp:PROC

; Lnode frame (above the 48-byte outgoing area).
KN_S    EQU 48      ; s
KN_T    EQU 56      ; t
KN_RE   EQU 64      ; re = rp + un + vn
KN_SGN  EQU 72      ; sign of vm1
KN_CY3  EQU 80      ; carry into limb 3n
KN_WS2  EQU 88      ; scratch + 2n
KN_SIZE EQU 96

.code

; ---------------------------------------------------------------------------------------------
; Entry point.
ALIGN 16
numetron_karatsuba_mul PROC FRAME
    push    rbx
    .pushreg rbx
    push    rbp
    .pushreg rbp
    push    rdi
    .pushreg rdi
    push    rsi
    .pushreg rsi
    push    r12
    .pushreg r12
    push    r13
    .pushreg r13
    push    r14
    .pushreg r14
    push    r15
    .pushreg r15
    sub     rsp, 56                     ; outgoing area (48) + alignment
    .allocstack 56
    .endprolog

    mov     rdi, rcx                    ; rp
    mov     rsi, rdx                    ; up
    mov     rdx, r8                     ; un
    mov     rcx, r9                     ; vp
    mov     r8, QWORD PTR [rsp + 160]   ; vn  (56 + 8 pushes + return address + 32 home)
    mov     r15, QWORD PTR [rsp + 168]  ; ctx
    mov     r9, QWORD PTR [r15 + 16]    ; scratch
    call    Lmul

    add     rsp, 56
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rsi
    pop     rdi
    pop     rbp
    pop     rbx
    ret
numetron_karatsuba_mul ENDP

; ---------------------------------------------------------------------------------------------
; Lmul: rp[0..un+vn) = u * v for un >= vn >= 1.
;   rdi rp, rsi up, rdx un, rcx vp, r8 vn, r9 scratch, r15 ctx.
; Preserves rbx, rbp, r12..r15; clobbers the rest (rdi and rsi included). Must be called from a
; frame with a 48-byte outgoing area: the basecase is reached by a tail jump that passes vn in
; that frame's fifth-argument slot.
ALIGN 16
Lmul PROC
    cmp     r8, QWORD PTR [r15 + 8]     ; vn < threshold -> basecase
    jb      mul_base
    lea     rax, [r8 + r8]
    cmp     rax, rdx                    ; 2*vn > un -> Karatsuba
    ja      Lnode
mul_base:
    mov     QWORD PTR [rsp + 40], r8    ; vn: the caller's fifth-argument slot
    mov     rax, rcx
    mov     rcx, rdi                    ; rp
    mov     r8, rdx                     ; un
    mov     rdx, rsi                    ; up
    mov     r9, rax                     ; vp
    jmp     QWORD PTR [r15]
Lmul ENDP

; ---------------------------------------------------------------------------------------------
; Lnode: one Karatsuba node, same registers and contract as Lmul, Karatsuba-applicable sizes.
; rbx rp, rbp scratch, r12 n, r13 up, r14 vp; the rest in the frame (KN_*).
; 5 pushes + 96 bytes keep rsp 16-aligned at the calls.
ALIGN 16
Lnode PROC FRAME
    push    rbx
    .pushreg rbx
    push    rbp
    .pushreg rbp
    push    r12
    .pushreg r12
    push    r13
    .pushreg r13
    push    r14
    .pushreg r14
    sub     rsp, KN_SIZE
    .allocstack KN_SIZE
    .endprolog

    mov     rbx, rdi
    mov     rbp, r9
    mov     r13, rsi
    mov     r14, rcx
    mov     rax, rdx
    shr     rax, 1                      ; s = un / 2
    mov     r12, rdx
    sub     r12, rax                    ; n = un - s
    mov     QWORD PTR [rsp + KN_S], rax
    lea     r9, [rdx + r8]
    lea     r9, [rdi + r9*8]
    mov     QWORD PTR [rsp + KN_RE], r9
    lea     r9, [rbp + r12*8]
    lea     r9, [r9 + r12*8]
    mov     QWORD PTR [rsp + KN_WS2], r9
    sub     r8, r12                     ; t = vn - n
    mov     QWORD PTR [rsp + KN_T], r8
    jz      node_t0

    ; |a0 - a1| -> rp[0..n)
    mov     rdi, rbx
    mov     rsi, r13
    mov     rdx, r12
    lea     rcx, [r13 + r12*8]
    mov     r8, QWORD PTR [rsp + KN_S]
    call    Labsdiff
    mov     QWORD PTR [rsp + KN_SGN], rax
    ; |b0 - b1| -> rp[n..2n)
    lea     rdi, [rbx + r12*8]
    mov     rsi, r14
    mov     rdx, r12
    lea     rcx, [r14 + r12*8]
    mov     r8, QWORD PTR [rsp + KN_T]
    call    Labsdiff
    imul    rax, QWORD PTR [rsp + KN_SGN]
    mov     QWORD PTR [rsp + KN_SGN], rax ; +1 subtract vm1, -1 add it, 0 no vm1
    test    rax, rax
    jz      node_vinf

    ; vm1 = |a0-a1| * |b0-b1| -> scratch[0..2n)
    mov     rdi, rbp
    mov     rsi, rbx
    mov     rdx, r12
    lea     rcx, [rbx + r12*8]
    mov     r8, r12
    mov     r9, QWORD PTR [rsp + KN_WS2]
    call    Lmul

node_vinf:
    ; vinf = a1 * b1 -> rp[2n..re)   (s >= t >= 1)
    lea     rdi, [rbx + r12*8]
    lea     rdi, [rdi + r12*8]
    lea     rsi, [r13 + r12*8]
    mov     rdx, QWORD PTR [rsp + KN_S]
    lea     rcx, [r14 + r12*8]
    mov     r8, QWORD PTR [rsp + KN_T]
    mov     r9, QWORD PTR [rsp + KN_WS2]
    call    Lmul

    ; v0 = a0 * b0 -> rp[0..2n), overwriting |a0-a1|, |b0-b1|
    mov     rdi, rbx
    mov     rsi, r13
    mov     rdx, r12
    mov     rcx, r14
    mov     r8, r12
    mov     r9, QWORD PTR [rsp + KN_WS2]
    call    Lmul

    ; middle columns (-/+ vm1): rax = carry into limb 2n, rdx = carry into limb 3n (signed)
    mov     rdx, QWORD PTR [rsp + KN_S]
    add     rdx, QWORD PTR [rsp + KN_T]
    sub     rdx, r12                    ; h = s + t - n
    mov     rax, QWORD PTR [rsp + KN_SGN]
    test    rax, rax
    jz      node_interp0
    mov     rdi, rbx                    ; (mov keeps the flags of the test)
    mov     rsi, r12
    mov     rcx, rbp
    js      node_interp_add
    call    Linterp_sub
    jmp     node_carry
node_interp_add:
    call    Linterp_add
    jmp     node_carry
node_interp0:
    mov     rcx, rbx                    ; numetron_karatsuba_interp(rp, n, h), Microsoft x64
    mov     r8, rdx
    mov     rdx, r12
    call    numetron_karatsuba_interp
    mov     rdx, rax
    shr     rdx, 32
    mov     eax, eax

node_carry:
    mov     QWORD PTR [rsp + KN_CY3], rdx
    lea     rdi, [rbx + r12*8]
    lea     rdi, [rdi + r12*8]          ; rp + 2n
    mov     rsi, QWORD PTR [rsp + KN_RE]
    call    Lprop
    mov     rax, QWORD PTR [rsp + KN_CY3]
    lea     rdi, [rbx + r12*8]
    lea     rdi, [rdi + r12*8]
    lea     rdi, [rdi + r12*8]          ; rp + 3n
    mov     rsi, QWORD PTR [rsp + KN_RE]
    call    Lprop

node_ret:
    add     rsp, KN_SIZE
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

node_t0:
    ; t == 0 (vn == n): u*v = a0*v + a1*v*B^n.
    ; v0 = a0 * b0 -> rp[0..2n); the scratch is still free, its children may use all of it
    mov     rdi, rbx
    mov     rsi, r13
    mov     rdx, r12
    mov     rcx, r14
    mov     r8, r12
    mov     r9, rbp
    call    Lmul
    ; b0 * a1 -> scratch[0..n+s)   (n >= s)
    mov     rdi, rbp
    mov     rsi, r14
    mov     rdx, r12
    lea     rcx, [r13 + r12*8]
    mov     r8, QWORD PTR [rsp + KN_S]
    lea     r9, [rbp + r12*8]
    lea     r9, [r9 + r8*8]
    call    Lmul
    ; rp[2n..re) = 0, then rp[n..re) += scratch[0..n+s) (no carry out: the product fits)
    lea     rdi, [rbx + r12*8]
    lea     rdi, [rdi + r12*8]
    mov     rsi, QWORD PTR [rsp + KN_RE]
    xor     eax, eax
node_t0_zero:
    mov     QWORD PTR [rdi], rax
    lea     rdi, [rdi + 8]
    cmp     rdi, rsi
    jne     node_t0_zero
    lea     rcx, [rbx + r12*8]          ; numetron_add_n(rp + n, rp + n, scratch, n + s)
    mov     rdx, rcx
    mov     r8, rbp
    mov     r9, r12
    add     r9, QWORD PTR [rsp + KN_S]
    call    numetron_add_n
    jmp     node_ret
Lnode ENDP

; ---------------------------------------------------------------------------------------------
; Labsdiff: r[0..xn) = |x - y|, rax = sign of x - y (1, -1, 0), for 1 <= yn <= xn
; (y zero-extended). rdi r, rsi x, rdx xn, rcx y, r8 yn. Clobbers rax, rcx, rdx, rsi, rdi,
; r8..r11. Only the limbs up to the highest differing one are subtracted; the ones above are zero.
; Frame: home space for numetron_sub_n, then four save slots.
ALIGN 16
Labsdiff PROC FRAME
    sub     rsp, 72
    .allocstack 72
    .endprolog

    mov     r10, rdx
ad_top:                                 ; a non-zero x limb at or above yn: x > y
    cmp     r10, r8
    je      ad_cmp
    cmp     QWORD PTR [rsi + r10*8 - 8], 0
    jne     ad_long
    dec     r10
    jmp     ad_top
ad_cmp:                                 ; r10 = yn: find the highest differing limb
    mov     rax, QWORD PTR [rsi + r10*8 - 8]
    cmp     rax, QWORD PTR [rcx + r10*8 - 8]
    jne     ad_ne
    dec     r10
    jnz     ad_cmp
    xor     eax, eax                    ; x == y: r = 0, sign 0
ad_zero_all:
    mov     QWORD PTR [rdi], rax
    lea     rdi, [rdi + 8]
    dec     rdx
    jnz     ad_zero_all
    add     rsp, 72
    ret
ad_ne:                                  ; limb r10-1 is the highest differing one
    mov     r11d, 1                     ; (mov keeps the flags of the cmp)
    ja      ad_sub
    xchg    rsi, rcx
    mov     r11, -1
ad_sub:                                 ; r[0..r10) = rsi - rcx, r[r10..xn) = 0
    mov     QWORD PTR [rsp + 32], r11
    mov     QWORD PTR [rsp + 40], rdi
    mov     QWORD PTR [rsp + 48], rdx
    mov     QWORD PTR [rsp + 56], r10
    mov     r9, r10                     ; numetron_sub_n(r, rsi, rcx, r10), Microsoft x64
    mov     r8, rcx
    mov     rdx, rsi
    mov     rcx, rdi
    call    numetron_sub_n
    mov     r10, QWORD PTR [rsp + 56]
    mov     rdx, QWORD PTR [rsp + 48]
    mov     rdi, QWORD PTR [rsp + 40]
    mov     rax, QWORD PTR [rsp + 32]
    xor     ecx, ecx
    jmp     ad_zc
ad_z:
    mov     QWORD PTR [rdi + r10*8], rcx
    inc     r10
ad_zc:
    cmp     r10, rdx
    jne     ad_z
    add     rsp, 72
    ret
ad_long:                                ; r[0..yn) = x - y, r[yn..xn) = x[yn..xn) - borrow
    mov     QWORD PTR [rsp + 32], rdi
    mov     QWORD PTR [rsp + 40], rsi
    mov     QWORD PTR [rsp + 48], rdx
    mov     QWORD PTR [rsp + 56], r8
    mov     r9, r8                      ; numetron_sub_n(r, x, y, yn), Microsoft x64
    mov     r8, rcx
    mov     rdx, rsi
    mov     rcx, rdi
    call    numetron_sub_n
    mov     r8, QWORD PTR [rsp + 56]
    mov     rdx, QWORD PTR [rsp + 48]
    mov     rsi, QWORD PTR [rsp + 40]
    mov     rdi, QWORD PTR [rsp + 32]
    mov     rcx, rax                    ; borrow, 0 or 1
    jmp     ad_lc
ad_l:
    mov     r9, QWORD PTR [rsi + r8*8]
    sub     r9, rcx
    setc    cl                          ; rcx stays 0 or 1
    mov     QWORD PTR [rdi + r8*8], r9
    inc     r8
ad_lc:
    cmp     r8, rdx
    jne     ad_l
    mov     eax, 1
    add     rsp, 72
    ret
Labsdiff ENDP

; ---------------------------------------------------------------------------------------------
; Lprop: add the signed value rax at limb rdi, propagating the carry/borrow up to (not
; including) rsi; wraps silently past it. Clobbers rax, rdi. Leaf, no stack use.
ALIGN 16
Lprop PROC
    cmp     rdi, rsi
    je      p_ret
    test    rax, rax
    jz      p_ret
    js      p_neg
    add     QWORD PTR [rdi], rax
    jnc     p_ret
p_inc:
    lea     rdi, [rdi + 8]
    cmp     rdi, rsi
    je      p_ret
    add     QWORD PTR [rdi], 1
    jc      p_inc
p_ret:
    ret
p_neg:
    neg     rax
    sub     QWORD PTR [rdi], rax
    jnc     p_ret
p_dec:
    lea     rdi, [rdi + 8]
    cmp     rdi, rsi
    je      p_ret
    sub     QWORD PTR [rdi], 1
    jc      p_dec
    ret
Lprop ENDP

; ---------------------------------------------------------------------------------------------
; Linterp_sub / Linterp_add: the middle-column pass with vm1 folded in; same contract and
; registers as .Linterp_sub / .Linterp_add in karatsuba_mul.s:
;   rdi rp, rsi n, rdx h, rcx vm1 (2n limbs) -> rax carry into limb 2n, rdx into limb 3n (signed).
;   Preserves rbx, rbp, r12..r15; clobbers the rest.
; Five carry chains in byte registers: al cX, ah cL, bl cH, bh cV1, cl cV2.
; rdi p = rp + n + i, rdx q = vm1 + i, rsi n8, rbp m = -n8, r8..r11 column n, r12..r15 column 2n.
; Loop bounds in the frame: [rsp] end of phase-A singles, [rsp+8] end of phase A (p0 + h),
; [rsp+16] end of phase-B singles, [rsp+24] p0 + n.

KSTEP MACRO aop, withhi
    mov     r8, QWORD PTR [rdi]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]   ; X = H0 + Li
    setc    al
IF withhi
    mov     r12, QWORD PTR [rdi + rsi*2]
    add     bl, 0FFh
    adc     r12, r8                     ; X + Hi
ELSE
    mov     r12, r8
    add     bl, 0FFh
    adc     r12, 0
ENDIF
    setc    bl
    add     ah, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]   ; X + L0
    setc    ah
    add     bh, 0FFh
    aop     r8, QWORD PTR [rdx]         ; -/+ vm1 low
    setc    bh
    add     cl, 0FFh
    aop     r12, QWORD PTR [rdx + rsi]  ; -/+ vm1 high
    setc    cl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + rsi], r12
ENDM

KBLOCK MACRO aop, withhi
    mov     r8, QWORD PTR [rdi]
    mov     r9, QWORD PTR [rdi + 8]
    mov     r10, QWORD PTR [rdi + 16]
    mov     r11, QWORD PTR [rdi + 24]
    add     al, 0FFh
    adc     r8, QWORD PTR [rdi + rsi]
    adc     r9, QWORD PTR [rdi + rsi + 8]
    adc     r10, QWORD PTR [rdi + rsi + 16]
    adc     r11, QWORD PTR [rdi + rsi + 24]
    setc    al
IF withhi
    mov     r12, QWORD PTR [rdi + rsi*2]
    mov     r13, QWORD PTR [rdi + rsi*2 + 8]
    mov     r14, QWORD PTR [rdi + rsi*2 + 16]
    mov     r15, QWORD PTR [rdi + rsi*2 + 24]
    add     bl, 0FFh
    adc     r12, r8
    adc     r13, r9
    adc     r14, r10
    adc     r15, r11
ELSE
    mov     r12, r8
    mov     r13, r9
    mov     r14, r10
    mov     r15, r11
    add     bl, 0FFh
    adc     r12, 0
    adc     r13, 0
    adc     r14, 0
    adc     r15, 0
ENDIF
    setc    bl
    add     ah, 0FFh
    adc     r8, QWORD PTR [rdi + rbp]
    adc     r9, QWORD PTR [rdi + rbp + 8]
    adc     r10, QWORD PTR [rdi + rbp + 16]
    adc     r11, QWORD PTR [rdi + rbp + 24]
    setc    ah
    add     bh, 0FFh
    aop     r8, QWORD PTR [rdx]
    aop     r9, QWORD PTR [rdx + 8]
    aop     r10, QWORD PTR [rdx + 16]
    aop     r11, QWORD PTR [rdx + 24]
    setc    bh
    add     cl, 0FFh
    aop     r12, QWORD PTR [rdx + rsi]
    aop     r13, QWORD PTR [rdx + rsi + 8]
    aop     r14, QWORD PTR [rdx + rsi + 16]
    aop     r15, QWORD PTR [rdx + rsi + 24]
    setc    cl
    mov     QWORD PTR [rdi], r8
    mov     QWORD PTR [rdi + 8], r9
    mov     QWORD PTR [rdi + 16], r10
    mov     QWORD PTR [rdi + 24], r11
    mov     QWORD PTR [rdi + rsi], r12
    mov     QWORD PTR [rdi + rsi + 8], r13
    mov     QWORD PTR [rdi + rsi + 16], r14
    mov     QWORD PTR [rdi + rsi + 24], r15
ENDM

KINTERP MACRO pname, aop, cop
    LOCAL a1, a1c, a4, a4c, b1, b1c, b4, b4c
ALIGN 16
pname PROC FRAME
    push    rbx
    .pushreg rbx
    push    rbp
    .pushreg rbp
    push    r12
    .pushreg r12
    push    r13
    .pushreg r13
    push    r14
    .pushreg r14
    push    r15
    .pushreg r15
    sub     rsp, 40
    .allocstack 40
    .endprolog

    lea     rdi, [rdi + rsi*8]          ; p0 = rp + n
    mov     rax, rdx
    and     eax, 3
    lea     rax, [rdi + rax*8]
    mov     QWORD PTR [rsp], rax        ; p0 + h % 4
    lea     rax, [rdi + rdx*8]
    mov     QWORD PTR [rsp + 8], rax    ; p0 + h
    mov     r8, rsi
    sub     r8, rdx
    and     r8d, 3
    lea     r8, [rax + r8*8]
    mov     QWORD PTR [rsp + 16], r8    ; p0 + h + (n - h) % 4
    lea     rax, [rdi + rsi*8]
    mov     QWORD PTR [rsp + 24], rax   ; p0 + n
    shl     rsi, 3                      ; n8
    mov     rbp, rsi
    neg     rbp                         ; m
    mov     rdx, rcx                    ; q
    xor     eax, eax
    xor     ebx, ebx
    xor     ecx, ecx

    jmp     a1c
a1:
    KSTEP aop, 1
    lea     rdi, [rdi + 8]
    lea     rdx, [rdx + 8]
a1c:
    cmp     rdi, QWORD PTR [rsp]
    jne     a1
    jmp     a4c
a4:
    KBLOCK aop, 1
    lea     rdi, [rdi + 32]
    lea     rdx, [rdx + 32]
a4c:
    cmp     rdi, QWORD PTR [rsp + 8]
    jne     a4

    jmp     b1c
b1:
    KSTEP aop, 0
    lea     rdi, [rdi + 8]
    lea     rdx, [rdx + 8]
b1c:
    cmp     rdi, QWORD PTR [rsp + 16]
    jne     b1
    jmp     b4c
b4:
    KBLOCK aop, 0
    lea     rdi, [rdi + 32]
    lea     rdx, [rdx + 32]
b4c:
    cmp     rdi, QWORD PTR [rsp + 24]
    jne     b4

    movzx   edx, ah                     ; cL   (high-byte sources need non-REX destinations)
    movzx   esi, bh                     ; cV1
    movzx   eax, al                     ; cX
    movzx   ebx, bl                     ; cH
    movzx   ecx, cl                     ; cV2
    lea     rdi, [rax + rbx]            ; cX + cH
    add     rax, rdx                    ; cX + cL
    cop     rax, rsi
    cop     rdi, rcx
    mov     rdx, rdi

    add     rsp, 40
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret
pname ENDP
ENDM

KINTERP Linterp_sub, sbb, sub
KINTERP Linterp_add, adc, add

END
