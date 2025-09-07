# Numetron — Compile-time and runtime arbitrary-precision arithmetic
# (c) 2025 Alexander Pototskiy
# Licensed under the MIT License. See LICENSE file for details.

	.text
	.align	16, 0x90
	.globl	numetron_detect_platform
	.type	numetron_detect_platform,@function

# AMD
#Family	Model   Generation	    Codename
#0F	    04	    K8      	    ClawHammer
#0F	    05	    K8      	    SledgeHammer
#0F	    07	    K8	            Newcastle
#0F	    08	    K8	            Winchester
#0F  	0B	    K8      	    Venice/San Diego
#0F	    0F	    K8      	    Toledo/Manchester
#10	    02	    K10     	    Barcelona
#10	    08	    K10     	    Shanghai
#10	    09      K10     	    Istanbul
#12  	02	    K10.5   	    Llano
#14	    02	    Bobcat	        Ontario/Zacate
#15	    01	    Bulldozer	    Zambezi
#15	    02	    Piledriver	    Vishera
#15	    13	    Steamroller	    Kaveri
#15	    30	    Excavator	    Carrizo
#16	    01	    Jaguar	        Kabini/Temash
#17	    01	    Zen	            Summit Ridge
#17	    08	    Zen+	        Pinnacle Ridge
#17	    11	    Zen 2	        Matisse
#19	    21	    Zen 3	        Vermeer
#25	    01	    Zen 4	        Raphael

# Intel
#Family	Model   Generation	    Codename
#0F	    03      Pentium 4	    Prescott
#0F	    04	    Pentium 4	    Prescott (E0)
#0F	    06	    Pentium 4	    Prescott (F6)
#6	    0F	    Core 2	        Merom/Conroe
#6	    17	    Core 2	        Penryn
#6	    1D	    Nehalem	        Bloomfield
#6	    1E	    Nehalem	        Lynnfield
#6	    1F	    Nehalem	        Clarksfield
#6	    25	    Westmere	    Gulftown
#6	    2A	    Sandy Bridge	Sandy Bridge
#6	    2D	    Sandy Bridge-E	Sandy Bridge-E
#6	    3A	    Ivy Bridge	    Ivy Bridge
#6	    3C	    Haswell	        Haswell
#6	    3F	    Haswell-E	    Haswell-E
#6	    45	    Haswell-ULT	    Haswell-ULT
#6	    46	    Haswell-GT3e	Haswell-GT3e
#6	    3D	    Broadwell	    Broadwell
#6	    47	    Broadwell	    Broadwell
#6	    4E	    Skylake	        Skylake
#6	    5E	    Skylake	        Skylake
#6	    8E	    Kaby Lake	    Kaby Lake
#6	    9E	    Coffee Lake	    Coffee Lake
#6	    A5	    Comet Lake	    Comet Lake
#6	    A6	    Ice Lake	    Ice Lake
#6	    7D	    Tiger Lake	    Tiger Lake
#6	    8C	    Rocket Lake	    Rocket Lake
#6	    97	    Alder Lake	    Alder Lake
#6	    B7	    Raptor Lake	    Raptor Lake
#6	    BD	    Meteor Lake	    Meteor Lake

# returns (0- AMD, 10000h- Intel, 20000h - Atom, 30000h- VIA, FFFF - Unknown) + family * 0x100 + model
numetron_detect_platform:
    push %rbx
    push %rcx
    push %rdx

    # Check vendor
    xor %rax, %rax
    cpuid
    
    #  ebx = vendor[0..3]
    #  edx = vendor[4..7]
    #  ecx = vendor[8..11]

    #  AMD: "AuthenticAMD"
    cmp $0x68747541, %ebx     #  "Auth"
    jne .Lcheck_intel
    cmp $0x69746E65, %edx     #  "enti"
    jne .Lcheck_intel
    cmp $0x444D4163, %ecx     #  "cAMD"
    jne .Lcheck_intel

    #  AMD detected
    xor %r9d, %r9d
    jmp .Lget_fms

.Lcheck_intel:
    #  Intel: "GenuineIntel"
    cmp $0x756E6547, %ebx     #  "uneG"
    jne .Lcheck_via
    cmp $0x49656E69, %edx     #  "Ieni"
    jne .Lcheck_via
    cmp $0x6C65746E, %ecx     #  "letn"
    jne .Lcheck_via

    # Intel detected
    mov $0x10000, %r9d
    jmp .Lget_fms

.Lcheck_via:
    #  VIA: "VIA VIA VIA"
    cmp $0x20414956, %ebx
    jne .Lunknown
    cmp $0x20414956, %edx
    jne .Lunknown
    cmp $0x20414956, %ecx
    jne .Lunknown

    #  VIA detected
    mov $0x30000, %r9d
    jmp .Lget_fms

.Lunknown:
    mov $-1, %r9d
    jmp .Ldone

.Lget_fms:
    mov $1, %eax
    cpuid
    mov %eax, %ebx        #  eax = family/model/stepping
    shr $8, %ebx
    and $0x0F, %ebx       #  keep only family
    cmp $0x0F, %ebx       #  check if family == 0x0F
    jne 1f
    mov %eax, %edx        
    shr $20, %edx
    and $0xFF, %edx       # extended family
    add %edx, %ebx        # append extended family
1:
    shl $8, %ebx
    or %ebx, %r9d         # apply family = real_family << 8

    mov %eax, %ebx
    and $0xF0, %ebx       #  keep only model
    shr $4, %ebx
    or %ebx, %r9d         #  apply model

    mov %eax, %ebx
    and $0xF0000, %ebx    #  keep only extended model
    shr $0xC, %ebx
    or %ebx, %r9d         #  apply extended model
    
    #  check if Atom
    mov %r9d, %ebx
    and $0xFFFF00, %ebx   #  select vendor and family
    cmp $0x10600, %ebx
    jne .Ldone
    mov %r9b, %al
    cmp $0x1C, %al
    je .Latom_detected
    cmp $0x26, %al
    je .Latom_detected
    cmp $0x37, %al
    je .Latom_detected
    cmp $0x4A, %al
    je .Latom_detected
    cmp $0x4C, %al
    je .Latom_detected
    cmp $0x4D, %al
    je .Latom_detected
    cmp $0x5A, %al
    je .Latom_detected
    cmp $0x5C, %al
    je .Latom_detected
    cmp $0x5D, %al
    je .Latom_detected
    cmp $0x5F, %al
    je .Latom_detected
    cmp $0x7A, %al
    je .Latom_detected
    cmp $0x86, %al
    je .Latom_detected
    cmp $0x96, %al
    je .Latom_detected

    jmp .Ldone

.Latom_detected:
    #  Atom detected
    add $0x10000, %r9d
    jmp .Ldone

.Ldone:
    mov %r9d, %eax
    pop %rdx
    pop %rcx
    pop %rbx
    ret

    .size numetron_detect_platform,.-numetron_detect_platform
	.section .note.GNU-stack,"",@progbits
