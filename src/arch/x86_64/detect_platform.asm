; Numetron — Compile-time and runtime arbitrary-precision arithmetic
; (c) 2025 Alexander Pototskiy
; Licensed under the MIT License. See LICENSE file for details.

; Returns platform index in RAX according to the list:
PUBLIC numetron_detect_platform

.CODE

ALIGN 16

; AMD
;Family	Model   Generation	    Codename
;0F	    04	    K8      	    ClawHammer
;0F	    05	    K8      	    SledgeHammer
;0F	    07	    K8	            Newcastle
;0F	    08	    K8	            Winchester
;0F  	0B	    K8      	    Venice/San Diego
;0F	    0F	    K8      	    Toledo/Manchester
;10	    02	    K10     	    Barcelona
;10	    08	    K10     	    Shanghai
;10	    09      K10     	    Istanbul
;12  	02	    K10.5   	    Llano
;14	    02	    Bobcat	        Ontario/Zacate
;15	    01	    Bulldozer	    Zambezi
;15	    02	    Piledriver	    Vishera
;15	    13	    Steamroller	    Kaveri
;15	    30	    Excavator	    Carrizo
;16	    01	    Jaguar	        Kabini/Temash
;17	    01	    Zen	            Summit Ridge
;17	    08	    Zen+	        Pinnacle Ridge
;17	    11	    Zen 2	        Matisse
;19	    21	    Zen 3	        Vermeer
;25	    01	    Zen 4	        Raphael

; Intel
;Family	Model   Generation	    Codename
;0F	    03      Pentium 4	    Prescott
;0F	    04	    Pentium 4	    Prescott (E0)
;0F	    06	    Pentium 4	    Prescott (F6)
;6	    0F	    Core 2	        Merom/Conroe
;6	    17	    Core 2	        Penryn
;6	    1D	    Nehalem	        Bloomfield
;6	    1E	    Nehalem	        Lynnfield
;6	    1F	    Nehalem	        Clarksfield
;6	    25	    Westmere	    Gulftown
;6	    2A	    Sandy Bridge	Sandy Bridge
;6	    2D	    Sandy Bridge-E	Sandy Bridge-E
;6	    3A	    Ivy Bridge	    Ivy Bridge
;6	    3C	    Haswell	        Haswell
;6	    3F	    Haswell-E	    Haswell-E
;6	    45	    Haswell-ULT	    Haswell-ULT
;6	    46	    Haswell-GT3e	Haswell-GT3e
;6	    3D	    Broadwell	    Broadwell
;6	    47	    Broadwell	    Broadwell
;6	    4E	    Skylake	        Skylake
;6	    5E	    Skylake	        Skylake
;6	    8E	    Kaby Lake	    Kaby Lake
;6	    9E	    Coffee Lake	    Coffee Lake         CORE2
;6	    A5	    Comet Lake	    Comet Lake
;6	    A6	    Ice Lake	    Ice Lake
;6	    7D	    Tiger Lake	    Tiger Lake
;6	    8C	    Rocket Lake	    Rocket Lake
;6	    97	    Alder Lake	    Alder Lake
;6	    B7	    Raptor Lake	    Raptor Lake
;6	    BD	    Meteor Lake	    Meteor Lake

AtomModels equ 01Ch, 026h, 037h, 04Ah, 04Ch, 04Dh, 05Ah, 05Ch, 05Dh, 05Fh, 07Ah, 086h, 096h

CheckVals MACRO label, vals:VARARG
    IRP val, <vals>
        cmp al, val
        je label
    ENDM
ENDM

; returns (0- AMD, 10000h- Intel, 20000h - Atom, 30000h- VIA, FFFF - Unknown) + family * 0x100 + model
numetron_detect_platform PROC
    push rbx
    push rcx
    push rdx

    ; Check vendor
    xor rax, rax
    cpuid
    
    ; ebx = vendor[0..3]
    ; edx = vendor[4..7]
    ; ecx = vendor[8..11]

    ; AMD: "AuthenticAMD"
    cmp ebx, 'htuA'     ; "Auth"
    jne check_intel
    cmp edx, 'itne'     ; "enti"
    jne check_intel
    cmp ecx, 'DMAc'     ; "cAMD"
    jne check_intel

    ; AMD detected
    xor r9d, r9d
    jmp get_fms

check_intel:
    ; Intel: "GenuineIntel"
    cmp ebx, 'uneG'
    jne check_via
    cmp edx, 'Ieni'
    jne check_via
    cmp ecx, 'letn'
    jne check_via

    ; Intel detected
    mov r9d, 10000h
    jmp get_fms

check_via:
    ; VIA: "VIA VIA VIA"
    cmp ebx, 'AIV '
    jne unknown
    cmp edx, 'AIV '
    jne unknown
    cmp ecx, 'AIV '
    jne unknown

    ; VIA detected
    mov r9d, 30000h
    jmp get_fms

unknown:
    mov r9d, -1
    jmp done

get_fms:
    mov eax, 1
    cpuid
    mov ebx, eax       ; eax = family/model/stepping
    and ebx, 0F00h      ; keep only family
    or  r9d, ebx        ; apply family

    mov ebx, eax
    and ebx, 0F0h       ; keep only model
    shr ebx, 4
    or  r9d, ebx        ; apply model

    mov ebx, eax
    and ebx, 0F0000h   ; keep only extended model
    shr ebx, 12
    or  r9d, ebx       ; apply extended model
    
    ; check if Atom
    mov ebx, r9d
    and ebx, 0FFFF00h    ; select vendor and family
    cmp ebx, 10600h
    jne done
    mov al, r9b
    CheckVals atom_detected, %AtomModels
    jmp done

atom_detected:
    ; Atom detected
    add r9d, 10000h
    jmp done

done:
    mov eax, r9d
    pop rdx
    pop rcx
    pop rbx
    ret

numetron_detect_platform ENDP
END
