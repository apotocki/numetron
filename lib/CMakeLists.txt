set (LIBRARY_NAME numetron)
set (LIBRARY_HOME ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (NUMETRON_LIB_SOURCES
    # MASM source(s)
    ${LIBRARY_HOME}/src/arch/x86_64/alderlake/mul_basecase.s
    ${LIBRARY_HOME}/src/arch/x86_64/k8/mul_basecase.s
    ${LIBRARY_HOME}/src/arch/x86_64/core2/mul_basecase.s
    ${LIBRARY_HOME}/src/arch/x86_64/detect_platform.s
    ${LIBRARY_HOME}/src/arch/x86_64/detect_mul_basecase.s
)

add_library(${LIBRARY_NAME} STATIC ${NUMETRON_LIB_SOURCES})
