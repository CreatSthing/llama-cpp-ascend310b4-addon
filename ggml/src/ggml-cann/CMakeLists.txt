if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

# Auto-detech Soc type and Soc version, if detect failed, will abort build
set(SOC_VERSION "")
function(detect_ascend_soc_type SOC_VERSION)
    execute_process(
        COMMAND bash -c "npu-smi info|awk -F' ' 'NF > 0 && NR==7 {print $3}'"
        OUTPUT_VARIABLE npu_info
        RESULT_VARIABLE npu_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if("${npu_info}" STREQUAL "" OR ${npu_result})
        message(FATAL_ERROR "Auto-detech ascend soc type failed, please specify manually or check ascend device working normally.")
    endif()
    set(${SOC_VERSION} "Ascend${npu_info}" PARENT_SCOPE)
endfunction()

if(NOT SOC_TYPE)
    detect_ascend_soc_type(SOC_VERSION)
    set(SOC_TYPE "${SOC_VERSION}")
    message(STATUS "CANN: SOC_VERSION auto-detected is:${SOC_VERSION}")
endif()

string(TOLOWER ${SOC_TYPE} SOC_VERSION) # SOC_VERSION need lower

# Construct Soc specify compile option: ASCEND_#Soc_Major_SN. Such as ASCEND_910B, ASCEND_310P.
string(REGEX MATCH "[0-9]+[a-zA-Z]" SOC_TYPE_MAJOR_SN "${SOC_VERSION}")
set(SOC_TYPE_COMPILE_OPTION "ASCEND_${SOC_TYPE_MAJOR_SN}")
string(TOUPPER ${SOC_TYPE_COMPILE_OPTION} SOC_TYPE_COMPILE_OPTION)
message(STATUS "CANN: SOC_VERSION =  ${SOC_VERSION}")
option(USE_ACL_GRAPH "Enable CANN graph execution (ACL graph mode)" OFF)

if(USE_ACL_GRAPH AND (SOC_TYPE_MAJOR_SN STREQUAL "310P" OR SOC_TYPE_COMPILE_OPTION STREQUAL "ASCEND_310P"))
    message(FATAL_ERROR
        "CANN Graph (ACL graph mode) is not supported on 310P devices. "
        "Please build with -DUSE_ACL_GRAPH=OFF or use a supported SOC.")
endif()

if (CANN_INSTALL_DIR)
    # Only Support Linux.
    if (NOT UNIX)
        message(FATAL_ERROR "CANN: CANN toolkit supports unix but not ${CMAKE_SYSTEM_NAME}")
    endif()

    # Supported platforms: x86-64, arm64
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    else()
        message(FATAL_ERROR "CANN: CANN toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    # Set header and libs
    set(CANN_INCLUDE_DIRS
        ${CANN_INSTALL_DIR}/include
        ${CANN_INSTALL_DIR}/include/aclnn
        ${CANN_INSTALL_DIR}/acllib/include
    )

    list(APPEND CANN_LIBRARIES
        ascendcl
        nnopbase
        opapi
        acl_op_compiler
    )

    file(GLOB GGML_SOURCES_CANN "*.cpp")

    ggml_add_backend_library(ggml-cann ${GGML_SOURCES_CANN})
    target_link_libraries(ggml-cann PRIVATE ${CANN_LIBRARIES})
    target_include_directories(ggml-cann PRIVATE ${CANN_INCLUDE_DIRS})
    target_link_directories(ggml-cann PRIVATE ${CANN_INSTALL_DIR}/lib64)

    set(CANN_CUSTOM_OPS_DIR "$ENV{GGML_CANN_CUSTOM_OPS_DIR}" CACHE PATH "Optional CANN custom operator vendor directory")
    if (EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/include/aclnn_attention_step_custom.h" AND
        EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/lib/libcust_opapi.so")
        target_compile_definitions(ggml-cann PRIVATE GGML_CANN_HAS_ATTENTION_STEP_CUSTOM)
        target_include_directories(ggml-cann PRIVATE ${CANN_CUSTOM_OPS_DIR}/op_api/include)
        target_link_directories(ggml-cann PRIVATE ${CANN_CUSTOM_OPS_DIR}/op_api/lib)
        target_link_libraries(ggml-cann PRIVATE cust_opapi)
        message(STATUS "CANN: AttentionStepCustom is available.")
    endif()

    if (EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/include/aclnn_w8a8_quantize_custom.h" AND
        EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/include/aclnn_matmul_w8a8_i32_custom.h" AND
        EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/include/aclnn_w8a8_dequant_custom.h" AND
        EXISTS "${CANN_CUSTOM_OPS_DIR}/op_api/lib/libcust_opapi.so")
        target_compile_definitions(ggml-cann PRIVATE GGML_CANN_HAS_A8W8_CUSTOM)
        target_include_directories(ggml-cann PRIVATE ${CANN_CUSTOM_OPS_DIR}/op_api/include)
        target_link_directories(ggml-cann PRIVATE ${CANN_CUSTOM_OPS_DIR}/op_api/lib)
        target_link_libraries(ggml-cann PRIVATE cust_opapi)
        message(STATUS "CANN: custom A8W8 operators are available.")
    endif()

    if (EXISTS "${CANN_INSTALL_DIR}/include/aclnnop/aclnn_fused_infer_attention_score_v2.h")
        target_compile_definitions(ggml-cann PRIVATE GGML_CANN_HAS_FUSED_INFER_ATTENTION_SCORE_V2)
        message(STATUS "CANN: FusedInferAttentionScoreV2 is available.")
    else()
        message(STATUS "CANN: FusedInferAttentionScoreV2 is unavailable; flash attention will use the generic graph.")
    endif()

    target_compile_definitions(ggml-cann PRIVATE "-D${SOC_TYPE_COMPILE_OPTION}")

    if (USE_ACL_GRAPH)
        target_compile_definitions(ggml-cann PRIVATE USE_ACL_GRAPH)
        message(STATUS "CANN: USE_ACL_GRAPH is enabled.")
    else()
        message(STATUS "CANN: USE_ACL_GRAPH is disabled.")
    endif()

    message(STATUS "CANN: CANN_INCLUDE_DIRS =  ${CANN_INCLUDE_DIRS}")
    message(STATUS "CANN: CANN_LIBRARIES =  ${CANN_LIBRARIES}")
else()
    message(FATAL_ERROR "CANN: Can't find CANN_INSTALL_DIR, did you forget to source set_var.sh?")
endif()
