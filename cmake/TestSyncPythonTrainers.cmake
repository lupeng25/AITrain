if(NOT DEFINED AITRAIN_PYTHON_TRAINERS_SOURCE
    OR NOT DEFINED AITRAIN_PYTHON_TRAINERS_DESTINATION
    OR NOT DEFINED AITRAIN_BUILD_ROOT
    OR NOT DEFINED AITRAIN_SYNC_SCRIPT)
    message(FATAL_ERROR "Python trainer 同步测试缺少必要参数。")
endif()

file(MAKE_DIRECTORY "${AITRAIN_PYTHON_TRAINERS_DESTINATION}")
set(stale_file "${AITRAIN_PYTHON_TRAINERS_DESTINATION}/stale-sentinel.txt")
file(WRITE "${stale_file}" "stale")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
        "-DAITRAIN_PYTHON_TRAINERS_SOURCE=${AITRAIN_PYTHON_TRAINERS_SOURCE}"
        "-DAITRAIN_PYTHON_TRAINERS_DESTINATION=${AITRAIN_PYTHON_TRAINERS_DESTINATION}"
        "-DAITRAIN_BUILD_ROOT=${AITRAIN_BUILD_ROOT}"
        -P "${AITRAIN_SYNC_SCRIPT}"
    RESULT_VARIABLE sync_result
)
if(NOT sync_result EQUAL 0)
    message(FATAL_ERROR "Python trainer 同步脚本失败：${sync_result}")
endif()
if(EXISTS "${stale_file}")
    message(FATAL_ERROR "Python trainer 同步未删除 stale sentinel。")
endif()
if(NOT EXISTS "${AITRAIN_PYTHON_TRAINERS_DESTINATION}/adapter_sdk.py")
    message(FATAL_ERROR "Python trainer 同步缺少 adapter_sdk.py。")
endif()
file(GLOB_RECURSE cache_entries
    "${AITRAIN_PYTHON_TRAINERS_DESTINATION}/__pycache__"
    "${AITRAIN_PYTHON_TRAINERS_DESTINATION}/*.pyc"
    "${AITRAIN_PYTHON_TRAINERS_DESTINATION}/*.pyo")
if(cache_entries)
    message(FATAL_ERROR "Python trainer 同步包含缓存文件：${cache_entries}")
endif()
