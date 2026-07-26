if(NOT DEFINED AITRAIN_PYTHON_TRAINERS_SOURCE
    OR NOT DEFINED AITRAIN_PYTHON_TRAINERS_DESTINATION
    OR NOT DEFINED AITRAIN_BUILD_ROOT)
    message(FATAL_ERROR "同步 Python Adapter 需要源目录和目标目录。")
endif()

file(REAL_PATH "${AITRAIN_PYTHON_TRAINERS_DESTINATION}" destination)
file(REAL_PATH "${AITRAIN_BUILD_ROOT}/bin/python_trainers" expected_destination)
if(NOT destination STREQUAL expected_destination)
    message(FATAL_ERROR "拒绝清理非固定 Python Adapter 目标目录：${destination}")
endif()

file(REMOVE_RECURSE "${destination}")
file(MAKE_DIRECTORY "${destination}")
file(COPY "${AITRAIN_PYTHON_TRAINERS_SOURCE}/"
    DESTINATION "${destination}"
    PATTERN "__pycache__" EXCLUDE
    PATTERN ".pytest_cache" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE)
