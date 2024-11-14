CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(ResNetCPP)

set(CMAKE_CXX_STANDARD 17)

add_library(ResNetCPP SHARED 
  src/resnet_export.h
  src/resnet.cc
  src/resnet.h
  )

find_package(OpenCV  REQUIRED)
find_package(onnxruntime  REQUIRED)

message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "onnxruntime_FOUND: ${onnxruntime_FOUND}")

if (NOT ONNX_RUNTIME_LIB)
  message(STATUS "ONNX_RUNTIME_LIB is not set, trying to find it")
  find_library(ONNX_RUNTIME_LIB onnxruntime ${onnxruntime_DIR}/lib)
else ()
  message(STATUS "ONNX_RUNTIME_LIB is set to ${ONNX_RUNTIME_LIB}")
endif ()

target_include_directories(ResNetCPP PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories(ResNetCPP PRIVATE ${onnxruntime_INCLUDE_DIRS})


target_link_libraries(ResNetCPP ${OpenCV_LIBS} ${ONNX_RUNTIME_LIB})

set_target_properties(ResNetCPP PROPERTIES DEFINE_SYMBOL "ResNetCPP_EXPORTS")

add_executable(ResNetTest src/test.cc)

target_link_libraries(ResNetTest ResNetCPP)
