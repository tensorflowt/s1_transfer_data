
include_directories(./inc)
aux_source_directory(./src src_cpp)

set(target sak_common)
add_library(${target} SHARED ${src_cpp})

## Install the target
INSTALL(TARGETS ${target}  DESTINATION lib64)
