message(STATUS "Building D_TREE...")

file(GLOB D_TREE_SOURCES *.cpp)
list(REMOVE_ITEM D_TREE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/breakdown_dt_ours.cpp)
list(REMOVE_ITEM D_TREE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/breakdown_dt_te0.cpp)
list(REMOVE_ITEM D_TREE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/gbdt.cpp)

add_library(d_tree STATIC ${D_TREE_SOURCES})
target_include_directories(d_tree PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_link_libraries(d_tree PUBLIC yatfhe_lib)

add_executable(d_tree_ours_main breakdown_dt_ours.cpp)
target_include_directories(d_tree_ours_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_include_directories(d_tree_ours_main PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(d_tree_ours_main PRIVATE d_tree yatfhe_lib)

add_executable(d_tree_te0_main breakdown_dt_te0.cpp)
target_include_directories(d_tree_te0_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_include_directories(d_tree_te0_main PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(d_tree_te0_main PRIVATE d_tree yatfhe_lib)

add_executable(gbdt_main gbdt.cpp)
target_include_directories(gbdt_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)
target_include_directories(gbdt_main PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(gbdt_main PRIVATE d_tree yatfhe_lib)

message(STATUS "Building D_TREE...DONE!!")
