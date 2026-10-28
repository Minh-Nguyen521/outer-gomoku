cmake_minimum_required(VERSION 3.25)

project(outer_gomoku LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(oog_core STATIC
  src/board.cpp
  src/bitboard.cpp
  src/eval_params.cpp
  src/eval.cpp
  src/eval_v2.cpp
  src/tt.cpp
  src/engines/search.cpp
  src/engines/search_v2.cpp
  src/engines/search_v3.cpp
  src/engines/search_minibal.cpp
  src/engines/search_uct_pb.cpp
  src/engines/search_vct.cpp
  src/engines/search_nullkiller.cpp
  src/engines/search_threatcand.cpp
  src/engines/search_v4.cpp
  src/engines/search_v5.cpp
  src/engines/search_evalv2.cpp
  src/engines/search_q.cpp
  src/book.cpp
  src/piskvork.cpp
)
# `src/` holds the shared core (board, bitboard, eval, tt, book, protocol);
# `src/engines/` holds our engine variants, one pair of files each -- the same
# one-directory-per-engine separation used for third_party opponents.
target_include_directories(oog_core PUBLIC src src/engines)
target_compile_options(oog_core PRIVATE
  -Wall
  -Wextra
  -Wpedantic
  -Wconversion
  -Wshadow
)
target_compile_features(oog_core PUBLIC cxx_std_20)

# Make the curated opening book available next to the CLI binary.
configure_file(src/openings.txt ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/openings.txt COPYONLY)

option(OOG_BUILD_CLI "Build the CLI driver" ON)
option(OOG_BUILD_PROTOCOL "Build the Gomocup/Piskvork protocol binary" ON)
option(OOG_BUILD_TESTS "Build unit tests" ON)
option(OOG_BUILD_BENCH "Build the benchmark tool" ON)
option(OOG_BUILD_COMPARE "Build the Search vs SearchV2 comparison tool" ON)
option(OOG_BUILD_HEADROOM "Build the move-ordering headroom probe" ON)

if(OOG_BUILD_CLI)
  add_executable(oog_cli src/main.cpp)
  target_link_libraries(oog_cli PRIVATE oog_core)

  add_executable(oog_cli_v3 src/main_v3.cpp)
  target_link_libraries(oog_cli_v3 PRIVATE oog_core)
endif()

if(OOG_BUILD_PROTOCOL)
  add_executable(oog_piskvork tools/piskvork_main.cpp)
  target_link_libraries(oog_piskvork PRIVATE oog_core)
endif()

if(OOG_BUILD_BENCH)
  add_executable(oog_bench tools/bench.cpp)
  target_link_libraries(oog_bench PRIVATE oog_core)
endif()

if(OOG_BUILD_COMPARE)
  add_executable(oog_compare tools/compare_search.cpp)
  target_link_libraries(oog_compare PRIVATE oog_core)
endif()

if(OOG_BUILD_HEADROOM)
  add_executable(oog_headroom tools/ordering_headroom.cpp)
  target_link_libraries(oog_headroom PRIVATE oog_core)
endif()

if(OOG_BUILD_TESTS)
  enable_testing()
  add_executable(oog_tests
    tests/test_main.cpp
    tests/board_test.cpp
    tests/bitboard_test.cpp
    tests/eval_test.cpp
    tests/eval_v2_test.cpp
    tests/search_test.cpp
    tests/search_new_strategies_test.cpp
    tests/search_v4_test.cpp
    tests/search_v5_test.cpp
    tests/book_test.cpp
    tests/piskvork_test.cpp
  )
  target_include_directories(oog_tests PRIVATE third_party/doctest)
  target_link_libraries(oog_tests PRIVATE oog_core)
  add_test(NAME oog_tests COMMAND oog_tests)
endif()
