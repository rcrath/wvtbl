cmake_minimum_required(VERSION 3.16)
project(wvtbl LANGUAGES CXX)

add_executable(wvtbl wvtbl.cpp)
target_compile_features(wvtbl PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(wvtbl PRIVATE /W3 /wd4996)
else()
    target_compile_options(wvtbl PRIVATE -Wall -Wextra -O2)
endif()
