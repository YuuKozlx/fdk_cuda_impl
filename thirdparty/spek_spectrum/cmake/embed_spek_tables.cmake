if(NOT DEFINED INPUT_BIN OR NOT DEFINED OUTPUT_CPP)
  message(FATAL_ERROR "embed_spek_tables.cmake requires INPUT_BIN and OUTPUT_CPP")
endif()

if(EXISTS "${OUTPUT_CPP}" AND "${OUTPUT_CPP}" IS_NEWER_THAN "${INPUT_BIN}")
  return()
endif()

file(READ "${INPUT_BIN}" hex_data HEX)
string(LENGTH "${hex_data}" hex_length)
get_filename_component(output_directory "${OUTPUT_CPP}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
set(byte_lines "")
set(includes "")
set(part_index 0)
set(part_lines 0)
set(position 0)

# Convert bounded chunks so CMake does not create one enormous regex result.
while(position LESS hex_length)
  math(EXPR remaining "${hex_length} - ${position}")
  set(chunk_length 4096)
  if(remaining LESS chunk_length)
    set(chunk_length ${remaining})
  endif()
  string(SUBSTRING "${hex_data}" ${position} ${chunk_length} chunk)
  string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1," bytes "${chunk}")
  string(APPEND byte_lines "\n    ${bytes}")
  math(EXPR position "${position} + ${chunk_length}")
  math(EXPR part_lines "${part_lines} + 1")
  if(part_lines EQUAL 150 OR position EQUAL hex_length)
    if(part_index LESS 10)
      set(part_name "spek_tables_part_0${part_index}.inc")
    else()
      set(part_name "spek_tables_part_${part_index}.inc")
    endif()
    file(WRITE "${output_directory}/${part_name}" "${byte_lines}\n")
    string(APPEND includes "#include \"${part_name}\"\n")
    set(byte_lines "")
    set(part_lines 0)
    math(EXPR part_index "${part_index} + 1")
  endif()
endwhile()

set(generated [=[
// Generated from spek_tables.bin. Do not edit.
#include <cstddef>
#include <cstdint>

namespace spek_spectrum::embedded {

alignas(16) extern const std::uint8_t kData[] = {]=])
string(APPEND generated "\n${includes}};\n")
string(APPEND generated [=[
extern const std::size_t kDataSize = sizeof(kData);

}  // namespace spek_spectrum::embedded
]=])

get_filename_component(output_directory "${OUTPUT_CPP}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT_CPP}" "${generated}")
