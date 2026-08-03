function(forgeir_enable_strict_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX /permissive- /EHsc)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
        )
    else()
        message(WARNING
            "ForgeIR has no strict warning profile for ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()
