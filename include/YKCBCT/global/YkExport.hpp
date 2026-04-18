#pragma once
#ifdef _WIN32
#  ifdef YKCBCT_EXPORTS
#    define YK_API __declspec(dllexport)
#  else
#    define YK_API __declspec(dllimport)
#  endif
#else
#  define YK_API __attribute__((visibility("default")))
#endif