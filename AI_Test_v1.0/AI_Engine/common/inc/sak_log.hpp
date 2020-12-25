#ifndef __SAK_LOGGING_HPP__
#define __SAK_LOGGING_HPP__

#include <iostream>
#ifdef ENABLE_LOG_DEBUG
    #define LOG_DEBUG(...) std::cout<<"SAK_LOG_DEBUG: "<<__VA_ARGS__<<std::endl
#else
    #define LOG_DEBUG(...) 
#endif

#ifdef ENABLE_LOG_ERROR
    #define LOG_ERROR(...) std::cout<<"SAK_LOG_ERROR: "<<__VA_ARGS__<<std::endl
#else
    #define LOG_ERROR(...) 
#endif

#ifdef ENABLE_LOG_INFO
    #define LOG_INFO(...) std::cout<<"SAK_LOG_INFO: "<<__VA_ARGS__<<std::endl
#else
    #define LOG_INFO(...) 
#endif

#endif
