#ifndef __SAK_UTILS_HPP__
#define __SAK_UTILS_HPP__

#include <string>
#include <vector>

__declspec(dllexport) double time_stamp();

__declspec(dllexport) std::string time_string();

__declspec(dllexport) bool isInTimeRange(int beg_hour, int beg_min, int end_hour, int end_min);

__declspec(dllexport) std::vector<std::string> split(std::string strToSplit, char delimeter);

#endif
