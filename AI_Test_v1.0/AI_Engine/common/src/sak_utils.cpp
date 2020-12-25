#include <time.h>  
#include <sys/types.h> 
#include <time.h>  
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <windows.h>
#include "sak_utils.hpp"

static bool isTimeAfter(int target_hour, int target_min, int cur_hour, int cur_min){
    if (cur_hour>target_hour)
        return true;
    else if (cur_hour==target_hour && cur_min>=target_min)
        return true;
    return false;
}

static bool isTimeBefore(int target_hour, int target_min, int cur_hour, int cur_min){
    if (cur_hour<target_hour)
        return true;
    else if (cur_hour==target_hour && cur_min<=target_min)
        return true;
    return false;
}

double time_stamp() {
    struct timeval time;
    //gettimeofday(&time, 0);
    //return (time.tv_sec * 1000000 + time.tv_usec);
    return 0;
}

std::string time_string(){
    char buffer[80];
    time_t t = time(NULL);
    struct tm *timeinfo = localtime(&t);
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

bool isInTimeRange(int beg_hour, int beg_min, int end_hour, int end_min){
    time_t t = time(NULL);
    struct tm *tmp = localtime(&t);
    int year = tmp->tm_year+1900;
    int mon = tmp->tm_mon+1;
    int day = tmp->tm_mday;
    int hour = tmp->tm_hour;
    int min = tmp->tm_min;

    if (true == isTimeAfter(beg_hour, beg_min, hour, min) &&
        true == isTimeBefore(end_hour, end_min, hour, min))
        return true;
    
    return false;
}

std::vector<std::string> split(std::string strToSplit, char delimeter)
{
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (getline(ss, item, delimeter))
    {
        splittedStrings.push_back(item);
    }
    return splittedStrings;
}
