#ifndef __POSE_ESTIMATION_ALPHAPOSE_HPP__
#define __POSE_ESTIMATION_ALPHAPOSE_HPP__

#include "two_stage_pose_estimation.hpp"
#include "sak_log.hpp"
#ifdef ENABLE_TORCH
#include "opencv2/opencv.hpp"

class EstimAlphaPose : public HPestimation
{
public:
    EstimAlphaPose();
    virtual ~EstimAlphaPose();

public:
    virtual int init(const HpesParam &param);
    virtual int predict();

protected:
    void* m_module_wrapper;
    int m_device_id;
};

#else

class EstimAlphaPose : public HPestimation
{
public:
    EstimAlphaPose()=default;
    virtual ~EstimAlphaPose()=default;

public:
    virtual int init(const HpesParam &param) { 
        LOG_ERROR("Please implement AlphaPose");
        return -1;
    }
    virtual int predict() { 
        LOG_ERROR("Please implement AlphaPose");
        return -1; 
    }
};

#endif // ENABLE_TORCH

#endif // __POSE_ESTIMATION_ALPHAPOSE_HPP__