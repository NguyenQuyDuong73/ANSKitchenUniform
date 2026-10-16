#pragma once
#include "Rect.h"
namespace ByteTrack
{
    struct Object
    {
        Rect<float> rect;
        int label;
        float prob;
        float left;
        float top;
        float right;
        float bottom;
        std::string object_id;

        Object(const Rect<float> &_rect,
               const int &_label,
               const float &_prob, 
               const float& _left,
               const float& _top,
               const float& _right,
               const float& _bottom,
               const std::string& _object_id);
    };
}