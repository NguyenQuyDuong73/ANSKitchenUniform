#pragma once
#ifndef ANSBYTETRACK_H
#define ANSBYTETRACK_H
#include "BYTETracker.h"
class  ANSByteTracker
{
private:
    ByteTrack::BYTETracker tracker;
public:
    ~ANSByteTracker();
    ANSByteTracker();
    bool UpdateParameters(int frameRate,
        int trackBuffer,
        double trackThreshold,
        double highThreshold,
        double matchThresold);
    std::vector<ByteTrack::BYTETracker::STrackPtr> Update(std::vector<ByteTrack::Object> objects);
};
#endif 
