#pragma once

#include "Rect.h"
#include "KalmanFilter.h"
#include <cstddef>
namespace ByteTrack
{
    enum class STrackState {
        New = 0,
        Tracked = 1,
        Lost = 2,
        Removed = 3,
    };

    class STrack
    {
    public:
        STrack(const Rect<float>& rect, const float& score,int _class_id, float _left, float _top,
               float _right, float _bottom, std::string _object_id);
        ~STrack();

        const Rect<float>& getRect() const;
        const STrackState& getSTrackState() const;

        const bool& isActivated() const;
        const float& getScore() const;
        const size_t& getTrackId() const;
        const size_t& getFrameId() const;
        const size_t& getStartFrameId() const;
        const size_t& getTrackletLength() const;

        void activate(const size_t& frame_id, const size_t& track_id);
        void reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id = -1);

        void predict();
        void update(const STrack &new_track, const size_t &frame_id);

        void markAsLost();
        void markAsRemoved();

    private:
        KalmanFilter kalman_filter_;
        KalmanFilter::StateMean mean_;
        KalmanFilter::StateCov covariance_;
        bool is_activated_;
        float score_;

        Rect<float> rect_;
        STrackState state_;

        size_t track_id_;
        size_t frame_id_;
        size_t start_frame_id_;
        size_t tracklet_len_;

        void updateRect();
    public:
        float left; // left, top, right, bottom (original bounding of detected object) 
        float top;
        float right;
        float bottom;
        std::string object_id;
        int class_id;
    };
}