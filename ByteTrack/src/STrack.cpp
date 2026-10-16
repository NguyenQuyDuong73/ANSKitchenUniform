#include "STrack.h"
#include <cstddef>

ByteTrack::STrack::STrack(const Rect<float>& rect, 
                         const float& score, 
                         int _class_id,
                         float _left, 
                         float _top,
                         float _right, 
                         float _bottom, 
                         std::string _object_id) :
    kalman_filter_(),
    mean_(),
    covariance_(),
    rect_(rect),
    state_(STrackState::New),
    is_activated_(false),
    score_(score),
    track_id_(0),
    frame_id_(0),
    start_frame_id_(0),
    tracklet_len_(0),
    class_id(_class_id),
    left(_left),
    right(_right),
    top(_top),
    bottom(_bottom),
    object_id(_object_id)
{
}

ByteTrack::STrack::~STrack()
{
}

const ByteTrack::Rect<float>& ByteTrack::STrack::getRect() const
{
    return rect_;
}

const ByteTrack::STrackState& ByteTrack::STrack::getSTrackState() const
{
    return state_;
}

const bool& ByteTrack::STrack::isActivated() const
{
    return is_activated_;
}
const float& ByteTrack::STrack::getScore() const
{
    return score_;
}

const size_t& ByteTrack::STrack::getTrackId() const
{
    return track_id_;
}

const size_t& ByteTrack::STrack::getFrameId() const
{
    return frame_id_;
}

const size_t& ByteTrack::STrack::getStartFrameId() const
{
    return start_frame_id_;
}

const size_t& ByteTrack::STrack::getTrackletLength() const
{
    return tracklet_len_;
}

void ByteTrack::STrack::activate(const size_t& frame_id, const size_t& track_id)
{
    kalman_filter_.initiate(mean_, covariance_, rect_.getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    if (frame_id == 1)
    {
        is_activated_ = true;
    }
    track_id_ = track_id;
    frame_id_ = frame_id;
    start_frame_id_ = frame_id;
    tracklet_len_ = 0;
}

void ByteTrack::STrack::reActivate(const STrack &new_track, const size_t &frame_id, const int &new_track_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    if (0 <= new_track_id)
    {
        track_id_ = new_track_id;
    }
    frame_id_ = frame_id;
    tracklet_len_ = 0;
    this->score_ = new_track.score_;
    this->class_id = new_track.class_id;
    this->left = new_track.left;
    this->top = new_track.top;
    this->right = new_track.right;
    this->bottom = new_track.bottom;
    this->object_id = new_track.object_id;
}

void ByteTrack::STrack::predict()
{
    if (state_ != STrackState::Tracked)
    {
        mean_[7] = 0;
    }
    kalman_filter_.predict(mean_, covariance_);
}

void ByteTrack::STrack::update(const STrack &new_track, const size_t &frame_id)
{
    kalman_filter_.update(mean_, covariance_, new_track.getRect().getXyah());

    updateRect();

    state_ = STrackState::Tracked;
    is_activated_ = true;
    score_ = new_track.getScore();
    frame_id_ = frame_id;
    tracklet_len_++;
    this->score_ = new_track.score_;
    this->class_id = new_track.class_id;
    this->left = new_track.left;
    this->top = new_track.top;
    this->right = new_track.right;
    this->bottom = new_track.bottom;
    this->object_id = new_track.object_id;
}

void ByteTrack::STrack::markAsLost()
{
    state_ = STrackState::Lost;
}

void ByteTrack::STrack::markAsRemoved()
{
    state_ = STrackState::Removed;
}

void ByteTrack::STrack::updateRect()
{
    rect_.width() = mean_[2] * mean_[3];
    rect_.height() = mean_[3];
    rect_.x() = mean_[0] - rect_.width() / 2;
    rect_.y() = mean_[1] - rect_.height() / 2;
}
