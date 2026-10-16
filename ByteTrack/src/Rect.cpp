#include "Rect.h"
#include <algorithm>
template <typename T>
ByteTrack::Rect<T>::Rect(const T &x, const T &y, const T &width, const T &height) :
    tlwh({x, y, width, height})
{
}

template <typename T>
ByteTrack::Rect<T>::~Rect()
{
}

template <typename T>
const T& ByteTrack::Rect<T>::x() const
{
    return tlwh[0];
}

template <typename T>
const T& ByteTrack::Rect<T>::y() const
{
    return tlwh[1];
}

template <typename T>
const T& ByteTrack::Rect<T>::width() const
{
    return tlwh[2];
}

template <typename T>
const T& ByteTrack::Rect<T>::height() const
{
    return tlwh[3];
}

template <typename T>
T& ByteTrack::Rect<T>::x()
{
    return tlwh[0];
}

template <typename T>
T& ByteTrack::Rect<T>::y()
{
    return tlwh[1];
}

template <typename T>
T& ByteTrack::Rect<T>::width()
{
    return tlwh[2];
}

template <typename T>
T& ByteTrack::Rect<T>::height()
{
    return tlwh[3];
}

template <typename T>
const T& ByteTrack::Rect<T>::tl_x() const
{
    return tlwh[0];
}

template <typename T>
const T& ByteTrack::Rect<T>::tl_y() const
{
    return tlwh[1];
}

template <typename T>
T ByteTrack::Rect<T>::br_x() const
{
    return tlwh[0] + tlwh[2];
}

template <typename T>
T ByteTrack::Rect<T>::br_y() const
{
    return tlwh[1] + tlwh[3];
}

template <typename T>
ByteTrack::Tlbr<T> ByteTrack::Rect<T>::getTlbr() const
{
    return {
        tlwh[0],
        tlwh[1],
        tlwh[0] + tlwh[2],
        tlwh[1] + tlwh[3],
    };
}

template <typename T>
ByteTrack::Xyah<T> ByteTrack::Rect<T>::getXyah() const
{
    return {
        tlwh[0] + tlwh[2] / 2,
        tlwh[1] + tlwh[3] / 2,
        tlwh[2] / tlwh[3],
        tlwh[3],
    };
}

template<typename T>
float ByteTrack::Rect<T>::calcIoU(const Rect<T>& other) const
{
    const float box_area = (other.tlwh[2] + 1) * (other.tlwh[3] + 1);
    const float iw = std::min(tlwh[0] + tlwh[2], other.tlwh[0] + other.tlwh[2]) - std::max(tlwh[0], other.tlwh[0]) + 1;
    float iou = 0;
    if (iw > 0)
    {
        const float ih = std::min(tlwh[1] + tlwh[3], other.tlwh[1] + other.tlwh[3]) - std::max(tlwh[1], other.tlwh[1]) + 1;
        if (ih > 0)
        {
            const float ua = (tlwh[0] + tlwh[2] - tlwh[0] + 1) * (tlwh[1] + tlwh[3] - tlwh[1] + 1) + box_area - iw * ih;
            iou = iw * ih / ua;
        }
    }
    return iou;
}

template<typename T>
ByteTrack::Rect<T> ByteTrack::generate_rect_by_tlbr(const ByteTrack::Tlbr<T>& tlbr)
{
    return ByteTrack::Rect<T>(tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]);
}

template<typename T>
ByteTrack::Rect<T> ByteTrack::generate_rect_by_xyah(const ByteTrack::Xyah<T>& xyah)
{
    const auto width = xyah[2] * xyah[3];
    return ByteTrack::Rect<T>(xyah[0] - width / 2, xyah[1] - xyah[3] / 2, width, xyah[3]);
}

// explicit instantiation
template class ByteTrack::Rect<int>;
template class ByteTrack::Rect<float>;

template ByteTrack::Rect<int> ByteTrack::generate_rect_by_tlbr<int>(const ByteTrack::Tlbr<int>&);
template ByteTrack::Rect<float> ByteTrack::generate_rect_by_tlbr<float>(const ByteTrack::Tlbr<float>&);

template ByteTrack::Rect<int> ByteTrack::generate_rect_by_xyah<int>(const ByteTrack::Xyah<int>&);
template ByteTrack::Rect<float> ByteTrack::generate_rect_by_xyah<float>(const ByteTrack::Xyah<float>&);
