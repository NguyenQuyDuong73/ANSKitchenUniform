#include "Object.h"
ByteTrack::Object::Object(const ByteTrack::Rect<float> &_rect,
                          const int &_label,
                          const float &_prob,
						  const float& _left,
						  const float& _top,
						  const float& _right,
						  const float& _bottom,
	                      const std::string& _object_id) : 
						  rect(_rect), 
						  label(_label), 
						  prob(_prob),
						  left(_left),
						  top(_top),
						  right(_right),
						  bottom(_bottom),
						  object_id(_object_id)
{
}