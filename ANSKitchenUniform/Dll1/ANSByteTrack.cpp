#include "ANSByteTrack.h"
#include <iostream>
ANSByteTracker::ANSByteTracker() {
    tracker.update_parameters(30, 30, 0.5, 0.6, 0.8);
}

ANSByteTracker::~ANSByteTracker() {

}

bool ANSByteTracker::UpdateParameters(int frameRate,
    int trackBuffer,
    double trackThreshold,
    double highThreshold,
    double matchThresold) {
	tracker.update_parameters(frameRate, trackBuffer, trackThreshold, highThreshold, matchThresold);
	return true;
}

std::vector<ByteTrack::BYTETracker::STrackPtr> ANSByteTracker::Update(std::vector<ByteTrack::Object> objects) {
    std::vector<ByteTrack::BYTETracker::STrackPtr> outputs = tracker.update(objects);

	// you can access the tracked objects here
	for (const auto& obj : outputs) {
		std::cout << "Track ID: " << obj->getTrackId() << ", Score: " << obj->getScore() << std::endl;
	}
	return outputs;

}