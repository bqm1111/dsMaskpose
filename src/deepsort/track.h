#ifndef TRACK_H
#define TRACK_H

#include <future>

#include "dataType.h"
#include "utility.h"

#include "kalmanfilter.h"

class Track
{
    /*"""
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """*/
    enum TrackState {Tentative = 1, Confirmed, Deleted};

public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const FEATURE& feature, 
          const std::string& det_class = "", const cv::Scalar& col = cv::Scalar());
    void predit(std::shared_ptr<KalmanFilter> &kf);
    void update(std::shared_ptr<KalmanFilter> & kf, const DETECTION_ROW &detection);
    void set_dt(std::shared_ptr<KalmanFilter> &kf, const float& dt);
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();
    DETECTBOX to_tlwh();
    DETECTBOX to_xyah();
    int time_since_update;
    int track_id;
    std::string detection_class;
    cv::Scalar color;
    FEATURESS features;
    FEATURE last_feature;
    KAL_MEAN mean;
    KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // TRACK_H
