// Physical armour-plate corners from two light bars, by line extrapolation.
//
// WHY THIS EXISTS
// ---------------
// The naive approach reads the four plate corners straight off the two bars'
// end points (minAreaRect box corners). Those four numbers are independent
// measurements, so every bit of noise on a bar's end (a partially occluded tip,
// a JPEG-blocked edge, one LED out) goes straight into the PnP input.
//
// Here the corners are DERIVED instead:
//   * each bar contributes its centre axis, extended past its own ends;
//   * its outer edge line is that axis offset by half the bar width, outward;
//   * the plate's top/bottom boundary lines are taken perpendicular to the mean
//     bar axis (a plate is a rectangle, so its top and bottom edges must be
//     perpendicular to its light bars);
//   * the four corners are the intersections of those lines.
// A corner is therefore constrained by whole lines instead of by one noisy tip.
//
// The result is still a general quadrilateral in the image (perspective is NOT
// flattened into a parallelogram: each outer edge keeps its own bar's
// direction), so PnP still sees a true perspective projection.
//
// ORDER: P1 top-left, P2 top-right, P3 bottom-right, P4 bottom-left (clockwise,
// non-self-intersecting). This must match PnpSolver's object points.
//
// INTEGRATION POINT (low intrusion)
// ---------------------------------
// In ArmorMatcher::build() (src/armor_matcher.cpp), replace:
//
//     outer_pair(left, right, &lt, &lb);
//     outer_pair(right, left, &rt, &rb);
//     armor.corners = {lt, rt, rb, lb};
//
// with:
//
//     std::vector<cv::Point2f> corners;
//     if (armorCornersFromLightBars(left.rect, right.rect, &corners)) {
//       armor.corners = {corners[0], corners[1], corners[2], corners[3]};
//     } else {                       // degenerate bars: keep the old fallback
//       outer_pair(left, right, &lt, &lb);
//       outer_pair(right, left, &rt, &rb);
//       armor.corners = {lt, rt, rb, lt};
//     }
//
// and run refineArmorCornersSubPix() on the plate corners before solvePnP().
//
// CONTRACT
// --------
// * No deep learning, no global state, no config changes.
// * The plate centre is the mean of the four corners and is used ONLY for the
//   overlay and the tracker. It is never fed to PnP - passing the centre instead
//   of the corners would silently drop the corner information and make the pose
//   unobservable.
#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace autoaim {

/// Corners of an armour plate from its two light bars.
///
/// @param left   left light bar (any RotatedRect: minAreaRect output or built
///               from a fitted axis; the long/short sides are detected
///               geometrically, so the OpenCV angle convention does not matter)
/// @param right  right light bar
/// @param corners OUT: exactly 4 points, P1 TL, P2 TR, P3 BR, P4 BL
/// @return false if the geometry degenerates (bars parallel to the boundary
///         lines, a bar with zero size, or a self-intersecting result). Callers
///         should fall back to the bar end points in that case.
/// @param edge_offset where each outer edge line sits relative to its light bar:
///                    +1 = the bar's OUTER side (plate width = outer-to-outer),
///                     0 = the bar's centre line,
///                    -1 = the bar's INNER side (plate width = the gap between
///                         the two bars). Use this when the box should hug the
///                         inside of the bars instead of the outside.
bool armorCornersFromLightBars(const cv::RotatedRect& left, const cv::RotatedRect& right,
                               std::vector<cv::Point2f>* corners,
                               float edge_offset = 1.0f);

/// Sub-pixel refinement of the four plate corners, in place.
///
/// Run this on a grayscale (or single-channel difference) image: the plate
/// corners sit on the high-contrast boundary of the light bars, which is exactly
/// what cornerSubPix wants. After refinement the reprojection error typically
/// drops, but the acceptance test (reject the frame when the error exceeds the
/// configured threshold) stays exactly as it was.
///
/// @param image        single-channel or BGR image; converted internally
/// @param corners      in/out, must hold exactly 4 points
/// @param win          half window size passed to cornerSubPix
/// @param max_shift_px reject the refinement if any corner moves further than
///                     this (it means the corner locked onto the wrong edge)
/// @return false if refinement was rejected; `corners` is then left untouched
bool refineArmorCornersSubPix(const cv::Mat& image, std::vector<cv::Point2f>* corners,
                              int win = 5, int max_iters = 30, double eps = 0.01,
                              double max_shift_px = 2.0);

/// Plate centre = mean of the four corners. Visualisation / tracking only.
void armorCenterFromCorners(const std::vector<cv::Point2f>& corners,
                            cv::Point2f* center);

}  // namespace autoaim
