#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<std::pair<size_t, size_t>> merge_intervals(
    std::vector<std::pair<size_t, size_t>> intervals) {
  if (intervals.empty()) { return {}; }

  std::vector<std::pair<size_t, size_t>> merged;
  merged.push_back(intervals[0]);

  for (size_t i = 1; i < intervals.size(); ++i) {
    auto &lastInterval = merged.back();
    const auto &currentInterval = intervals[i];

    if (lastInterval.second == currentInterval.first) {
      // Merge the intervals
      lastInterval.second = currentInterval.second;
    } else {
      // Add the current interval as it is
      merged.push_back(currentInterval);
    }
  }

  return merged;
}

PYBIND11_MODULE(interval_op, m) {
  m.def("merge_intervals", &merge_intervals, "Merge non-overlapping intervals.");
}
