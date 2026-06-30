#pragma once

#include "common.h"

struct interval {
    double min, max;
    HD constexpr interval() : min(+infinity), max(-infinity) {}
    HD constexpr interval(double _min, double _max) : min(_min), max(_max) {}
    HD interval(const interval& a, const interval& b) {
        min = std::fmin(a.min, b.min);
        max = std::fmax(a.max, b.max);
    }
    HD bool contains(double x) const { return min <= x && x <= max; }
    HD bool surrounds(double x) const { return min < x && x < max; }
    HD bool less_than(double x) const { return max < x; }
    HD bool greater_than(double x) const { return min > x; }
    HD double clamp(double x) const { return std::fmax(min, std::fmin(x, max)); }
    HD interval expand(double delta) const {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }
    HD double size() const { return max - min; }
    HD static constexpr interval empty() { return interval(); }
    HD static constexpr interval universe() { return interval(-infinity, +infinity); }
};
