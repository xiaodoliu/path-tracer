#pragma once

#include "common.h"

struct interval{
    double min, max;
    HD interval() : min(+infinity), max(-infinity) {}
    HD interval(double _min, double _max) : min(_min), max(_max) {}
    HD bool contains(double x) const {return min <= x && x <= max;}
    HD bool surrounds(double x) const {return min < x && x < max;}
    HD bool less_than(double x) const {return max < x;}
    HD bool greater_than(double x) const {return min > x;}
    static const interval empty, universe;
};

const interval interval::empty = interval();
const interval interval::universe = interval(-infinity, +infinity);