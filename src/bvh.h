#pragma once

#include "aabb.h"
#include "hittable.h"

#include <algorithm>

struct bvh_node {
    aabb box;
    int left;
    int right;
    int start;
    int count;
};
