# pragma once

#include <cmath>
#include <iostream>
#include "common.h"

class vec3{
public:
    // constructor
    HD vec3() : elem{0, 0, 0} {}
    HD vec3(double _x, double _y, double _z) : elem{_x, _y, _z} {}

    HD double x() const {return elem[0];}
    HD double y() const {return elem[1];}
    HD double z() const {return elem[2];}

    // plus a vector
    HD vec3& operator+=(const vec3& v) {
        elem[0] += v.elem[0];
        elem[1] += v.elem[1];
        elem[2] += v.elem[2];
        return *this;
    }
    
    // minus a vector
    HD vec3& operator-=(const vec3& v) {
        elem[0] -= v.elem[0];
        elem[1] -= v.elem[1];
        elem[2] -= v.elem[2];
        return *this;
    }
    // mulitply by a scalar
    HD vec3& operator*=(double t) {
        elem[0] *= t;
        elem[1] *= t;
        elem[2] *= t;
        return *this;
    }

    // negative
    HD vec3 operator-() const {
        return vec3(-elem[0], -elem[1], -elem[2]);
    } 

    // length squared
    HD double length_squared() const{
        return elem[0] * elem[0] + elem[1] * elem[1] + elem[2] * elem[2];
    }

    // length
    HD double length() const {
        return std::sqrt(length_squared());
    }

private:
    double elem[3];
};

typedef vec3 point3;

// Output stream
inline std::ostream& operator<<(std::ostream& os, const vec3& v){
    return os << v.x() << " " << v.y() << " " << v.z();
}

// plus
HD inline vec3 operator+(const vec3& u, const vec3& v){
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

// minus
HD inline vec3 operator-(const vec3& u, const vec3& v){
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() -v.z());
}

// multiply by a scalar on the left
HD inline vec3 operator*(double t, const vec3& v){
    return vec3(t * v.x(), t * v.y(), t * v.z());
}

// multiply by a scalar on the right
HD inline vec3 operator*(const vec3& v, double t){
    return vec3(v.x() * t, v.y() * t, v.z() * t);
}

// devide by a scalar
HD inline vec3 operator/(const vec3& v, double t){
    return vec3(v.x() / t, v.y() / t, v.z() / t);
}

// dot product
HD inline double dot(const vec3& u, const vec3& v){
    return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

// cross product
HD inline vec3 cross(const vec3& u, const vec3& v){
    return vec3(
        u.y() * v.z() - u.z() * v.y(),
        u.z() * v.x() - u.x() * v.z(),
        u.x() * v.y() - u.y() * v.x()
    );
}

// normalize
HD inline vec3 normalize(const vec3& v){
    return v / v.length();
}
