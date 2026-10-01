#include "inference/model.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    const auto model = inference::LinearModel::load("../models/travel_time_model.txt");
    inference::Features row{
        {"route_distance_km", 1.2}, {"route_complexity", 0.55},
        {"crowd_density", 0.6}, {"closure_active", 1.0},
        {"average_speed_mps", 1.1}, {"reroute_pressure", 0.5},
    };
    const double prediction = model.predict(row);
    assert(prediction > 190.0 && prediction < 220.0);
    const auto batch = model.predict_batch({row, row, row});
    assert(batch.size() == 3);
    assert(std::abs(batch[0] - batch[2]) < 1e-12);
    std::cout << "inference runtime tests passed\n";
}
