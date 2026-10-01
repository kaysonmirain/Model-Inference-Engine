#include "inference/model.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace inference {

namespace {
std::vector<std::string> split(const std::string& line, char delimiter) {
    std::vector<std::string> values;
    std::stringstream input(line);
    std::string value;
    while (std::getline(input, value, delimiter)) values.push_back(value);
    return values;
}
}  // namespace

LinearModel LinearModel::load(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open model: " + path);

    LinearModel model;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto parts = split(line, '=');
        if (parts.size() != 2) throw std::runtime_error("invalid model line: " + line);
        if (parts[0] == "bias") model.bias_ = std::stod(parts[1]);
        else if (parts[0] == "output_scale") model.output_scale_ = std::stod(parts[1]);
        else if (parts[0].starts_with("weight.")) {
            const std::string name = parts[0].substr(7);
            model.feature_order_.push_back(name);
            model.weights_[name] = std::stod(parts[1]);
        }
    }
    if (model.feature_order_.empty()) throw std::runtime_error("model has no features");
    return model;
}

double LinearModel::predict(const Features& features) const {
    double result = bias_;
    for (const auto& name : feature_order_) {
        const auto value = features.find(name);
        if (value == features.end()) throw std::runtime_error("missing feature: " + name);
        result += weights_.at(name) * value->second;
    }
    return std::max(0.0, result * output_scale_);
}

std::vector<double> LinearModel::predict_batch(const std::vector<Features>& rows) const {
    std::vector<double> output;
    output.reserve(rows.size());
    for (const auto& row : rows) output.push_back(predict(row));
    return output;
}

std::vector<Features> load_csv(const std::string& path, const std::vector<std::string>& expected_features) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open input: " + path);
    std::string line;
    if (!std::getline(input, line)) return {};
    const auto headers = split(line, ',');
    for (const auto& required : expected_features) {
        if (std::find(headers.begin(), headers.end(), required) == headers.end()) {
            throw std::runtime_error("input missing column: " + required);
        }
    }

    std::vector<Features> rows;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto values = split(line, ',');
        if (values.size() != headers.size()) throw std::runtime_error("malformed input row");
        Features row;
        for (std::size_t i = 0; i < headers.size(); ++i) row[headers[i]] = std::stod(values[i]);
        rows.push_back(std::move(row));
    }
    return rows;
}

void write_predictions(const std::string& path, const std::vector<double>& predictions) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream output(path);
    output << "row_id,predicted_trip_seconds\n";
    output << std::setprecision(17);
    for (std::size_t index = 0; index < predictions.size(); ++index) {
        output << index + 1 << ',' << predictions[index] << '\n';
    }
}

}  // namespace inference
