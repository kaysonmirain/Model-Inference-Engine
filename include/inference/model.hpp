#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

using Features = std::unordered_map<std::string, double>;

class LinearModel {
public:
    static LinearModel load(const std::string& path);
    [[nodiscard]] double predict(const Features& features) const;
    [[nodiscard]] std::vector<double> predict_batch(const std::vector<Features>& rows) const;
    [[nodiscard]] const std::vector<std::string>& feature_order() const { return feature_order_; }

private:
    double bias_{};
    double output_scale_{1.0};
    std::vector<std::string> feature_order_;
    std::unordered_map<std::string, double> weights_;
};

std::vector<Features> load_csv(const std::string& path, const std::vector<std::string>& expected_features);
void write_predictions(const std::string& path, const std::vector<double>& predictions);

}  // namespace inference

