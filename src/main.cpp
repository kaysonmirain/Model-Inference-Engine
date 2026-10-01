#include "inference/model.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string model_path = "models/travel_time_model.txt";
    std::string input_path = "data/sample_features.csv";
    std::string output_path = "results/predictions.csv";
    int repeat = 10000;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (argument == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (argument == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (argument == "--repeat" && i + 1 < argc) repeat = std::atoi(argv[++i]);
    }

    try {
        const auto model = inference::LinearModel::load(model_path);
        const auto rows = inference::load_csv(input_path, model.feature_order());
        std::vector<double> predictions;
        const auto started = std::chrono::steady_clock::now();
        for (int run = 0; run < repeat; ++run) predictions = model.predict_batch(rows);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        inference::write_predictions(output_path, predictions);
        const auto total = static_cast<double>(rows.size()) * repeat;
        std::cout << "rows=" << rows.size() << " repeated=" << repeat
                  << " predictions=" << static_cast<long long>(total)
                  << " elapsed_seconds=" << elapsed
                  << " predictions_per_second=" << static_cast<long long>(total / elapsed)
                  << " output=" << output_path << '\n';
    } catch (const std::exception& error) {
        std::cerr << "model-runtime: " << error.what() << '\n';
        return 1;
    }
}
