#ifndef NATURE_DATASET_H
#define NATURE_DATASET_H
#include <torch/torch.h>
#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

namespace dcgan{
    class NatureDataset: public torch::data::Dataset<NatureDataset>{
        std::vector<std::string> imagePaths;
    public:
        explicit NatureDataset(const fs::path& dataDir);

        // Override get()
        torch::data::Example<> get(size_t index) override;

        inline torch::optional<size_t> size() const override{ return imagePaths.size(); }
    private:
        void loadImages(const fs::path& dataDir);
    };
};

#endif // NATURE_DATASET_H