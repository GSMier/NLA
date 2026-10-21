#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <lis.h>
#include <unsupported/Eigen/SparseExtra>

typedef Eigen::SparseMatrix<double> SpMat;
using namespace Eigen;

// note that this works only for 1 channel images (grayscale)
void save_image_grayscale(const VectorXd& image, const char* filename, int width, int height)
{   
    
    std::vector<unsigned char> output(image.size());
    
    for (Eigen::Index index = 0; index < image.size(); ++index)
    {
        output[index] = static_cast<unsigned char>(std::clamp(image(index), 0.0, 255.0));
    }
    
    unsigned char* data = output.data();
    
    stbi_write_png(filename, width, height, 1, data, width);
    
}

// because it was ugly to load images too :P
// note that that the matrix is passed by reference!! (gotta go fast)
MatrixXd load_image(const char* image_name, int &width, int &height, int &channels, MatrixXd &original)
{

    unsigned char *image_data = stbi_load(image_name, &width, &height, &channels, 1); // Force load as grayscale

    if (!image_data) std::cerr << "Error: Could not load image " << image_name << "\n";

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            const int index = (i * width + j);
            original(i, j) = static_cast<int>(image_data[index]);
        }
    }

    stbi_image_free(image_data);

    return original;

}
    
int main(int argc, char* argv[])
{

    int width, height, channels;
    const char* image_name = "deer.jpg";

    MatrixXd original(height, width);
    load_image(image_name, width, height, channels, original);

    // TODO everything  

}