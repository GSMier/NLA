
#include <Eigen/Dense>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

template <typename Derived> std::string get_shape(const EigenBase<Derived> &x) {
  std::ostringstream oss;
  oss << "(" << x.rows() << ", " << x.cols() << ")";
  return oss.str();
}

int main() {

  int width, height, channels;
  unsigned char *image_data = stbi_load("deer.jpg", &width, &height, &channels,
                                        1); // Force load as grayscale

  if (!image_data) {
    std::cerr << "Error: Could not load image " << "deer.jpg" << std::endl;
    return 1;
  }
  /**
   * @brief 1st Task: Load an image
   *
   * @return MatrixXd
   */
  MatrixXd original(height, width);

  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int index = (i * width + j);
      original(i, j) = static_cast<int>(image_data[index]);
    }
  }

  /**
   * @brief 2nd task: Add random noise to the red channel of the image
   *
   */
  MatrixXd noisy = original;
  noisy.array() += MatrixXd::Random(height, width).array() * 50.0;

  std::vector<unsigned char> output(static_cast<size_t>(width) * height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int index = (i * width + j);
      output[index] =
          static_cast<unsigned char>(std::clamp(noisy(i, j), 0.0, 255.0));
    }
  }

  if (!stbi_write_png("deer_noisy.png", width, height, 1, output.data(),
                      height)) {
    std::cerr << "Error: Could not write deer_noisy.png" << std::endl;
    return 1;
  }

  /**
   * @brief Reshape the matrices as vector
   *
   */
  Map<VectorXd> v(original.data(), original.size());
  Map<VectorXd> w(noisy.data(), noisy.size());

  std::cout << "Shape of image_data: " << get_shape(v) << std::endl;
  std::cout << "Shape of noisy image: " << get_shape(w) << std::endl;
  stbi_image_free(image_data);

  // Euclidean norm
  double euclidean_norm = v.norm();
  std::cout << "Euclidean norm of the original image: " << euclidean_norm
            << std::endl;

  return 0;
}
