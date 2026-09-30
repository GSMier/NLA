#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef Eigen::SparseMatrix<double> SpMat;

using namespace Eigen;

template <typename Derived> std::string get_shape(const EigenBase<Derived> &x) {
  std::ostringstream oss;
  oss << "(" << x.rows() << ", " << x.cols() << ")";
  return oss.str();
}

int main() {

  MatrixXd Hav1(3, 3);
  Hav1 << 1, 1, 1, 1, 4, 1, 1, 1, 1;
  Hav1 *= 1. / 12;
  // Flip kernel matrix Hav1 for convolution

  MatrixXd Hav2(5, 5);
  Hav2 << 0, 1, 2, 1, 0, 1, 4, 8, 4, 1, 2, 8, 16, 8, 2, 1, 4, 8, 4, 1, 0, 1, 2,
      1, 0;
  Hav2 *= 1. / 80;

  MatrixXd Hsh1(3, 3);
  Hsh1 << 0, -3, 0, -1, 9, -3, 0, -1, 0;

  MatrixXd Hed1(3, 3);
  Hed1 << 0, -1, 0, -1, 4, -1, 0, -1, 0;

  MatrixXd Hed2(3, 3);
  Hed2 << -1, 0, 1, -2, 0, 2, -1, 0, 1;

  //  Load the image as an Eigen matrix with size m × n. Each entry in the
  //  matrix corresponds
  // to a pixel on the screen and takes a value somewhere between 0 (black) and
  // 255 (white). Report the size of the matrix.

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

  // • Introduce a noise signal into the loaded image by adding random
  // ﬂuctuations of color ranging between [−50, 50] to each pixel. Export the
  // resulting image in .png and upload it.

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

  // • Reshape the original and noisy images as vectors v and w, respectively.
  // Verify that each vector has m n components. Report here the Euclidean norm
  // of v.

  /**
   * @brief Reshape the matrices as vector
   *
   */
  VectorXd v = original.reshaped<RowMajor>(1, original.size()).transpose();

  VectorXd w = noisy.reshaped<RowMajor>(1, noisy.size()).transpose();

  std::cout << "Shape of image_data: " << get_shape(v) << std::endl;
  std::cout << "Shape of noisy image: " << get_shape(w) << std::endl;
  stbi_image_free(image_data);

  // Euclidean norm
  double euclidean_norm = v.norm();
  std::cout << "Euclidean norm of the original image: " << euclidean_norm
            << std::endl;

  // • Write the convolution operation corresponding to the smoothing kernel
  // Hav1 as a matrix vector multiplication between a matrix A1 having size mn ×
  // mn and the image vector. Report the number of non-zero entries in A1.
  const int rows = noisy.rows();
  const int cols = noisy.cols();
  const int mn = rows * cols;
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<size_t>(mn) * Hav1.size());

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int row = i * cols + j;
      for (int r = 0; r < Hav1.rows(); ++r) {
        for (int c = 0; c < Hav1.cols(); ++c) {
          const int src_i = i + r - Hav1.rows() / 2;
          const int src_j = j + c - Hav1.cols() / 2;
          if (src_i >= 0 && src_i < rows && src_j >= 0 && src_j < cols) {
            const int col = src_i * cols + src_j;
            triplets.emplace_back(row, col, Hav1(r, c));
          }
        }
      }
    }
  }

  Eigen::SparseMatrix<double> A1(mn, mn);
  A1.setFromTriplets(triplets.begin(), triplets.end());
  std::cout << "Number of non-zero entries in A1: " << A1.nonZeros()
            << std::endl;

  // • Apply the previous smoothing filter to the noisy image by performing the
  // matrix vector multiplication A1w. Export and upload the resulting image.

  VectorXd filtered_vec = A1 * w;
  MatrixXd filtered_image(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      filtered_image(i, j) = filtered_vec(i * cols + j);
    }
  }

  std::vector<unsigned char> filtered_output(static_cast<size_t>(cols) * rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const int index = i * cols + j;
      filtered_output[index] = static_cast<unsigned char>(
          std::clamp(filtered_image(i, j), 0.0, 255.0));
    }
  }

  if (!stbi_write_png("deer_A1_filtered.png", cols, rows, 1,
                      filtered_output.data(), cols)) {
    std::cerr << "Error: Could not write deer_A1_filtered.png" << std::endl;
    return 1;
  }

  std::cout << "Saved filtered image to deer_A1_filtered.png" << std::endl;

  // • Write the convolution operation corresponding to the sharpening kernel
  // Hsh1 as a matrix vector multiplication by a matrix A2 having size mn × mn.
  // Report the number of non-zero entries in A2. Is A2 symmetric?

  // • Apply the previous sharpening ﬁlter to the original image by performing
  // the matrix vector multiplication A2v. Export and upload the resulting
  // image.

  // • Export the Eigen matrix A2 and vector w in the .mtx format. Using a
  // suitable iterative solver and preconditioning technique available in the
  // LIS library compute the approximate solution to the linear system A2 x = w
  // prescribing a tolerance of 10−12. Report here the iteration count and the
  // ﬁnal residual.

  // • Convert the previous approximate solution vector x into a .png image.
  // Upload the resulting ﬁle here.

  // • Write the convolution operation corresponding to the detection kernel
  // Hed2 as a matrix vector multiplication by a matrix A3 having size mn × mn.
  // Is m3 symmetric?

  // • Apply the previous edge detection ﬁlter to the original image by
  // performing the matrix vector multiplication A3v. Export and upload the
  // resulting image.

  // • Using a suitable iterative solver available in the Eigen library compute
  // the approximate so- lution of the linear system (4I +A3)y = w, where I
  // denotes the identity matrix, prescribing a tolerance of 10−10. Report here
  // the iteration count and the ﬁnal residual.

  // • Import the previous approximate solution vector y in Eigen and convert it
  // into a .png image and upload it.

  return 0;
}
