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

// this because it was really ugly to save the image everytime
void save_image(const VectorXd& image, const char* filename,
                int width, int height)
{
  
  std::vector<unsigned char> output(image.size());

  for (Eigen::Index index = 0; index < image.size(); ++index)
  {
    output[index] = static_cast<unsigned char>(std::clamp(image(index), 0.0, 255.0));
  }

  unsigned char* data = output.data();

  stbi_write_png(filename, width, height, 1, data, width);

}

SpMat generate_conv_matrix(const MatrixXd& conv_mat, int width, int height) {

  // get the matrix relative range (so that it works with all types of odd matrices)
  const int half_mat_rows = conv_mat.rows()/2;
  const int half_mat_cols = conv_mat.cols()/2;

  // vector where the sparse matrix will be stored
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(width) * height * conv_mat.size());

  // we pass on all pixels...
  for (int row = 0; row < height; ++row)
  {
    for (int col = 0; col < width; ++col) 
    {

      // current pixel
      const int index = width * row + col;

      // ...all the value of the conv matrix
      for (int current_mat_row = -half_mat_rows;
           current_mat_row <= half_mat_rows;
           ++current_mat_row)
      {
        for (int current_mat_col = -half_mat_cols;
             current_mat_col <= half_mat_cols;
             ++current_mat_col)
          {
           
            // check if the pixel is in a valid position (inside of the image)
            const int mat_pixel_row = row + current_mat_row;
            const int mat_pixel_col = col + current_mat_col;
            
            // ...if not we continue
            if (mat_pixel_row < 0 || mat_pixel_row >= height ||
              mat_pixel_col < 0 || mat_pixel_col >= width)
              {
                continue;
              }
            
            // get the index of the pixel that the conv_mat is referring to
            const int current_mat_index = width * (mat_pixel_row) + (mat_pixel_col);
            // and its value
            const double current_mat_value = conv_mat(half_mat_rows + current_mat_row,
                                                      half_mat_cols + current_mat_col);

            // optimization - if it's zero we can avoid adding it 
            if (current_mat_value == 0.0) { continue; }

            // we add the values to the triplets array
            triplets.emplace_back(index, current_mat_index, current_mat_value);

          }
      }
    }
  }

  // we create the sparse matrix where each row contains
  // the weights to calculate each pixel
  SpMat matrix(width * height, width * height);
  matrix.setFromTriplets(triplets.begin(), triplets.end());

  return matrix;

}

bool is_symmetric(const SpMat &matrix)
{
  return matrix.isApprox(matrix.transpose());
}

std::string get_shape(int rows, int cols)
{
  return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

int main(int argc, char* argv[]) {

  // NICELY FORMATTED PLEASE DON'T CHANGE!!!
  // IT TOOK ME AN HOUR T.T
  MatrixXd Hav1(3, 3);
  Hav1 << 1, 1, 1,
          1, 4, 1,
          1, 1, 1;
  Hav1 *= 1. / 12;

  MatrixXd Hav2(5, 5);
  Hav2 << 0, 1,  2, 1, 0,
          1, 4,  8, 4, 1,
          2, 8, 16, 8, 2,
          1, 4,  8, 4, 1,
          0, 1,  2, 1, 0;
  Hav2 *= 1. / 80;

  MatrixXd Hsh1(3, 3);
  Hsh1 <<  0, -3,  0,
          -1,  9, -3,
           0, -1,  0;

  MatrixXd Hed1(3, 3);
  Hed1 <<  0, -1,  0,
          -1,  4, -1,
           0, -1,  0;

  MatrixXd Hed2(3, 3);
  Hed2 << -1, 0, 1,
          -2, 0, 2,
          -1, 0, 1;

  const char* image_name = "deer.jpg";

  //  Load the image as an Eigen matrix with size m × n. Each entry in the
  //  matrix corresponds
  // to a pixel on the screen and takes a value somewhere between 0 (black) and
  // 255 (white). Report the size of the matrix.

  int width, height, channels;
  unsigned char *image_data = stbi_load(image_name, &width, &height, &channels, 1); // Force load as grayscale

  if (!image_data) {
    std::cerr << "Error: Could not load image " << image_name << "\n";
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
  stbi_image_free(image_data);

  // • Introduce a noise signal into the loaded image by adding random
  // ﬂuctuations of color ranging between [−50, 50] to each pixel. Export the
  // resulting image in .png and upload it.

  /**
   * @brief 2nd task: Add random noise to the gray channel of the image
   *
   */
  MatrixXd noisy = original;
  noisy.array() += MatrixXd::Random(height, width).array() * 50.0;
  // the image export is in the next step, since my save method only works with 
  // vectors and in the next step we convert from matrix to vector :)
  
  // • Reshape the original and noisy images as vectors v and w, respectively.
  // Verify that each vector has m n components. Report here the Euclidean norm
  // of v.
  
  /**
  * @brief Reshape the matrices as vector
  *
  */
  VectorXd v = original.reshaped<RowMajor>(1, original.size()).transpose();
  VectorXd w = noisy.reshaped<RowMajor>(1, noisy.size()).transpose();
  
  save_image(w, "noisy_image.png", width, height);
  
  std::cout << "m: " << width << " n: " << height << "\n";
  std::cout << "m x n: " << width * height << "\n";
  std::cout << "Shape of image_data: " << get_shape(v.rows(), v.cols()) << "\n";
  std::cout << "Shape of noisy image: " << get_shape(w.rows(), w.cols()) << "\n";

  std::cout << "Correct vector sizes? " << (v.size() == width * height && w.size() == width * height) << "\n";
  
  // Euclidean norm
  double euclidean_norm = v.norm();
  std::cout << "Euclidean norm of the original image: " << euclidean_norm << "\n";

  // • Write the convolution operation corresponding to the smoothing kernel
  // Hav1 as a matrix vector multiplication between a matrix A1 having size mn ×
  // mn and the image vector. Report the number of non-zero entries in A1.

  // Generate the sparse matrix for the smoothing kernel Hav1.
  SpMat A1 = generate_conv_matrix(Hav1, width, height);

  // Report its dimensions and number of non-zero entries.
  std::cout << "Shape of A1: " << get_shape(A1.rows(), A1.cols()) << "\n";
  std::cout << "Number of non-zero entries in A1: " << A1.nonZeros() << "\n";

  // // • Apply the previous smoothing filter to the noisy image by performing the
  // // matrix vector multiplication A1w. Export and upload the resulting image.

  VectorXd filtered_vec = A1 * w;
  save_image(filtered_vec, "deer_A1_filtered.png", width, height);  

  // • Write the convolution operation corresponding to the sharpening kernel
  // Hsh1 as a matrix vector multiplication by a matrix A2 having size mn × mn.
  // Report the number of non-zero entries in A2. Is A2 symmetric?

  SpMat A2 = generate_conv_matrix(Hsh1, width, height);
  std::cout << "Non-zero entries in A2: " << A2.nonZeros() << "\n";
  std::cout << "Is A2 symmetric? " << is_symmetric(A2) << "\n";

  // • Apply the previous sharpening ﬁlter to the original image by performing
  // the matrix vector multiplication A2v. Export and upload the resulting
  // image.

  VectorXd sharpened_image = A2 * v;
  save_image(sharpened_image, "deer_sharpened.png", width, height);

  // • Export the Eigen matrix A2 and vector w in the .mtx format. Using a
  // suitable iterative solver and preconditioning technique available in the
  // LIS library compute the approximate solution to the linear system A2 x = w
  // prescribing a tolerance of 10^−12. Report here the iteration count and the
  // ﬁnal residual.

  Eigen::saveMarket(A2, "A2.mtx");

  FILE* out = fopen("w.mtx","w");
  fprintf(out, "%%%%MatrixMarket vector coordinate real general\n");
  fprintf(out, "%d\n", static_cast<int>(w.size()));
  for (int i = 0; i < w.size(); i++) {
    fprintf(out,"%d %.16e\n", i + 1 , w(i));
  }
  fclose(out);

  // LIS
  lis_initialize(&argc, &argv);

  // declarations
  LIS_MATRIX A2_lis;
  LIS_VECTOR w_lis;
  LIS_VECTOR x_lis;
  LIS_SOLVER solver;

  // actually create and fill empty objects
  lis_matrix_create(LIS_COMM_WORLD, &A2_lis);
  lis_input_matrix(A2_lis, "A2.mtx");
  lis_vector_create(LIS_COMM_WORLD, &w_lis);
  lis_input_vector(w_lis, "w.mtx");

  lis_vector_duplicate(A2_lis, &x_lis); // x same size of matrix A2_lis

  // solver
  lis_solver_create(&solver);
  lis_solver_set_option("-i bicgstab -p iluc", solver); // bicgstab with ilu as precondtioner
  lis_solver_set_option("-tol 1.0e-12", solver); // tol given
  lis_solve(A2_lis, w_lis, x_lis, solver); 

  // read #s to report and cout them
  LIS_INT iterations;
  LIS_REAL residual;
  lis_solver_get_iter(solver, &iterations);
  lis_solver_get_residualnorm(solver, &residual);
  std::cout << "LIS iterations: " << iterations << "\n";
  std::cout << "LIS final relative residual: " << residual << "\n";

  // copy solution into Eigen, needed for next task
  VectorXd x(width * height);
  lis_vector_get_values(x_lis, 0, width * height, x.data());

  // free space and close LIS!
  lis_solver_destroy(solver);
  lis_vector_destroy(x_lis);
  lis_vector_destroy(w_lis);
  lis_matrix_destroy(A2_lis);
  lis_finalize();

  // • Convert the previous approximate solution vector x into a .png image.
  // Upload the resulting ﬁle here.

  save_image(x, "deer_x.png", width, height);

  // • Write the convolution operation corresponding to the detection kernel
  // Hed2 as a matrix vector multiplication by a matrix A3 having size mn × mn.
  // Is m3 symmetric?

  SpMat A3 = generate_conv_matrix(Hed2, width, height);
  std::cout << "Non-zero entries in A3: " << A3.nonZeros() << "\n";
  std::cout << "Is A3 symmetric? " << is_symmetric(A3) << "\n";

  // • Apply the previous edge detection ﬁlter to the original image by
  // performing the matrix vector multiplication A3v. Export and upload the
  // resulting image.

  VectorXd edge_image = A3 * v;
  save_image(edge_image, "deer_edges.png", width, height);

  // • Using a suitable iterative solver available in the Eigen library compute
  // the approximate solution of the linear system (4I + A3)y = w, where I
  // denotes the identity matrix, prescribing a tolerance of 10^−10. Report here
  // the iteration count and the ﬁnal residual.

  // S = 4I + A3 is not simmetric, only the diagonal changes. Use again BiCGSTAB
  // build S and sparse I
  SpMat I(width * height, width * height);
  I.setIdentity();
  SpMat S = 4.0 * I + A3;

  Eigen::BiCGSTAB<SpMat> bicgstab;
  bicgstab.setTolerance(1.0e-10);
  bicgstab.compute(S);
  VectorXd y = bicgstab.solve(w);
  std::cout << "Eigen iterations: " << bicgstab.iterations() << "\n";
  std::cout << "Eigen final relative residual: " << bicgstab.error() << "\n";

  // • Import the previous approximate solution vector y in Eigen and convert it
  // into a .png image and upload it.

  save_image(y, "deer_y.png", width, height);

  return 0;
}

