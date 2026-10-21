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

using namespace Eigen;

int main() {

  // • Load the image as an Eigen matrix A with size m x n. Each entry in the
  // matrix corresponds to a pixel on the screen and takes a value somewhere
  // between 0 (black) and 255 (white). Compute the matrix product A^T A and
  // report the euclidean norm of A^T A.  

  // • Solve the eigenvalue problem A^T A x = λ x using the proper solver
  // provided by the Eigen library. Report the two largest computed singular
  // values of A.

  // • Export matrix A^T A in the matrix market format and move it to the
  // lis-2.1.10_test folder. Using the proper iterative solver available in
  // the LIS library compute the largest eigenvalue of A^T A up to a tolerance
  // of 10^-10. Report the computed eigenvalue. Is the result in agreement with
  // the one obtained in the previous point?

  // • Find a shift μ ∈ R yielding an acceleration of the previous
  // eigensolver. Report μ and the number of iterations required to achieve a
  // tolerance of 10^-10.

  // • Using the SVD module of the Eigen library, perform a singular value
  // decomposition of the matrix A. Report the Euclidean norm of the diagonal
  // matrix Σ of the singular values.

  // • Compute the matrices C and D described in (1) assuming k = 30 and
  // k = 90. Report the number of nonzero entries in the matrices C and D.

  // • Compute the compressed images as the matrix product C D^T (again for
  // k = 30 and k = 90). Export and upload the resulting images in .png.

//===============================================================================

  // • Using Eigen create a black and white checkerboard image (as the one
  // depicted below) with height and width equal to 400 pixels. Report the
  // Euclidean norm of the matrix corresponding to the image.

  // • Introduce a noise into the checkerboard image by adding random
  // fluctuations of color ranging between [-60, 60] to each pixel. Export the
  // resulting image in .png and upload it.

  // • Using the SVD module of the Eigen library, perform a singular value
  // decomposition of the matrix corresponding to the noisy image. Report the
  // two largest computed singular values.

  // • Starting from the previously computed SVD, create the matrices C and D
  // defined in (1) assuming k = 6 and k = 12. Report the size of the matrices
  // C and D.

  // • Compute the compressed images as the matrix product C D^T (again for
  // k = 6 and k = 12). Export and upload the resulting images in .png.

  // • Compare the compressed images with the original and noisy images.
  // Comment the results.

  return 0;
}