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

#include <unsupported/Eigen/SparseExtra>

typedef Eigen::SparseMatrix<double> SpMat;
using namespace Eigen;

// note that this works only for 1 channel images (grayscale)
// removed the const because for non const vector the vector would have been passed as a copy
void save_image(VectorXd& image, const char* filename, int width, int height)
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

    original.resize(height, width); // Right size!  

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

void generate_checkerboard(VectorXd &checkerboard, int c_size, int row_len, int tile_height, int tile_size)
{

    bool black = true;

    for (int i = 0; i <= c_size; ++i)
    {
        // everything is a const or constexpr so this gotta be FAST.
        if (i % tile_size == 0 && i % tile_height != 0) black = !black;

        if (black)
        {
            checkerboard.coeffRef(i) = 0.0;
            continue;
        }
        checkerboard.coeffRef(i) = 255.0;    

    }

}

int main(int argc, char* argv[])
{

    int width, height, channels;
    const char* image_name = "deer.jpg";

    // • Load the image as an Eigen matrix A with size m x n. Each entry in the
    // matrix corresponds to a pixel on the screen and takes a value somewhere
    // between 0 (black) and 255 (white). Compute the matrix product A^T A and
    // report the euclidean norm of A^T A.  

    MatrixXd A;   
    load_image(image_name, width, height, channels, A);

    std::cout << "Size of A (m, n): (" << A.rows() << ", " << A.cols() << ")\n";

    // Compute A^T A, n x n, symmetric
    MatrixXd ATA = A.transpose() * A;

    std::cout << "Size of A^T A: (" << ATA.rows() << ", " << ATA.cols() << ")\n";
    std::cout << "Euclidean norm of A^T A: " << ATA.norm() << "\n";

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

    const unsigned int row_len = 400;
    const unsigned int col_len = 400;
    const unsigned int tile_size = 40;

    constexpr unsigned int c_size = row_len * col_len;
    constexpr unsigned int tile_height = row_len * tile_size;
    VectorXd checkerboard(c_size);
    generate_checkerboard(checkerboard, c_size, row_len, tile_height, tile_size);
    // save_image(checkerboard, "checkerboard.png", row_len, col_len);
    std::cout << "Euclidean norm of the plain checkerboard: " << checkerboard.norm() << "\n";

    // • Introduce a noise into the checkerboard image by adding random
    // fluctuations of color ranging between [-60, 60] to each pixel. Export the
    // resulting image in .png and upload it.

    const int noise_bounds = 60;
    VectorXd noisy = checkerboard;
    noisy += VectorXd::Random(c_size) * noise_bounds;
    // the clamping is inside save_image, so we don't need to worry about it here
    save_image(noisy, "noisy_checkerboard.png", row_len, col_len);

    // • Using the SVD module of the Eigen library, perform a singular value
    // decomposition of the matrix corresponding to the noisy image. Report the
    // two largest computed singular values.
    
    MatrixXd noisy_mat = noisy.reshaped<RowMajor>(row_len, col_len);

    // the computationOptions are here to get the values of U and V. we can choose between full and thin but
    // since our checkerboard is square it doesn't change anything
    BDCSVD<MatrixXd> svd(noisy_mat, ComputeThinU | ComputeThinV);

    MatrixXd U = svd.matrixU();
    VectorXd S = svd.singularValues();
    // MatrixXd S_mat = S.asDiagonal();
    MatrixXd V = svd.matrixV();    

    // do NOT print this, it's gonna spam.
    // std::cout << "Singular values of the noisy checkerboard: " << S;
    std::cout << "The two largest singular values are " << S[0] << " and " << S[1] << "\n"; 

    // • Starting from the previously computed SVD, create the matrices C and D
    // defined in (1) assuming k = 6 and k = 12. Report the size of the matrices
    // C and D.

    // • Compute the compressed images as the matrix product C D^T (again for
    // k = 6 and k = 12). Export and upload the resulting images in .png.

    // • Compare the compressed images with the original and noisy images.
    // Comment the results.

    return 0;

}