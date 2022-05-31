/* arm-linux-gnueabihf-gcc main.c -static -mfpu=neon -ftree-vectorize -o main */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <complex.h>
#include "fft-real-pair.h"

/*-------------------------------------------------------------------------
   This computes an in-place complex-to-complex FFT
   x and y are the real and imaginary arrays of 2^m points.
   dir =  1 gives forward transform
   dir = -1 gives reverse transform

     Formula: forward
                  N-1
                  ---
              1   \          - j k 2 pi n / N
      X(n) = ---   >   x(k) e                    = forward transform
              N   /                                n=0..N-1
                  ---
                  k=0

      Formula: reverse
                  N-1
                  ---
                  \          j k 2 pi n / N
      X(n) =       >   x(k) e                    = forward transform
                  /                                n=0..N-1
                  ---
                  k=0
*/
int FFT(int dir, int m, double *x, double *y)
{
    long nn, i, i1, j, k, i2, l, l1, l2;
    double c1, c2, tx, ty, t1, t2, u1, u2, z;

    /* Calculate the number of points */
    nn = 1;
    for (i = 0; i < m; i++)
        nn *= 2;

    /* Do the bit reversal */
    i2 = nn >> 1;
    j = 0;
    for (i = 0; i < nn - 1; i++)
    {
        if (i < j)
        {
            tx = x[i];
            ty = y[i];
            x[i] = x[j];
            y[i] = y[j];
            x[j] = tx;
            y[j] = ty;
        }
        k = i2;
        while (k <= j)
        {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Compute the FFT */
    c1 = -1.0;
    c2 = 0.0;
    l2 = 1;
    for (l = 0; l < m; l++)
    {
        l1 = l2;
        l2 <<= 1;
        u1 = 1.0;
        u2 = 0.0;
        for (j = 0; j < l1; j++)
        {
            for (i = j; i < nn; i += l2)
            {
                i1 = i + l1;
                t1 = u1 * x[i1] - u2 * y[i1];
                t2 = u1 * y[i1] + u2 * x[i1];
                x[i1] = x[i] - t1;
                y[i1] = y[i] - t2;
                x[i] += t1;
                y[i] += t2;
            }
            z = u1 * c1 - u2 * c2;
            u2 = u1 * c2 + u2 * c1;
            u1 = z;
        }
        c2 = sqrt((1.0 - c1) / 2.0);
        if (dir == 1)
            c2 = -c2;
        c1 = sqrt((1.0 + c1) / 2.0);
    }

    /* Scaling for forward transform */
    if (dir == 1)
    {
        for (i = 0; i < nn; i++)
        {
            x[i] /= (double)nn;
            y[i] /= (double)nn;
        }
    }

    return (true);
}

/*-------------------------------------------------------------------------
   Calculate the closest but lower power of two of a number
   twopm = 2**m <= n
   Return true if 2**m == n
*/
int Powerof2(int n, int *m, int *twopm)
{
    if (n <= 1)
    {
        *m = 0;
        *twopm = 1;
        return (false);
    }

    *m = 1;
    *twopm = 2;
    do
    {
        (*m)++;
        (*twopm) *= 2;
    } while (2 * (*twopm) <= n);

    if (*twopm != n)
        return (false);
    else
        return (true);
}

/*-------------------------------------------------------------------------
   Perform a 2D FFT inplace given a complex 2D array
   The direction dir, 1 for forward, -1 for reverse
   The size of the array (nx,ny)
   Return false if there are memory problems or
      the dimensions are not powers of 2
*/
int FFT2D(double **cimag, double **crel, int nx, int ny, int dir)
{
    int i, j;
    int m, twopm;
    double *real, *imag;

    /* Transform the rows */
    real = (double *)malloc(nx * sizeof(double));
    imag = (double *)malloc(nx * sizeof(double));
    if (real == NULL || imag == NULL)
        return (false);
    if (!Powerof2(nx, &m, &twopm) || twopm != nx)
        return (false);
    for (j = 0; j < ny; j++)
    {
        for (i = 0; i < nx; i++)
        {
            real[i] = crel[i][j];
            imag[i] = cimag[i][j];
        }
        FFT(dir, m, real, imag);
        for (i = 0; i < nx; i++)
        {
            crel[i][j] = real[i];
            cimag[i][j] = imag[i];
        }
    }
    free(real);
    free(imag);

    /* Transform the columns */
    real = (double *)malloc(ny * sizeof(double));
    imag = (double *)malloc(ny * sizeof(double));
    if (real == NULL || imag == NULL)
        return (false);
    if (!Powerof2(ny, &m, &twopm) || twopm != ny)
        return (false);
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            real[j] = crel[i][j];
            imag[j] = cimag[i][j];
        }
        FFT(dir, m, real, imag);
        for (j = 0; j < ny; j++)
        {
            crel[i][j] = real[j];
            cimag[i][j] = imag[j];
        }
    }
    free(real);
    free(imag);

    return (true);
}

typedef struct
{
    char p;
    int format;
    int width;
    int height;
    int intensity;
    unsigned char **pixels;
} image;

void read_pgm_file(FILE *fin, image *img);
void write_pgm_file(FILE *fout, image *img);
void median_filter(image *img, int window);
void threshold(image *img, int threshold);
void morph_dilation(image *img);
void morph_erosion(image *img);
void homomorphic_filter(image *img);

int main(int argc, char *argv[])
{
    image img;

    FILE *fin = fopen(argv[1], "r");
    if (fin == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }

    FILE *fout = fopen(argv[2], "w");
    if (fout == 0)
    {
        fclose(fout);
        printf("Error, aborting.\n");
        return -1;
    }

    read_pgm_file(fin, &img);

    // median_filter(&img, 3);

    // threshold(&img, 0x7A);

    // morph_dilation(&img);

    // morph_erosion(&img);

    homomorphic_filter(&img);

    write_pgm_file(fout, &img);

    fclose(fin);
    fclose(fout);
    return 0;
}

void read_pgm_file(FILE *fin, image *img)
{
    fscanf(fin, "%c%d\n", &img->p, &img->format);
    fscanf(fin, "%d %d\n", &img->width, &img->height);
    fscanf(fin, "%d\n", &img->intensity);

    img->pixels = (unsigned char **)malloc(img->height * sizeof(unsigned char *));

    for (int i = 0; i < img->height; i++)
    {
        img->pixels[i] = (unsigned char *)malloc(img->width * sizeof(unsigned char));
    }

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            fread(&img->pixels[i][j], sizeof(unsigned char), 1, fin);
        }
    }
}

void write_pgm_file(FILE *fout, image *img)
{
    fprintf(fout, "P5\n");
    fprintf(fout, "%d %d\n", img->width, img->height);
    fprintf(fout, "%d\n", img->intensity);

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            fwrite(&img->pixels[i][j], sizeof(unsigned char), 1, fout);
        }
    }
}

unsigned char img_get_value(image *img, int height, int width)
{
    if (width >= 0 && width < img->width && height >= 0 && height < img->height)
    {
        return img->pixels[height][width];
    }
    else
    {
        return 0x0;
    }
}

double img_get_value_double(double **img, int height, int width)
{
    if (width >= 0 && width < 512 && height >= 0 && height < 512)
    {
        return img[width][height];
    }
    else
    {
        return 0;
    }
}

int compare(const void *a, const void *b)
{
    return (*(unsigned char *)a - *(unsigned char *)b);
}

void median_filter(image *img, int window)
{
    unsigned char *matrix;

    matrix = (unsigned char *)malloc(window * window * sizeof(unsigned char));

    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            int index = 0;

            for (int n = -1; n < +2; n++)
                for (int m = -1; m < +2; m++)
                    matrix[index++] = img_get_value(img, y + n, x + m);

            qsort(matrix, window * window, sizeof(unsigned char), compare);

            img->pixels[y][x] = matrix[4];
        }
    }
}

void threshold(image *img, int threshold)
{
    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            if (img->pixels[y][x] > threshold)
            {
                img->pixels[y][x] = 0xff;
            }
            else
            {
                img->pixels[y][x] = 0x0;
            }
        }
    }
}

void morph_dilation(image *img)
{
    unsigned char **img_copy = (unsigned char **)malloc(img->height * sizeof(unsigned char *));

    for (int i = 0; i < img->height; i++)
    {
        img_copy[i] = (unsigned char *)malloc(img->width * sizeof(unsigned char));
        memcpy(img_copy[i], img->pixels[i], img->width * sizeof(unsigned char));
    }

    for (int x = 1; x < (img->width - 1); x++)
    {
        for (int y = 1; y < (img->height - 1); y++)
        {
            if (img->pixels[y][x] == 0xff)
            {
                img_copy[y][x] = 0xff;
                img_copy[y - 1][x] = 0xff;
                img_copy[y + 1][x] = 0xff;
                img_copy[y][x - 1] = 0xff;
                img_copy[y][x + 1] = 0xff;
                img_copy[y - 1][x - 1] = 0xff;
                img_copy[y - 1][x + 1] = 0xff;
                img_copy[y + 1][x - 1] = 0xff;
                img_copy[y + 1][x + 1] = 0xff;
            }
        }
    }

    for (int i = 0; i < img->height; i++)
    {
        memcpy(img->pixels[i], img_copy[i], img->width * sizeof(unsigned char));
    }

    free(img_copy);
}

void morph_erosion(image *img)
{
    unsigned char **img_copy = (unsigned char **)malloc(img->height * sizeof(unsigned char *));

    for (int i = 0; i < img->height; i++)
    {
        img_copy[i] = (unsigned char *)malloc(img->width * sizeof(unsigned char));
        memcpy(img_copy[i], img->pixels[i], img->width * sizeof(unsigned char));
    }

    int sum = 0;

    for (int x = 1; x < (img->width - 1); x++)
    {
        for (int y = 1; y < (img->height - 1); y++)
        {
            sum = img->pixels[y][x] + img->pixels[y - 1][x] + img->pixels[y + 1][x] + img->pixels[y][x - 1] +
                  img->pixels[y][x + 1] + img->pixels[y - 1][x - 1] + img->pixels[y - 1][x + 1] +
                  img->pixels[y + 1][x - 1] + img->pixels[y + 1][x + 1];

            if (img->pixels[y][x] == 0xff && sum < 0x7F8)
            {
                img_copy[y][x] = 0x0;
            }
        }
    }

    for (int i = 0; i < img->height; i++)
    {
        memcpy(img->pixels[i], img_copy[i], img->width * sizeof(unsigned char));
    }

    free(img_copy);
}

void shift2(double **x, int width, int height)
{
    int m2, n2;
    int i, k;
    // complex x[m][n];
    double tmp13, tmp24;

    m2 = width / 2;  // half of row dimension
    n2 = height / 2; // half of column dimension

    // interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4

    for (i = 0; i < m2; i++)
    {
        for (k = 0; k < n2; k++)
        {
            tmp13 = x[i][k];
            x[i][k] = x[i + m2][k + n2];
            x[i + m2][k + n2] = tmp13;

            tmp24 = x[i + m2][k];
            x[i + m2][k] = x[i][k + n2];
            x[i][k + n2] = tmp24;
        }
    }
}


void homomorphic_filter(image *img)
{
    double **img_copy_real = (double **)malloc(img->width * sizeof(double *));
    double **img_copy_imag = (double **)malloc(img->width * sizeof(double *));

    for (int i = 0; i < img->width; i++)
    {
        img_copy_real[i] = (double *)malloc(img->height * sizeof(double));
        img_copy_imag[i] = (double *)malloc(img->height * sizeof(double));
    }

    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            img_copy_real[x][y] = log(img->pixels[y][x] + 1);
            img_copy_imag[x][y] = 0.0;
        }
    }

    FFT2D(img_copy_imag, img_copy_real, img->width, img->height, 1);
    shift2(img_copy_real, img->width, img->height);
    shift2(img_copy_imag, img->width, img->height);

    // gaussian high pass filter

    int core_x = (img->width / 2); // Spectrum center coordinates
    int core_y = (img->height / 2);
    int d0 = 50; // Filter radius
    double h;
    // Parameters:
    float rh = 3.0;
    float rl = 0.5;
    float c = 2;
    for (int i = 0; i < img->width; i++)
    {
        for (int j = 0; j < img->height; j++)
        {
            h = (rh - rl) * (1 - exp(-c * ((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) / (d0 * d0))) + rl;
            img_copy_real[i][j] = img_copy_real[i][j] * h;
            img_copy_imag[i][j] = img_copy_imag[i][j] * h;
        }
    }

    shift2(img_copy_imag, img->width, img->height);
    shift2(img_copy_real, img->width, img->height);

    FFT2D(img_copy_imag, img_copy_real, img->width, img->height, -1);

    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            img->pixels[y][x] = (unsigned char)exp(img_copy_real[x][y] + 1);
        }
    }
}