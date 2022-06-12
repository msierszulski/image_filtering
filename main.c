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
#include <time.h>
#include <arm_neon.h>

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
    int
    FFT(int dir, int m, float *x, float *y)
{
    long nn, i, i1, j, k, i2, l, l1, l2;
    float c1, c2, tx, ty, t1, t2, u1, u2, z;

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
            x[i] /= (float)nn;
            y[i] /= (float)nn;
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
int FFT2D(float **cimag, float **crel, int nx, int ny, int dir)
{
    int i, j;
    int m, twopm;
    float *real, *imag;

    /* Transform the rows */
    real = (float *)malloc(nx * sizeof(float));
    imag = (float *)malloc(nx * sizeof(float));
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
    real = (float *)malloc(ny * sizeof(float));
    imag = (float *)malloc(ny * sizeof(float));
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

int FFT2D_neon(float **cimag, float **crel, int nx, int ny, int dir)
{
    int i, j;
    int m, twopm;
    float *real, *imag;

    /* Transform the rows */
    real = (float *)malloc(nx * sizeof(float));
    imag = (float *)malloc(nx * sizeof(float));
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
    real = (float *)malloc(ny * sizeof(float));
    imag = (float *)malloc(ny * sizeof(float));
    if (real == NULL || imag == NULL)
        return (false);
    if (!Powerof2(ny, &m, &twopm) || twopm != ny)
        return (false);
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j += 4)
        {
            float32x4_t tmp1 = vld1q_f32(&crel[i][j]);
            float32x4_t tmp2 = vld1q_f32(&cimag[i][j]);
            vst1q_f32(&real[j], tmp1);
            vst1q_f32(&imag[j], tmp2);
        }
        FFT(dir, m, real, imag);
        for (j = 0; j < ny; j += 4)
        {
            float32x4_t tmp1 = vld1q_f32(&real[j]);
            float32x4_t tmp2 = vld1q_f32(&imag[j]);
            vst1q_f32(&crel[i][j], tmp1);
            vst1q_f32(&cimag[i][j], tmp2);
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
void median_filter_neon(image *img, int window);
void threshold(image *img, int threshold);
void threshold_neon(image *img, int threshold);
void morph_dilation(image *img);
void morph_dilation_neon(image *img);
void morph_erosion(image *img);
void morph_erosion_neon(image *img);
void homomorphic_filter(image *img);
void homomorphic_filter_neon(image *img);

int main(int argc, char *argv[])
{
    clock_t start_t, end_t;
    float total_t;

    image img1;
    FILE *fin = fopen(argv[1], "r");
    if (fin == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin, &img1);
    fclose(fin);

    image img2;
    FILE *fin2 = fopen(argv[1], "r");
    if (fin2 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin2, &img2);
    fclose(fin2);

    image img3;
    FILE *fin3 = fopen(argv[1], "r");
    if (fin3 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin3, &img3);
    fclose(fin3);

    image img4;
    FILE *fin4 = fopen(argv[1], "r");
    if (fin4 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin4, &img4);
    fclose(fin4);

    image img5;
    FILE *fin5 = fopen(argv[1], "r");
    if (fin5 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin5, &img5);
    fclose(fin5);

    image img6;
    FILE *fin6 = fopen(argv[1], "r");
    if (fin6 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin6, &img6);
    fclose(fin6);

    image img7;
    FILE *fin7 = fopen(argv[1], "r");
    if (fin7 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin7, &img7);
    fclose(fin7);

    image img8;
    FILE *fin8 = fopen(argv[1], "r");
    if (fin8 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin8, &img8);
    fclose(fin8);

    image img9;
    FILE *fin9 = fopen(argv[1], "r");
    if (fin9 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin9, &img9);
    fclose(fin9);

    image img10;
    FILE *fin10 = fopen(argv[1], "r");
    if (fin10 == 0)
    {
        printf("Error, aborting.\n");
        return -1;
    }
    read_pgm_file(fin10, &img10);
    fclose(fin10);

    FILE *fout1 = fopen("median.pgm", "w");
    FILE *fout2 = fopen("threshold.pgm", "w");
    FILE *fout3 = fopen("dilation.pgm", "w");
    FILE *fout4 = fopen("erosion.pgm", "w");
    FILE *fout5 = fopen("homomorphic.pgm", "w");
    FILE *fout6 = fopen("median_neon.pgm", "w");
    FILE *fout7 = fopen("threshold_neon.pgm", "w");
    FILE *fout8 = fopen("dilation_neon.pgm", "w");
    FILE *fout9 = fopen("erosion_neon.pgm", "w");
    FILE *fout10 = fopen("homomorphic_neon.pgm", "w");

    if (fout1 == 0 || fout2 == 0 || fout3 == 0 || fout4 == 0 || fout5 == 0 || fout6 == 0 || fout7 == 0 || fout8 == 0 || fout9 == 0 || fout10 == 0)
    {
        fclose(fout1);
        fclose(fout2);
        fclose(fout3);
        fclose(fout4);
        fclose(fout5);
        fclose(fout6);
        fclose(fout7);
        fclose(fout8);
        fclose(fout9);
        fclose(fout10);
        printf("Error, aborting.\n");
        return -1;
    }

    printf("-------------------------------------------------------\n");
    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    median_filter(&img1, 3);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(median) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    median_filter_neon(&img6, 3);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(median_neon) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold(&img2, 0x7A);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshold) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold_neon(&img7, 0x7A);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshhold_neon) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold(&img3, 0x7A);
    morph_dilation(&img3);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshold + dilation) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold_neon(&img8, 0x7A);
    morph_dilation_neon(&img8);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshold_neon + dilation_neon) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold(&img4, 0x7A);
    morph_erosion(&img4);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshold + erosion) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    threshold_neon(&img9, 0x7A);
    morph_erosion_neon(&img9);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(threshold_neon + erosion_neon) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    homomorphic_filter(&img5);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(homomorphic) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    start_t = clock();
    printf("Starting of the program, start_t = %ld\n", start_t);

    homomorphic_filter_neon(&img10);

    end_t = clock();
    printf("End of the big loop, end_t = %ld\n", end_t);
    total_t = (float)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("(homomorphic_neon) Total time taken by CPU: %f (s)\n", total_t);
    printf("-------------------------------------------------------\n");

    write_pgm_file(fout1, &img1);
    write_pgm_file(fout2, &img2);
    write_pgm_file(fout3, &img3);
    write_pgm_file(fout4, &img4);
    write_pgm_file(fout5, &img5);
    write_pgm_file(fout6, &img6);
    write_pgm_file(fout7, &img7);
    write_pgm_file(fout8, &img8);
    write_pgm_file(fout9, &img9);
    write_pgm_file(fout10, &img10);

    fclose(fout1);
    fclose(fout2);
    fclose(fout3);
    fclose(fout4);
    fclose(fout5);
    fclose(fout6);
    fclose(fout7);
    fclose(fout8);
    fclose(fout9);
    fclose(fout10);
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

float img_get_value_float(float **img, int height, int width)
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

#define vminmax_u8(a, b)                 \
    do                                   \
    {                                    \
        uint8x16_t minmax_tmp = (a);     \
        (a) = vminq_u8((a), (b));        \
        (b) = vmaxq_u8(minmax_tmp, (b)); \
    } while (0)

void median_filter_neon(image *img, int window)
{
    for (int j = 1; j < img->height - 1; j++)
    {
        for (int i = 1; i < img->width - 1; i += 16)
        {
            uint8x16_t q0, q1, q2, q3, q4, q5, q6, q7, q8;

            q0 = vld1q_u8(&img->pixels[j - 1][i - 1]);
            q1 = vld1q_u8(&img->pixels[j - 1][i]);
            q2 = vld1q_u8(&img->pixels[j - 1][i + 1]);
            q3 = vld1q_u8(&img->pixels[j][i - 1]);
            q4 = vld1q_u8(&img->pixels[j][i]);
            q5 = vld1q_u8(&img->pixels[j][i + 1]);
            q6 = vld1q_u8(&img->pixels[j + 1][i - 1]);
            q7 = vld1q_u8(&img->pixels[j + 1][i]);
            q8 = vld1q_u8(&img->pixels[j + 1][i + 1]);

            /* Paeth's 9-element sorting network */
            vminmax_u8(q0, q3);
            vminmax_u8(q1, q4);
            vminmax_u8(q0, q1);
            vminmax_u8(q2, q5);
            vminmax_u8(q0, q2);
            vminmax_u8(q4, q5);
            vminmax_u8(q1, q2);
            vminmax_u8(q3, q5);
            vminmax_u8(q3, q4);
            vminmax_u8(q1, q3);
            vminmax_u8(q1, q6);
            vminmax_u8(q4, q6);
            vminmax_u8(q2, q6);
            vminmax_u8(q2, q3);
            vminmax_u8(q4, q7);
            vminmax_u8(q2, q4);
            vminmax_u8(q3, q7);
            vminmax_u8(q4, q8);
            vminmax_u8(q3, q8);
            vminmax_u8(q3, q4);

            // q4 now - median values
            vst1q_u8(&img->pixels[j][i], q4);
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

void threshold_neon(image *img, int threshold)
{
    uint8x16_t thresholdvec = vdupq_n_u8(threshold);

    for (int j = 0; j < img->height; j++)
    {
        for (int i = 0; i < img->width; i += 16)
        {
            uint8x16_t inputNeon = vld1q_u8(&img->pixels[j][i]);
            uint8x16_t partialResult = vcgeq_u8(thresholdvec, inputNeon);
            partialResult = vcleq_u8(thresholdvec, inputNeon);

            vst1q_u8(&img->pixels[j][i], partialResult);
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

void morph_dilation_neon(image *img)
{
    unsigned char **img_copy = (unsigned char **)malloc(img->height * sizeof(unsigned char *));

    for (int i = 0; i < img->height; i++)
    {
        img_copy[i] = (unsigned char *)malloc(img->width * sizeof(unsigned char));
        memcpy(img_copy[i], img->pixels[i], img->width * sizeof(unsigned char));
    }

    int wing = 1;
    for (int y = wing; y < img->height - wing - 1; y++)
    {
        for (int x = 0; x < img->width; x += 16)
        {
            uint8x16_t val = vld1q_u8(img->pixels[y - wing + 1] + x);
            for (int k = -wing + 2; k <= wing; k++)
                val = vmaxq_u8(val, vld1q_u8(img->pixels[y + k] + x));
            vst1q_u8(img_copy[y] + x, vmaxq_u8(val, vld1q_u8(img->pixels[y - wing] + x)));
            vst1q_u8(img_copy[y + 1] + x, vmaxq_u8(val, vld1q_u8(img->pixels[y + wing + 1] + x)));
        }
    }
    for (int y = 0; y < img->height; ++y)
    {
        for (int x = 0; x < img->width; x += 16)
        {
            uint8x16_t val = vld1q_u8(img_copy[y] + x - wing);
            for (int j = x - wing + 1; j <= x + wing; ++j)
                val = vmaxq_u8(val, vld1q_u8(img_copy[y] + j));
            vst1q_u8(img_copy[y] + x, val);
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

            if (img->pixels[y][x] == 0xff && sum < 0x8f7) // 0x7F8)
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

void morph_erosion_neon(image *img)
{
    unsigned char **img_copy = (unsigned char **)malloc(img->height * sizeof(unsigned char *));

    for (int i = 0; i < img->height; i++)
    {
        img_copy[i] = (unsigned char *)malloc(img->width * sizeof(unsigned char));
        memcpy(img_copy[i], img->pixels[i], img->width * sizeof(unsigned char));
    }

    int wing = 1;
    for (int y = wing; y < img->height - wing - 1; y++)
    {
        for (int x = 0; x < img->width; x += 16)
        {
            uint8x16_t val = vld1q_u8(img->pixels[y - wing + 1] + x);
            for (int k = -wing + 2; k <= wing; k++)
                val = vminq_u8(val, vld1q_u8(img->pixels[y + k] + x));
            vst1q_u8(img_copy[y] + x, vminq_u8(val, vld1q_u8(img->pixels[y - wing] + x)));
            vst1q_u8(img_copy[y + 1] + x, vminq_u8(val, vld1q_u8(img->pixels[y + wing + 1] + x)));
        }
    }
    for (int y = 0; y < img->height; ++y)
    {
        for (int x = 0; x < img->width; x += 16)
        {
            uint8x16_t val = vld1q_u8(img_copy[y] + x - wing);
            for (int j = x - wing + 1; j <= x + wing; ++j)
                val = vminq_u8(val, vld1q_u8(img_copy[y] + j));
            vst1q_u8(img_copy[y] + x, val);
        }
    }

    for (int i = 0; i < img->height; i++)
    {
        memcpy(img->pixels[i], img_copy[i], img->width * sizeof(unsigned char));
    }

    free(img_copy);
}

void shift2(float **x, int width, int height)
{
    int m2, n2;
    int i, k;
    // complex x[m][n];
    float tmp13, tmp24;

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

void shift2_neon(float **x, int width, int height)
{
    int m2, n2;
    int i, k;
    // complex x[m][n];
    // float tmp13, tmp24;

    m2 = width / 2;  // half of row dimension
    n2 = height / 2; // half of column dimension

    // interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4

    for (i = 0; i < m2; i++)
    {
        for (k = 0; k < n2; k += 4)
        {
            float32x4_t tmp13 = vld1q_f32(&x[i][k]);
            float32x4_t tmp14 = vld1q_f32(&x[i + m2][k + n2]);
            vst1q_f32(&x[i][k], tmp14);
            vst1q_f32(&x[i + m2][k + n2], tmp13);

            float32x4_t tmp24 = vld1q_f32(&x[i + m2][k]);
            float32x4_t tmp25 = vld1q_f32(&x[i][k + n2]);
            vst1q_f32(&x[i + m2][k], tmp25);
            vst1q_f32(&x[i][k + n2], tmp24);
        }
    }
}

void homomorphic_filter(image *img)
{
    float **img_copy_real = (float **)malloc(img->width * sizeof(float *));
    float **img_copy_imag = (float **)malloc(img->width * sizeof(float *));
    float **h = (float **)malloc(img->width * sizeof(float *));

    for (int i = 0; i < img->width; i++)
    {
        img_copy_real[i] = (float *)malloc(img->height * sizeof(float));
        img_copy_imag[i] = (float *)malloc(img->height * sizeof(float));
        h[i] = (float *)malloc(img->height * sizeof(float));
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
    // Parameters:
    float rh = 3.0;
    float rl = 0.5;
    float c = 2;
    for (int i = 0; i < img->width; i++)
    {
        for (int j = 0; j < img->height; j++)
        {
            h[i][j] = (rh - rl) * (1 - exp(-c * ((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) / (d0 * d0))) + rl;
        }
    }

    for (int i = 0; i < img->width; i++)
    {
        for (int j = 0; j < img->height; j++)
        {
            img_copy_real[i][j] = img_copy_real[i][j] * h[i][j];
            img_copy_imag[i][j] = img_copy_imag[i][j] * h[i][j];
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

void homomorphic_filter_neon(image *img)
{
    float **img_copy_real = (float **)malloc(img->width * sizeof(float *));
    float **img_copy_imag = (float **)malloc(img->width * sizeof(float *));
    float **h = (float **)malloc(img->width * sizeof(float *));

    for (int i = 0; i < img->width; i++)
    {
        img_copy_real[i] = (float *)malloc(img->height * sizeof(float));
        img_copy_imag[i] = (float *)malloc(img->height * sizeof(float));
        h[i] = (float *)malloc(img->height * sizeof(float));
    }

    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            img_copy_real[x][y] = log(img->pixels[y][x] + 1);
            img_copy_imag[x][y] = 0.0;
        }
    }

    FFT2D_neon(img_copy_imag, img_copy_real, img->width, img->height, 1);
    shift2_neon(img_copy_real, img->width, img->height);
    shift2_neon(img_copy_imag, img->width, img->height);

    // gaussian high pass filter
    int core_x = (img->width / 2); // Spectrum center coordinates
    int core_y = (img->height / 2);
    int d0 = 50; // Filter radius
    // float h;
    // Parameters:
    float rh = 3.0;
    float rl = 0.5;
    float c = 2;
    for (int i = 0; i < img->width; i++)
    {
        for (int j = 0; j < img->height; j++)
        {
            h[i][j] = (rh - rl) * (1 - exp(-c * ((i - core_x) * (i - core_x) + (j - core_y) * (j - core_y)) / (d0 * d0))) + rl;
        }
    }

    for (int i = 0; i < img->width; i++)
    {
        for (int j = 0; j < img->height; j += 4)
        {
            float32x4_t img_copy_real_vec = vld1q_f32(&img_copy_real[i][j]);
            float32x4_t img_copy_imag_vec = vld1q_f32(&img_copy_imag[i][j]);
            float32x4_t gauss = vld1q_f32(&h[i][j]);
            float32x4_t partialResultreal = vmulq_f32(img_copy_real_vec, gauss);
            float32x4_t partialResultimag = vmulq_f32(img_copy_imag_vec, gauss);

            vst1q_f32(&img_copy_real[i][j], partialResultreal);
            vst1q_f32(&img_copy_imag[i][j], partialResultimag);
        }
    }

    shift2_neon(img_copy_imag, img->width, img->height);
    shift2_neon(img_copy_real, img->width, img->height);

    FFT2D_neon(img_copy_imag, img_copy_real, img->width, img->height, -1);

    for (int x = 0; x < img->width; x++)
    {
        for (int y = 0; y < img->height; y++)
        {
            img->pixels[y][x] = (unsigned char)exp(img_copy_real[x][y] + 1);
        }
    }
}