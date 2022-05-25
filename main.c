/* arm-linux-gnueabihf-gcc main.c -static -mfpu=neon -ftree-vectorize -o main */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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

void homomorphic_filter(image *img)
{
    double **img_copy = (double **)malloc(img->height * sizeof(double *));

    for (int i = 0; i < img->height; i++)
    {
        img_copy[i] = (double *)malloc(img->width * sizeof(double));
    }

    for (int x = 1; x < (img->width - 1); x++)
    {
        for (int y = 1; y < (img->height - 1); y++)
        {
            img_copy[y][x] = log(img->pixels[y][x]);
            printf("%f ", img_copy[y][x]);
        }
        printf("\n");
    }
}