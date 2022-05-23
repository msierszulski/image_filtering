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

    median_filter(&img, 3);

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

unsigned char img_get_value(image *img, int width, int height)
{
    if (width >= 0 && width < img->width && height >= 0 && height < img->height)
    {
        return img->pixels[height][width];
    }
    else
    {
        return '\0';
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

    for (int y = 0; y < img->width; y++)
    {
        for (int x = 0; x < img->height; x++)
        {
            int index = 0;

            for (int n = -1; n < +2; n++)
                for (int m = -1; m < +2; m++)
                    matrix[index++] = img_get_value(img, y + m, x + n);

            qsort(matrix, window * window, sizeof(unsigned char), compare);

            img->pixels[x][y] = matrix[4];
        }
    }
}
