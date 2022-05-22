/* arm-linux-gnueabihf-gcc main.c -static -mfpu=neon -ftree-vectorize -o main */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} rgb;

typedef struct
{
    char p;
    int format;
    int width;
    int height;
    int intensity;
    rgb **pixels;
} image;

void read_ppm_file(FILE *fin, image *img);
void write_ppm_file(FILE *fout, image *img);
void median_filter(image *img);

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

    read_ppm_file(fin, &img);

    median_filter(&img);

    write_ppm_file(fout, &img);

    fclose(fin);
    fclose(fout);
    return 0;
}

void read_ppm_file(FILE *fin, image *img)
{
    fscanf(fin, "%c%d\n", &img->p, &img->format);
    fscanf(fin, "%d %d\n", &img->width, &img->height);
    fscanf(fin, "%d\n", &img->intensity);

    img->pixels = (rgb **)malloc(img->height * sizeof(rgb *));

    for (int i = 0; i < img->height; i++)
    {
        img->pixels[i] = (rgb *)malloc(img->width * sizeof(rgb));
    }

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            fread(&img->pixels[i][j], sizeof(unsigned char), 3, fin);
        }
    }
}

void write_ppm_file(FILE *fout, image *img)
{
    fprintf(fout, "P6\n");
    fprintf(fout, "%d %d\n", img->width, img->height);
    fprintf(fout, "%d\n", img->intensity);

    for (int i = 0; i < img->height; i++)
    {
        for (int j = 0; j < img->width; j++)
        {
            fwrite(&img->pixels[i][j], sizeof(unsigned char), 3, fout);
        }
    }
}

void median_filter(image *img)
{
    for (int y = 0; y < img->height; y++)
    {
        for (int x = 0; x < img->width; x++)
        {
            printf("r: %d g: %d b: %d", img->pixels[y][x].r, img->pixels[y][x].g, img->pixels[y][x].b)
        }
        printf("\n");
    }
}
