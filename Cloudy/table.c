#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <errno.h>
#include "table.h"

double* Allocate_vect(const int dim)
{
    double *vect;
    vect = (double*)malloc(dim*sizeof(double));
    return vect;
}

double*** Allocate_matrix(const int dim1,const int dim2, const int dim3)
{
    int i,j;
    double *** mat = (double ***)malloc(dim1*sizeof(double**));

    for (i = 0; i< dim1; i++) {
        mat[i] = (double **) malloc(dim2*sizeof(double *));
        for (j = 0; j < dim2; j++) {
            mat[i][j] = (double *)malloc(dim3*sizeof(double));
        }
    }
    return mat;
}

void Erase_vect(double *vect)
{
    free(vect);
    vect = NULL;
}

size_t my_fread(void *ptr, size_t size, size_t dim, FILE * infile)
{
    size_t nread;

    if(!infile)
        return 0;

    if(size * dim > 0)
    {
        if((nread = fread(ptr, size, dim, infile)) != dim)
        {
            if(feof(infile))
                printf("Error in my_fread: end of file reached\n");
            else
                printf("Error in my_fread: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
    }
    else
        nread = 0;
    return nread;
}

void ReadFile2(char *filename)
{
    FILE *fp;
    int i, j, k; 
    double *nh_in, *Z_in, *T_in, *heat_rate, *cool_rate, *molwgt, *rho_in; 
    double *cooling_net,*heating_net;
    //double ***molw, ***cool_net;
    //double T_delta, Z_delta, rho_delta;
    int nh_dim, Z_dim = 4, T_dim = 61;

    fp = fopen(filename,"r");

    int ch=0;
    int row_dim=-1;
    while ((ch = fgetc(fp)) != EOF)
    {
        if (ch == '\n')
            row_dim++;
    }
    fseek(fp,0,SEEK_SET);

    nh_dim = row_dim/(Z_dim*T_dim);

    double mh=1.66e-24;

    nh_in = Allocate_vect(row_dim);
    Z_in = Allocate_vect(row_dim);
    T_in = Allocate_vect(row_dim); 
    heat_rate = Allocate_vect(row_dim);
    cool_rate = Allocate_vect(row_dim);
    molwgt = Allocate_vect(row_dim);
    rho_in = Allocate_vect(row_dim);
    cooling_net = Allocate_vect(row_dim);
    heating_net = Allocate_vect(row_dim);

    fscanf(fp,"%*[^\n]"); //salta la prima riga, commenti

    for(i=0; i<row_dim; i++)
    {
        fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf", &nh_in[i], &Z_in[i], &T_in[i], &heat_rate[i], &cool_rate[i], &molwgt[i], &rho_in[i]);
    }

    fclose(fp);

    ab_H[0] = pow(10.,nh_in[0])*mh/rho_in[0];
    ab_H[1] = pow(10.,nh_in[T_dim])*mh/rho_in[T_dim];
    ab_H[2] = pow(10.,nh_in[T_dim*2])*mh/rho_in[T_dim*2];
    ab_H[3] = pow(10.,nh_in[T_dim*3])*mh/rho_in[T_dim*3];
    //printf("%e %e %e %e \n",ab_H[0], ab_H[1], ab_H[2],ab_H[3]); 

    for(i=0; i<row_dim; i++)
    {
        heating_net[i] = heat_rate[i]/pow(rho_in[i]*1e24,2.);///(pow(10.,nh_in[i]+nh_in[i]));
        cooling_net[i] = cool_rate[i]/pow(rho_in[i]*1e24,2.);///(pow(10.,nh_in[i]+nh_in[i]));
                                                             //printf("%e %d\n",cooling_net[i],i); 
    }

    molw = Allocate_matrix(T_dim, Z_dim, nh_dim);
    cool_net = Allocate_matrix(T_dim, Z_dim, nh_dim);
    heat_net = Allocate_matrix(T_dim, Z_dim, nh_dim);

    for(k=0; k<nh_dim; k++)
    {
        for(j=0; j<Z_dim; j++)
        {
            for(i=0; i<T_dim; i++)        
            {
                int l = i + j * T_dim + k * T_dim * Z_dim;
                cool_net[i][j][k] = log10(cooling_net[l]);
                heat_net[i][j][k] = log10(heating_net[l]);
                molw[i][j][k] = molwgt[l];
            }
        }
    }

    T_delta = log10(T_in[1])-log10(T_in[0]);  
    //Z_delta = Z_in[T_dim] - Z_in[0];
    Z_delta = 0.5;
    rho_delta = nh_in[T_dim * Z_dim]-nh_in[0]; 

    temper_zero = log10(T_in[0]);
    //met_zero = Z_in[0];
    met_zero = -1.0;
    rho_zero = nh_in[0];

    Erase_vect(nh_in);
    Erase_vect(Z_in);
    Erase_vect(T_in);
    Erase_vect(heat_rate);
    Erase_vect(cool_rate);
    Erase_vect(molwgt);
    Erase_vect(rho_in);
    Erase_vect(cooling_net);
    Erase_vect(heating_net);

}
