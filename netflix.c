#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
// #include <mpi.h>

void get_walltime(double* wcTime);
void get_walltime_(double* wcTime);
void dump_3d(int c, int r, int z, double* array);
void dump_3d_linear(int length, double* array);
void generateRandomMatrix(int length, double* array);
double* createMatrix(int numElements);
float computeEui(int u, int i, int r, int f, double* q, double* p, double* megaMatrix);

int main(int argc, char *argv[]){

	int numProcessors, rank;

/* initialize MPI variables */
/*    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
*/

    double startTime, endTime;
    int i, u, f, y, r;
    int megaSize = 99072112 * 3;
    int n = 0;
    int m = 0;
    int factorCount = 32;
    double llamda = 0.005;
    double gamma = 0.00125;
    double errorLimit = 0.001;
    float e = 0;
    float totalError = 0;
    double* megaMatrix = createMatrix(megaSize);

    //open files
    FILE *input;
    if(argv[2] == NULL || argv[1] == NULL){
    	printf("not enough arguments");
    	exit(0);
    }
    input = fopen(argv[1], "r");
	if(!input){
        fclose(input);
        return 0;
	}

	FILE *output;
	output = fopen(argv[2], "w");
	if(!output){
        fclose(input);
        fclose(output);
        return 0;
	}

    int maxLineLength = 30;
	char* line = (char*)malloc(maxLineLength);
	assert(line);

    int count = 0;

	while(fgets(line, maxLineLength, input) != NULL){
		char* user = strtok(line, " ");
		char* movie = strtok(NULL, " ");
		char* rating = strtok(NULL, " ");
		int userNum = strtod(user, NULL);
		int movieNum = strtod(movie, NULL);
		int ratingNum = strtod(rating, NULL);
		if(userNum > n && movieNum > m){
			n = userNum;
			m = movieNum;
		}else if(movieNum > m){
			m = movieNum;
		}else if(userNum > n){
			n = userNum;
		}
        megaMatrix[count] = userNum;
        megaMatrix[count+1] = movieNum;
        megaMatrix[count+2] = ratingNum;
        count += 3;
	}

	free(line);

	double* q = createMatrix(m*factorCount);
	for(y = 0; y < m*factorCount; y++){
		q[y] = 1;
	}

	double* p = createMatrix(n*factorCount);
	for(y = 0; y < n*factorCount; y++){
		p[y] = 1;
	}

    float oldError = 0;
    
    get_walltime(&startTime);
	while(oldError-totalError > errorLimit || !oldError || oldError-totalError < 0){
		oldError = totalError;
		totalError = 0;
		#pragma omp parallel for schedule(guided,1)
		for(y = 0; y < megaSize; y+=3){
			u = megaMatrix[y];
			i = megaMatrix[y+1];
			r = megaMatrix[y+2];
        	e = computeEui(u, i, r, factorCount, q, p, megaMatrix);
        	double qNorm = 0;
        	double pNorm = 0;
        	for(f = 0; f < factorCount; f++){
        		double qi = q[i*factorCount+f];
        		double pu = p[u*factorCount+f];
        		qNorm += qi*qi;
        		pNorm += pu*pu;
        		q[i*factorCount+f] += gamma*(e*pu-(llamda*qi));
        		p[u*factorCount+f] += gamma*(e*qi-(llamda*pu));
        	}
        	e = e*e;
            double noise = llamda*(qNorm+pNorm);
        	totalError += (e+noise);
        }
//        printf("Minimization: %f\n", totalError);
	}
	get_walltime(&endTime);

    /* write output */
        fprintf(output, "user: 427501 movie:  8 rating: %f\n", e);
        fprintf(output, "user: 260749 movie: 18 rating: %f\n", e);
        fprintf(output, "user: 311872 movie: 28 rating: %f\n", e);
        fprintf(output, "user:  73318 movie: 30 rating: %f\n", e);
        fprintf(output, "user: 182071 movie: 30 rating: %f\n", e);


    /* get time elapsed and gigaflop rate */
    double timeElapsed = endTime-startTime; 
    printf("elapsed time: %f", timeElapsed);
    printf("\n");
    printf("\n");

    // free matrix pointers
//    free(megaMatrix);
//    free(q);
//    free(p);

    // close files
    fclose(input);
    fclose(output);

    /* Finalize MPI */
    //MPI_Finalize();
}

float computeEui(int u, int i, int r, int f, double* q, double* p, double* megaMatrix){
	int a;
	float Eui = 0;
	for(a = 0; a < f; a++){
        Eui += q[i*f+a] * p[u*f+a];
	}
	Eui = r - Eui;
	return Eui;
}

void dump_3d(int c, int r, int z, double* array){
    printf("I am computing d!\n");
    int i, j, k;
    for(i = 0; i < z; i++){
        for(k = 0; k < c; k++){
            for(j = 0; j < r; j++){
                printf("%f ",array[(i*c+k) *r + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void dump_3d_linear(int length, double* array){
    int i;
    for(i = 0; i < length; i++){
        printf("%f ", array[i]);
    }
}

void generateRandomMatrix(int length, double* array){
    int i;
    for(i = 0; i < length; i++){
        array[i] = rand();
    }
}

double* createMatrix(int numElements){
    double* array = (double*)calloc(numElements,sizeof(double));
    assert(array);
    return array;
}

void get_walltime_(double* wcTime){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
}

void get_walltime(double* wcTime){
    get_walltime_(wcTime);
}
