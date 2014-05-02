#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

void get_walltime(double* wcTime);
void get_walltime_(double* wcTime);
void dump_3d(int c, int r, int z, double* array);
void dump_3d_linear(int length, double* array);
void generateRandomMatrix(int length, double* array);
double* createMatrix(int numElements);
float computeEui(int position, int r, int f, double* q, double* p, double* megaMatrix);
float computeEui2(int i, int u, int r, int f, double* q, double* p, double* megaMatrix);
float dotProduct(double* q, double* p, int i, int u, int f);

int main(int argc, char *argv[]){

	int numProcessors, rank;

/* initialize MPI variables */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int const rootRank = 0;


    double startTime, endTime;
    int i, u, f, y, r, z, x, qLength, pLength, chunkSize;
    int megaSize = 99072112 * 3;
    int n = 0;
    int m = 0;
    int factorCount = 32;
    int factorCountPlusOne = factorCount+1;
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

    if(rootRank == rank){
        int maxLineLength = 30;
    	char* line = (char*)malloc(maxLineLength);
    	assert(line);

        int count = 0;

    	while(fgets(line, maxLineLength, input) != NULL && count < megaSize){
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
    }

    if(rootRank == rank){
        chunkSize = megaSize / (numProcessors-1);
        int whichChunk = 0;
        for(y = 0; y < numProcessors; y++){
            if(y != rootRank){
                MPI_Send(megaMatrix+(whichChunk*chunkSize), chunkSize, MPI_DOUBLE, y, 0, MPI_COMM_WORLD);
                whichChunk++;
            }
        }
    }else{
        megaSize = megaSize / (numProcessors-1);
        MPI_Recv(megaMatrix, megaSize, MPI_DOUBLE, rootRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    qLength = m*(factorCountPlusOne);
    int qInitialGuess = 2.5/f;
	double* q = createMatrix(qLength);
	for(y = 0; y < qLength; y++){
		q[y] = qInitialGuess;
	}

    pLength = n*(factorCountPlusOne);
	double* p = createMatrix(pLength);
	for(y = 0; y < pLength; y++){
		p[y] = qInitialGuess;
	}

    double* qHolder = createMatrix(qLength);
    double* pHolder = createMatrix(pLength);

    float oldError = 0;
    
    if(rootRank == rank){
        get_walltime(&startTime);
    }

    if(rootRank != rank){
    	while(oldError-totalError > errorLimit || !oldError || oldError-totalError < 0){
    		oldError = totalError;
    		totalError = 0;
            int positionCounter = 0;
            #pragma omp parallel for schedule(guided,1)
    		for(y = 0; y < megaSize; y+=3){
    			u = megaMatrix[y];
    			i = megaMatrix[y+1];
    			r = megaMatrix[y+2];
            	e = computeEui(positionCounter, r, factorCount, q, p, megaMatrix);
            	double qNorm = 0;
            	double pNorm = 0;
                q[positionCounter*(factorCountPlusOne)] = i;
                p[positionCounter*(factorCountPlusOne)] = u;
            	for(f = 1; f < (factorCountPlusOne); f++){
            		double qi = q[positionCounter*(factorCountPlusOne)+f];
            		double pu = p[positionCounter*(factorCountPlusOne)+f];
            		qNorm += qi*qi;
            		pNorm += pu*pu;
            		q[positionCounter*(factorCountPlusOne)+f] += gamma*(e*pu-(llamda*qi));
            		p[positionCounter*(factorCountPlusOne)+f] += gamma*(e*qi-(llamda*pu));
            	}
            	e = e*e;
                double noise = llamda*(qNorm+pNorm);
            	totalError += (e+noise);
                positionCounter++;
            }
        }
    }


    if(rootRank != rank){
        MPI_Send(q, qLength, MPI_DOUBLE, rootRank, 0, MPI_COMM_WORLD);
    }else{
        for(y = 0; y < numProcessors; y++){
            if(y != rank){
                MPI_Recv(qHolder, qLength, MPI_DOUBLE, y, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for(z = 0; z < chunkSize / (numProcessors-1); z++){
                    int firstIndexOfRow = z*(factorCountPlusOne);
                    double movieVal = qHolder[firstIndexOfRow];
                    for(x = 1; x < factorCountPlusOne; x++){
                        q[(int)movieVal*(factorCountPlusOne)+x] = qHolder[firstIndexOfRow+x];
                    }
                }
            }
        }
    }

    if(rootRank != rank){
        MPI_Send(p, pLength, MPI_DOUBLE, rootRank, 0, MPI_COMM_WORLD);
    }else{
        for(y = 0; y < numProcessors; y++){
            if(y != rank){
                MPI_Recv(pHolder, pLength, MPI_DOUBLE, y, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for(z = 0; z < chunkSize / (numProcessors-1); z++){
                    int firstIndexOfRow = z*(factorCountPlusOne);
                    double userVal = pHolder[firstIndexOfRow];
                    for(x = 1; x < factorCountPlusOne; x++){
                        p[(int)(userVal-1)*(factorCountPlusOne)+x] = pHolder[firstIndexOfRow+x];
                    }
                }
            }
        }
    }
   
    if(rootRank == rank){
        oldError = 0;
        totalError = 0;
        while(oldError-totalError > errorLimit || !oldError || oldError-totalError < 0){
            oldError = totalError;
            totalError = 0;
            int positionCounter = 0;
            #pragma omp parallel for schedule(guided,1)
            for(y = 0; y < n; y++){
                for(z = 0; z < m; z++){
                    e = computeEui2(z, y, megaMatrix[n*3+2], factorCount, q, p, megaMatrix);
                    int qNorm = 0;
                    int pNorm = 0;
                    for(f = 1; f < (factorCountPlusOne); f++){
                        double qi = q[z*(factorCountPlusOne)+f];
                        double pu = p[y*(factorCountPlusOne)+f];
                        qNorm += qi*qi;
                        pNorm += pu*pu;
                        q[z*(factorCountPlusOne)+f] += gamma*(e*pu-(llamda*qi));
                        p[y*(factorCountPlusOne)+f] += gamma*(e*qi-(llamda*pu));
                    }
                    e = e*e;
                    double noise = llamda*(qNorm+pNorm);
                    totalError += (e+noise);
                }
            }
        }

        get_walltime(&endTime);
 
        /* get time elapsed and gigaflop rate */
        double timeElapsed = endTime-startTime; 
        printf("elapsed time: %f", timeElapsed);
        printf("\n");
        printf("\n");

        /* write output */
        fprintf(output, "user: 427501 movie:  8 rating: %f\n", dotProduct(q, p, 8, 427501, factorCount));
        fprintf(output, "user: 260749 movie: 18 rating: %f\n", dotProduct(q, p, 18, 260749, factorCount));
        fprintf(output, "user: 311872 movie: 28 rating: %f\n", dotProduct(q, p, 28, 311872, factorCount));
        fprintf(output, "user:  73318 movie: 30 rating: %f\n", dotProduct(q, p, 30, 73318, factorCount));
        fprintf(output, "user: 182071 movie: 30 rating: %f\n", dotProduct(q, p, 30, 182071, factorCount));

  
    }
    // free matrix pointers
    free(qHolder);
    free(pHolder);
    free(megaMatrix);
    free(q);
    free(p);

    // close files
    fclose(input);
    fclose(output);

    /* Finalize MPI */
    MPI_Finalize();
}

float dotProduct(double* q, double* p, int i, int u, int f){
    float total = 0;
    int z;
    for(z = 1; z < (f+1); z++){
        total += q[(i-1)*(f+1)+z] * p[(u-1)*(f+1)+z];
    }
    return total;
}

float computeEui(int position, int r, int f, double* q, double* p, double* megaMatrix){
	int a;
	float Eui = 0;
	for(a = 1; a < (f+1); a++){
        Eui += q[position*(f+1)+a] * p[position*(f+1)+a];
	}
	Eui = r - Eui;
	return Eui;
}

float computeEui2(int i, int u, int r, int f, double* q, double* p, double* megaMatrix){
    int a;
    float Eui = 0;
    for(a = 1; a < (f+1); a++){
        Eui += q[i*(f+1)+a] * p[u*(f+1)+a];
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
