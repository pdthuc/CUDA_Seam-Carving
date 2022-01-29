#include <stdio.h>
#include <stdint.h>
#define FILTER_WIDTH 3
volatile __device__ int d_min = 0;

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


void readPnm(char * fileName, 
	int &numChannels, int &width, int &height, uint8_t * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	if (strcmp(type, "P2") == 0)
		numChannels = 1;
	else if (strcmp(type, "P3") == 0)
		numChannels = 3;
	else // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);

	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uint8_t *)malloc(width * height * numChannels);
	for (int i = 0; i < width * height * numChannels; i++)
		fscanf(f, "%hhu", &pixels[i]);

	fclose(f);
}

void writePnm(uint8_t * pixels, int numChannels, int width, int height, 
	char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	if (numChannels == 1)
		fprintf(f, "P2\n");
	else if (numChannels == 3)
		fprintf(f, "P3\n");
	else
	{
		fclose(f);
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	fprintf(f, "%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height * numChannels; i++)
		fprintf(f, "%hhu\n", pixels[i]);

	fclose(f);
}


void writeTxt(int * matrix, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			fprintf(f, "%i  ", matrix[j + i*width]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void seamCarvingHost(uint8_t * inPixels, int width, int height, uint8_t* outPixels, 
	int * filterX, int * filterY, int filterWidth, int times)
{
	int temp = filterWidth / 2;
	uint8_t * inGrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	int * energy_v = (int*)malloc(width*height*sizeof(int));
	int * importancy_h = (int*)malloc(width*height*sizeof(int));
	int * next_pixels_v = (int*)malloc(width*(height-1)*sizeof(int));
	int * next_pixels_h = (int*)malloc((width-1)*height*sizeof(int));

	for(int count = 0; count < times; count++)
	{
		//Convert RGB to Grayscale
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				int i = r * width + c;
				uint8_t red = inPixels[3 * i];
				uint8_t green = inPixels[3 * i + 1];
				uint8_t blue = inPixels[3 * i + 2];
				inGrayscale[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
			}
		}		
		//Calculate importancy of each pixel using Edge Detection
		for (int resultR = 0; resultR < height; resultR++)
		{
			for (int resultC = 0; resultC < width; resultC++)
			{
				float importancy_X = 0, importancy_Y = 0;

				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterValX = filterX[filterR*filterWidth + filterC];
						float filterValY = filterY[filterR*filterWidth + filterC];

						int inPixelsR = resultR - temp + filterR;
						int inPixelsC = resultC - temp + filterC;
						
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);

						uint8_t inPixel = inGrayscale[inPixelsR*width + inPixelsC];

						importancy_X += filterValX * inPixel;
						importancy_Y += filterValY * inPixel;
					}
				}
				energy_v[resultR*width + resultC] = abs(importancy_X) + abs(importancy_Y);
			}
		}
		// writeTxt(importancy, width, height, "importancy.txt");
		memcpy(importancy_h, energy_v, width*height*sizeof(int));

		int min, min_v;
		//Calculate seam from pixels importancy
		for(int r = height - 2; r >= 0; r--)
		{
			for(int c = 0; c < width; c++)
			{
				min = c + (r+1)*width;
				for (int idx = c-1; idx <= c+1; idx+=2)
				{
					if(idx < 0) idx = 0;
					if(idx > width-1) idx = width - 1;
					if(energy_v[idx + (r+1)*width] < energy_v[min])
						min = idx + (r+1)*width;
				}
				energy_v[c + r*width] += energy_v[min];			
				next_pixels_v[c + r*width] = min;							
			}	
		}

		// writeTxt(importancy, width, height, "updated_importancy.txt");

		//Find min seam
		size_t seamSize = height*sizeof(int);
		if(width > height) seamSize = width*sizeof(int);
		int* min_seam = (int*)malloc(seamSize);

		min_v = 0;
		for(int c = 1; c < width; c++)
		{
			if(energy_v[c] < energy_v[min_v]) min_v = c; 
		}

		min_seam[0] = min_v;
		for(int r = 0; r < height - 1; r++)
		{
			min_seam[r+1] = next_pixels_v[min_v];
			min_v = min_seam[r+1];
		}

		int idx = 0;
		//Remove min seam from the image vertically
		for(int i = 0; i < height; i++)
			{
				for(int j = 0; j < width; j++)
				{
					if(j + i * width < min_seam[0])
					{
						outPixels[3*(j + i * width)] = inPixels[3*(j + i * width)];
						outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width)+1];
						outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width)+2];
					}
					else 
					{
						if(j + i * width >= min_seam[idx+1] - idx - 1) idx++;

						if(j + i * width + idx + 1 > width*height - 1) break;
						
						outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + idx + 1)];
						outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + idx + 1)+1];
						outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + idx + 1)+2];
					}
				}
			}
		width--;
	
		inPixels = outPixels;
	}
	writePnm(outPixels, 3, width, height, "v1_host_out.pnm");
}

void seamCarving(uint8_t * inPixels, int width, int height, int * filterX, int* filterY, int filterWidth,
        uint8_t * outPixels, int times=1,
        bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer; 
	timer.Start();
	if (useDevice == false) // Use host
	{ 
		seamCarvingHost(inPixels, width, height, outPixels, filterX, filterY, filterWidth, times);
	}
	else 
	{    
        // Use device
	}
	timer.Stop();
	float time2 = timer.Elapsed();
	printf("%f ms\n", time2);
}


float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}

int main(int argc, char ** argv)
{
	if (argc == 4 && argc > 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();

	// Read input image file
	int numChannels, width, height, times;
	uint8_t * inPixels;
	dim3 blockSize(32,32);					

	readPnm(argv[1], numChannels, width, height, inPixels);
	char* type = argv[2];
	if (argc > 3)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}
	if (argc == 6)
		times = atoi(argv[5]);

	if (numChannels != 3)
	{
		return EXIT_FAILURE; // Input image must be RGB
	}
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Set up a simple filter with blurring effect 
	int filterWidth = FILTER_WIDTH;
	int filterX[] = {1, 0, -1,
					2, 0, -2,
					1, 0, -1};

	int filterY[] = {1, 2, 1, 
					0, 0, 0,
					-1, -2, -1};

	// Blur input image not using device
	uint8_t * correctOutPixels = (uint8_t *)malloc(width * height * numChannels * sizeof(uint8_t)); 
	uint8_t * outPixels = (uint8_t*)malloc(width * height * numChannels * sizeof(uint8_t));

	if(strcmp(type,"host") == 0)
	{
		// Seam carving by Host
		printf("Host time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, correctOutPixels, times);
	}

    // Free memories
	free(inPixels);
	free(correctOutPixels);
	free(outPixels);
}