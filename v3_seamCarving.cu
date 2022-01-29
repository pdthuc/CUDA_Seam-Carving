#include <stdio.h>
#include <stdint.h>
#define FILTER_WIDTH 3
volatile __device__ int idx_h = 0;
__constant__ int dc_filterX[FILTER_WIDTH * FILTER_WIDTH];
__constant__ int dc_filterY[FILTER_WIDTH * FILTER_WIDTH];

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


__global__ void convertRgb2GrayKernel(uint8_t * inPixels, int width, int height, 
	uint8_t * outPixels)
{
	// TODO
	// Reminder: gray = 0.299*red + 0.587*green + 0.114*blue  
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	if(x < width && y < height)
	{
		int i = y * width + x;
		uint8_t red = inPixels[3 * i];
		uint8_t green = inPixels[3 * i + 1];
		uint8_t blue = inPixels[3 * i + 2];
		outPixels[i] = 0.299f*red + 0.587f*green + 0.114f*blue;
	}
}

__global__ void edgeDetectionKernel(uint8_t * inPixels, int width, int height, 
	int filterWidth, 
	int * outPixels)
{
	// TODO
	extern __shared__ uint8_t s_inPixels[];
	int temp = filterWidth / 2;
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;

	if(j < width && i < height)
	{
		// Calculate number of SMEM a single thread can handle
		int stride = ((blockDim.x+temp*2)*(blockDim.y+temp*2)-1)/(blockDim.x*blockDim.y)+1;
		int idx = threadIdx.x + threadIdx.y*blockDim.x;
		int index_x, index_y, s_x, s_y;

		// Initialize SMEM
		for(int k = 0; k < stride; k++)
		{
			s_x = (idx*stride + k)%(blockDim.x+temp*2);
			s_y = (idx*stride + k)/(blockDim.x+temp*2);
			if(s_x + s_y*(blockDim.x+temp*2) > (blockDim.x+temp*2)*(blockDim.y+temp*2)-1)
			{
				break;
			}
			index_x = j - j%blockDim.x + s_x - temp;
			index_y = i - i%blockDim.y + s_y - temp;
			if(index_x < 0) index_x = 0;
			else if(index_x > width-1) index_x = width-1;

			if(index_y < 0) index_y = 0;
			else if(index_y > height-1) index_y = height-1;

			s_inPixels[s_x + s_y*(blockDim.x+temp*2)] = inPixels[index_x + index_y*width];
		}
		__syncthreads();
		
		// Calculate edge detection values using filterX and filterY matrix
		float result[] = {0.0f,0.0f};
		for (int r = 0; r < filterWidth; r++)
		{
			for (int c = 0; c < filterWidth; c++)
			{
				int s_x = threadIdx.x + c;
				int s_y = threadIdx.y + r;

				uint8_t org_pixel = s_inPixels[s_x+s_y*(blockDim.x+temp*2)];
				result[0] += org_pixel * dc_filterX[c + r * filterWidth];
				result[1] += org_pixel * dc_filterY[c + r * filterWidth];
			}
		}
		outPixels[j+i*width] = abs(result[0]) + abs(result[1]);
	}
}

__global__ void calculateEnergyVerticalKernel(int * inPixels, int width, int height,
				int * nextPixels)
{
	// TODO
	// gridSize = (Width-1)/blockSize + 1
	int i = threadIdx.x + threadIdx.y*blockDim.x + gridDim.x*blockIdx.x;
	int min;

	if(i < width)
		// Calculate seam from the bottom to the top
		for(int r = height-2; r >= 0; r--)
		{
			// Compare the middle pixel with left and right pixel to find min
			min = i + (r+1) * width;
			
			if(i-1 >= 0 && inPixels[i-1 + (r+1) * width] < inPixels[min])
				min = i-1 + (r+1) * width;
				
			if(i+1 <= width-1 && inPixels[i+1 + (r+1) * width] < inPixels[min])
					min = i+1 + (r+1) * width;
	
			// Add pixels to create the seam and save the position in nextPixels array to track
			inPixels[i + r * width] += inPixels[min];
			nextPixels[i + r * width] = min;
			__syncthreads();
		}
}


__global__ void removeSeamVerticalKernel(uint8_t * inPixels, int width, int height,
										int * seam, uint8_t * outPixels)
{
	int j = threadIdx.x + blockDim.x * blockIdx.x;
	int i = threadIdx.y + blockDim.y * blockIdx.y;
	int step = 0;

	// Find the new position for each pixels after seam removal and puts pixels in original image to its new position in output image
	if(j < width && i < height)
	{
		while(j + i * width >= seam[i + step] - (i + step))
		{
			if(i+step >= height)
				break;
			step++;
		}
		if(j + i * width + step + i <= width*height - 1)
		{
			outPixels[3*(j + i * width)] = inPixels[3*(j + i * width + step + i)];
			outPixels[3*(j + i * width)+1] = inPixels[3*(j + i * width + step + i)+1];
			outPixels[3*(j + i * width)+2] = inPixels[3*(j + i * width + step + i)+2];		
		}	
	}
}		


void seamCarvingKernel(uint8_t * inPixels, int width, int height, uint8_t * outPixels_v,
			int * filterX, int * filterY, int filterWidth, dim3 blockSize=dim3(1, 1),
        	int times = 1)
{
	// TODO
	const int zero = 0;
	size_t smem = (blockSize.x+filterWidth/2*2)*(blockSize.y+filterWidth/2*2)*sizeof(uint8_t);
	size_t seamSize = height*sizeof(int);
	if(width > height) seamSize = width*sizeof(int);
	cudaStream_t stream[2];

	for(int i = 0 ; i < 2 ; i++)
		cudaStreamCreate(&stream[i]);
	
    int width_v = width;
	uint8_t * d_inGrayscale_v, *d_inPixels_v, *d_outPixels_v;
	int *energy_v = (int*)malloc(width*height*sizeof(int));
    int* min_seam_v = (int*)malloc(seamSize);
	int * next_pixels_v = (int*)malloc(width*(height-1)*sizeof(int));
	int *d_energy_v, *d_next_pixels_v, *d_seam_v;

	CHECK(cudaMalloc(&d_inPixels_v, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_outPixels_v, width*height*3*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_inGrayscale_v, width*height*sizeof(uint8_t)));
	CHECK(cudaMalloc(&d_energy_v, width*height*sizeof(int)));
	CHECK(cudaMalloc(&d_next_pixels_v, width*(height-1)*sizeof(int)));
	CHECK(cudaMalloc(&d_seam_v, height*sizeof(int)));

	dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);

	// Grid size for seam calculation
	dim3 gridSize_vertical_flatten((width-1)/(blockSize.x*blockSize.y) + 1);

	// TODO: copy data from "filter" (on host) to "dc_filter" (on CMEM of device)
	CHECK(cudaMemcpyToSymbol(dc_filterX, filterX, filterWidth*filterWidth*sizeof(int)));
	CHECK(cudaMemcpyToSymbol(dc_filterY, filterY, filterWidth*filterWidth*sizeof(int)));
	CHECK(cudaMemcpy(d_inPixels_v, inPixels, width*height*3*sizeof(uint8_t), cudaMemcpyHostToDevice));

	for(int count = 0; count < times; count++)
	{
		// Convert RGB image to grayscale image
		convertRgb2GrayKernel<<<gridSize, blockSize>>>(d_inPixels_v, width_v, height, d_inGrayscale_v);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Calculate energy of each pixel using Edge Detection
		edgeDetectionKernel<<<gridSize, blockSize, smem>>>(d_inGrayscale_v, width_v, height, filterWidth, d_energy_v);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// // Test energy v
		// CHECK(cudaMemcpy(energy_v, d_energy_v, width_v*height*sizeof(int), cudaMemcpyDeviceToHost));
		// writeTxt(energy_v, width, height, "v3_device_pixel_energy.txt");


		//Calculate seam from pixels energy vertically
		calculateEnergyVerticalKernel<<<gridSize_vertical_flatten, blockSize, 0, stream[0]>>>(d_energy_v, width_v, height, d_next_pixels_v);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy from kernel memory to host memory in order to find min seam using host
		CHECK(cudaMemcpyAsync(energy_v, d_energy_v, width_v*height*sizeof(int), cudaMemcpyDeviceToHost, stream[0]));
		CHECK(cudaMemcpyAsync(next_pixels_v, d_next_pixels_v, width_v*(height-1)*sizeof(int), cudaMemcpyDeviceToHost, stream[0]));
		// writeTxt(energy_v, width, height, "v3_device_seam_energy.txt");

		//Find min seam
		int min_v = 0;
		for(int c = 1; c < width_v; c++)
			if(energy_v[c] < energy_v[min_v]) 
				min_v = c;

        min_seam_v[0] = min_v;
        for(int r = 0; r < height - 1; r++)
        {
            min_seam_v[r+1] = next_pixels_v[min_v];
            min_v = min_seam_v[r+1];
        }
            		
		//Remove min seam from the image
        CHECK(cudaMemcpy(d_seam_v, min_seam_v, height*sizeof(int), cudaMemcpyHostToDevice));
        removeSeamVerticalKernel<<<gridSize, blockSize>>>(d_inPixels_v, width_v, height, d_seam_v, d_outPixels_v);
        width_v--;		
		cudaDeviceSynchronize(); 
		CHECK(cudaGetLastError());

		CHECK(cudaMemcpy(d_inPixels_v, d_outPixels_v, width_v*height*3*sizeof(uint8_t), cudaMemcpyDeviceToDevice));
	}
	CHECK(cudaMemcpy(outPixels_v, d_outPixels_v, width_v*height*3*sizeof(uint8_t), cudaMemcpyDeviceToHost));
	writePnm(outPixels_v, 3, width_v, height, "v3_device_out.pnm");
}
			
void seamCarvingHost(uint8_t * inPixels, int width, int height, uint8_t* outPixels, 
	int * filterX, int * filterY, int filterWidth, int times)
{
	int temp = filterWidth / 2;
	uint8_t * inGrayscale = (uint8_t*)malloc(width*height*sizeof(uint8_t));
	int * energy_v = (int*)malloc(width*height*sizeof(int));
	int * next_pixels_v = (int*)malloc(width*(height-1)*sizeof(int));

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
					{
						min = idx + (r+1)*width;
					}

				}
				energy_v[c + r*width] += energy_v[min];			
				next_pixels_v[c + r*width] = min;							
			}	
		}

		//Find min seam
		size_t seamSize = height*sizeof(int);
		if(width > height) seamSize = width*sizeof(int);
		int* min_seam = (int*)malloc(seamSize);

		min_v = 0;
		for(int c = 1; c < width; c++)
			if(energy_v[c] < energy_v[min_v]) min_v = c; 

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
	writePnm(outPixels, 3, width, height, "v3_host_out.pnm");
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
	else // Use device
	{    
		seamCarvingKernel(inPixels, width, height, outPixels, filterX, filterY, filterWidth, blockSize, times);
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

	if(strcmp(type,"both") == 0)
	{
		//Seam carving by Device
		printf("Kernel time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times, true, blockSize);

		// Seam carving by Host
		printf("Host time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times);
	}
	else if(strcmp(type,"kernel") == 0)
	{
		//Seam carving by Device
		printf("Kernel time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times, true, blockSize);		
	}
	else if(strcmp(type,"host") == 0)
	{
		// // Seam carving by Host
		printf("Host time: \n");
		seamCarving(inPixels, width, height, filterX, filterY, filterWidth, outPixels, times);
	}

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(outPixels);
}
