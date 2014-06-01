#define RANDOM_SEED 24352445523
#define TASK_NUMBER 100000
#define MAX_QUEUE_SIZE 1000 // TODO temporary, fix it

//random algorithm from Java
uint random(ulong * seed)
{
	*seed = (*seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
	return *seed >> 16;
}
//Moro's Inverse Cumulative Normal Distribution function approximation from NVIDIA example
float MoroInvCNDgpu(unsigned int x)
{
	const float a1 = 2.50662823884f;
	const float a2 = -18.61500062529f;
	const float a3 = 41.39119773534f;
	const float a4 = -25.44106049637f;
	const float b1 = -8.4735109309f;
	const float b2 = 23.08336743743f;
	const float b3 = -21.06224101826f;
	const float b4 = 3.13082909833f;
	const float c1 = 0.337475482272615f;
	const float c2 = 0.976169019091719f;
	const float c3 = 0.160797971491821f;
	const float c4 = 2.76438810333863E-02f;
	const float c5 = 3.8405729373609E-03f;
	const float c6 = 3.951896511919E-04f;
	const float c7 = 3.21767881768E-05f;
	const float c8 = 2.888167364E-07f;
	const float c9 = 3.960315187E-07f;

	float z;

	bool negate = false;

	// Ensure the conversion to floating point will give a value in the
	// range (0,0.5] by restricting the input to the bottom half of the
	// input domain. We will later reflect the result if the input was
	// originally in the top half of the input domain

	if (x >= 0x80000000UL)
	{
		x = 0xffffffffUL - x;
		negate = true;
	}

	// x is now in the range [0,0x80000000) (i.e. [0,0x7fffffff])
	// Convert to floating point in (0,0.5]
	const float x1 = 1.0f / (float)0xffffffffUL;
	const float x2 = x1 / 2.0f;
	float p1 = x * x1 + x2;

	// Convert to floating point in (-0.5,0]
	float p2 = p1 - 0.5f;

	// The input to the Moro inversion is p2 which is in the range
	// (-0.5,0]. This means that our output will be the negative side
	// of the bell curve (which we will reflect if "negate" is true).

	// Main body of the bell curve for |p| < 0.42
	if (p2 > -0.42f)
	{
		z = p2 * p2;
		z = p2 * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
	}
	// Special case (Chebychev) for tail
	else
	{
		z = log(-log(p1));
		z = - (c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9))))))));
	}

	// If the original input (x) was in the top half of the range, reflect
	// to get the positive side of the bell curve
	return negate ? -z : z;
}
//random number from normal distribution
float normDistribution(ulong *seed, float mean, float dev)
{
	float out = MoroInvCNDgpu(random(seed))*dev + mean;

	return max(out, 0.0001f);
}

//queue data type and manipulation functions
typedef struct queue
{
	float data[MAX_QUEUE_SIZE+1];
	int p,k; //pierwszy, zaostatni
	int max_size;
} queue;
void init(queue * q, int max_size)
{
	q->p = q->k = 0;

	q->max_size = max_size;
}
float front(const queue * q)
{
	return q->data[q->p];
}
void pop(queue *q)
{
	q->p++;
	if(q->p > q->max_size)
		q->p = 0;
}
void push(queue *q, float l)
{
	q->data[q->k++] = l;
	if(q->k > q->max_size)
		q->k = 0;
}
bool empty(const queue *q)
{
	return q->p == q->k;
}
bool full(const queue *q)
{
	return q->k == q->p-1 || (q->k == q->max_size && q->p == 0);
}

kernel void RunSimulation(global float *L, int numElements, float interval_mean, float interval_dev, float requirement_mean, float requirement_dev, int queue_size)
{
	// get index into global data array
	int iGID = get_global_id(0);

	// bound check (equivalent to the limit on a 'for' loop for standard/serial C code
	if (iGID >= numElements)
		return;

	ulong seed = RANDOM_SEED + iGID;

	int i;
	float time = 0; //czas, jaki juz przetwarzamy pierwszy element
	int accepted = 0, rejected = 0;

	queue q;
	init(&q, queue_size);

	for(i=0; i<TASK_NUMBER; ++i)
	{
		//wygeneruj kolejne zadanie
		float interval = normDistribution(&seed, interval_mean, interval_dev);
		float req = normDistribution(&seed, requirement_mean, requirement_dev);

		//przetworz zadania w kolejce
		while(!empty(&q) && front(&q) - time < interval)
		{
			time = 0;
			interval -= (front(&q) - time);
			pop(&q);
		}

		if(!empty(&q))
			time += interval;

		//dodaj nowy element
		if(full(&q))
			rejected++;
		else
		{
			push(&q, req);
			accepted++;
		}
	}

	L[iGID] = ((float)rejected) / (accepted + rejected);
}


