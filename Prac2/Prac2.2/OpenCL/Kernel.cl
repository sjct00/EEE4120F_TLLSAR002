
//TODO: set your arguments for the kernel. Note that you have to indicate if the argument is global or local. Global arguments are accessable by both host and this target device. While local can only be accessed by the device running this kernel. eg __global int* inputMatrixA, __local int* groupMemory

__kernel void matrixMultiplication(__global int* matrixA, __global int* matrixB, __global int*Size, __global int* output){
	//work item and work groups numbers
	int workItemNum = get_global_id(0); //Work item ID
	int workGroupNum = get_group_id(0); //Work group ID
	int localGroupID = get_local_id(0); //Work items ID within each work group

	//printf("wg:%i wi%i\n",workGroupNum,localGroupID);
	
	//memory buffers
	int size = *Size;

	//determine index to use for 1D matrix
	int indexA = workGroupNum/size + localGroupID;
	int indexB = localGroupID*size + workGroupNum%size;

	//printf("wg:%i wi:%i\n",indexA,indexB);

	int A = matrixA[indexA];
	int B = matrixB[indexB];
	//printf("A:%i B:%i",A,B);
	
	
	local int result[100];
	
	//calculation
	int res = A*B;
	//printf("%i\n",res);
	
	result[localGroupID] = res;	
	

	//barrier that stops all work items here until all work items in the work group have executed this function
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int groupValue = 0;
	if (localGroupID == 0){
		for (int i = 0;i<size;i++)
		{
			groupValue += result[i];
		}
		//printf("%i\n",groupValue);
		output[workGroupNum] = groupValue;
	}
	
	
}




