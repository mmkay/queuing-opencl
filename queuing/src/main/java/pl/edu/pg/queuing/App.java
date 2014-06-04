package pl.edu.pg.queuing;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.System.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import mpi.*;

/**
 * Hello world!
 * based on https://github.com/JogAmp/jocl-demos/blob/master/src/com/jogamp/opencl/demos/hellojocl/HelloJOCL.java
 */
public class App 
{
	public static void main(String[] args) throws IOException {
		
		// set up MPI
		MPI.Init(args);
		int myRank = MPI.COMM_WORLD.Rank();
		int mpiSize = MPI.COMM_WORLD.Size();

        // set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        System.out.println("created "+context);
        
		if (myRank != 0) { // slave
			
			float[] buff = new float[5];
			
			MPI.COMM_WORLD.Recv(buff, 0, 5, MPI.FLOAT, 0, MPI.ANY_TAG);
			
			out.print("Data read by slave " + myRank + ":");
			for (float i : buff) {
				out.print(i + ",");
			}
			out.println();
	        
	        // always make sure to release the context under all circumstances
	        // not needed for this particular sample but recommented
	        try{
	            
	            // select fastest device
	            CLDevice device = context.getMaxFlopsDevice();
	            System.out.println("using "+device);
	
	            // create command queue on device.
	            CLCommandQueue queue = device.createCommandQueue();
	
	            int elementCount = 10; // Length of arrays to process
	//            int localWorkSize = 1;
	            int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256); // Local work size dimensions
	            int globalWorkSize = roundUp(localWorkSize, elementCount); // rounded up to the nearest multiple of the localWorkSize
	            
	            float intervalMean = 10.0f;
	            float intervalDev = 3.0f;
	            float requirementMean = 10.0f;
	            float requirementDev = 5.0f;
	            int queueSize = 10;
	
	            // load sources, create and build program
	            CLProgram program = context.createProgram(App.class.getResourceAsStream("kernel.cl")).build();
	            
	            CLBuffer<FloatBuffer> rejected = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);
	            CLBuffer<FloatBuffer> meanSystemDelay = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);
	            CLBuffer<FloatBuffer> processingTime = context.createFloatBuffer(globalWorkSize, WRITE_ONLY);
	            
	            // get a reference to the kernel function with the name 'VectorAdd'
	            // and map the buffers to its input parameters.
	            CLKernel kernel = program.createCLKernel("RunSimulation");
	            kernel.putArgs(rejected).putArgs(meanSystemDelay).putArgs(processingTime)
	            	.putArg(elementCount).putArg(intervalMean).putArg(intervalDev)
	            	.putArg(requirementMean).putArg(requirementDev).putArg(queueSize);
	
	            // asynchronous write of data to GPU device,
	            // followed by blocking read to get the computed results back.
	            long time = nanoTime();
	            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
	                .putReadBuffer(rejected, true)
	                .putReadBuffer(meanSystemDelay, true)
	                .putReadBuffer(processingTime, true);
	            time = nanoTime() - time;
	
	            // print first few elements of the resulting buffer to the console.
	            
	            for (int i = 0; i < elementCount; i++) {
	            	out.println("Rank " + myRank + " Thread " + i);
		            out.println("Rank " + myRank + " Rejected elements: " + rejected.getBuffer().get(i));
		            out.println("Rank " + myRank + " Mean system delay: " + meanSystemDelay.getBuffer().get(i));
		            out.println("Rank " + myRank + " Processing time: " + processingTime.getBuffer().get(i));
	            }
	
	            out.println("computation took: "+(time/1000000)+"ms");
	            
	        }finally{
	            // cleanup all resources associated with this context.
	            context.release();
	        }
		} else { // master
			ArrayList<Params> params = new ArrayList<>();
			BufferedReader br = null;
			String line;
			try {
				 
				br = new BufferedReader(new FileReader("./input.txt"));
				while ((line = br.readLine()) != null) {
					String[] param = line.split(",");
					params.add(new Params(
						Float.valueOf(param[0]), Float.valueOf(param[1]), Float.valueOf(param[2]), 
						Float.valueOf(param[3]), Float.valueOf(param[4])
					));
				}
				
				out.println("Params size: " + params.size());
				
				for (int i = 1; i < mpiSize; i++) {
					float[] buff;
					if (!params.isEmpty()) {
						Params param = params.get(0);
						buff = new float[] {param.intervalMean, param.intervalDev, param.requirementMean, param.requirementDev, param.queueSize};
						params.remove(0);
					} else {
						buff = new float[] {0f, 0f, 0f, 0f, 0f};
					}
					MPI.COMM_WORLD.Send(buff, 0, 5, MPI.FLOAT, i, 0);
				}
				
				// TODO send first params
		 
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (br != null) {
					try {
						br.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
			out.println("I am the master, my rank is " + myRank);
		}
    	MPI.Finalize();

    }

    private static void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while(buffer.remaining() != 0)
            buffer.put(rnd.nextFloat()*100);
        buffer.rewind();
    }

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
    
    static class Params {
    	public Params(float intervalMean, float intervalDev, float requirementMean, float requirementDev, 
    		float queueSize) {
    		this.intervalMean = intervalMean;
    		this.intervalDev = intervalDev;
    		this.requirementMean = requirementMean;
    		this.requirementDev = requirementDev;
    		this.queueSize = queueSize;
    	}
    	
    	public float intervalMean;
        public float intervalDev;
        public float requirementMean;
        public float requirementDev;
        public float queueSize;
    }
}
