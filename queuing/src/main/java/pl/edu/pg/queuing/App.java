package pl.edu.pg.queuing;

import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.System.*;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Random;

import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

/**
 * Hello world!
 * based on https://github.com/JogAmp/jocl-demos/blob/master/src/com/jogamp/opencl/demos/hellojocl/HelloJOCL.java
 */
public class App 
{
	public static void main(String[] args) throws IOException {

        // set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        System.out.println("created "+context);
        
        // always make sure to release the context under all circumstances
        // not needed for this particular sample but recommented
        try{
            
            // select fastest device
            CLDevice device = context.getMaxFlopsDevice();
            System.out.println("using "+device);

            // create command queue on device.
            CLCommandQueue queue = device.createCommandQueue();

            int elementCount = 1; // Length of arrays to process
            int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256); // Local work size dimensions
            int globalWorkSize = roundUp(localWorkSize, elementCount); // rounded up to the nearest multiple of the localWorkSize

            // load sources, create and build program
            CLProgram program = context.createProgram(App.class.getResourceAsStream("kernel.cl")).build();
            
            float rejected = 0;

            // get a reference to the kernel function with the name 'VectorAdd'
            // and map the buffers to its input parameters.
            CLKernel kernel = program.createCLKernel("RunSimulation");
            kernel.putArg(rejected);

            // asynchronous write of data to GPU device,
            // followed by blocking read to get the computed results back.
            long time = nanoTime();
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
            time = nanoTime() - time;

            // print first few elements of the resulting buffer to the console.
            
            out.println("Rejected elements: " + rejected);

            out.println("computation took: "+(time/1000000)+"ms");
            
        }finally{
            // cleanup all resources associated with this context.
            context.release();
        }

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
}
