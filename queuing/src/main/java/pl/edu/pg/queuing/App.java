package pl.edu.pg.queuing;

import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static java.lang.System.nanoTime;
import static java.lang.System.out;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Random;

import mpi.MPI;

import com.jogamp.opencl.CLBuffer;
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
		
		// set up MPI
		MPI.Init(args);
		int myRank = MPI.COMM_WORLD.Rank();
		int mpiSize = MPI.COMM_WORLD.Size();

        // set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        System.out.println("created "+context);
        
		if (myRank != 0) { // slave
			
			float[] buff = new float[5];
			float[] sendBuff = new float[8];
			boolean stop = false;
			
			while (!stop) {
				
				MPI.COMM_WORLD.Recv(buff, 0, 5, MPI.FLOAT, 0, MPI.ANY_TAG);
				
				out.print("Data read by slave " + myRank + ":");
				stop = true;
				for (float i : buff) {
					if (i != 0) {
						stop = false;
					}
					out.print(i + ",");
				}
				out.println();
				if (!stop) {
			        
			        // always make sure to release the context under all circumstances
			        // not needed for this particular sample but recommented
			        try{
			            
			            // select fastest device
                                    int rand = new Random().nextInt(context.getDevices().length);
//			            CLDevice device = context.getMaxFlopsDevice();
                                    CLDevice device = context.getDevices()[rand];
			            System.out.println("using "+device);
			
			            // create command queue on device.
			            CLCommandQueue queue = device.createCommandQueue();
			
			            int elementCount = 100; // Length of arrays to process
			//            int localWorkSize = 1;
			            int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256); // Local work size dimensions
			            int globalWorkSize = roundUp(localWorkSize, elementCount); // rounded up to the nearest multiple of the localWorkSize
			            
			            float intervalMean = buff[0];
			            float intervalDev = buff[1];
			            float requirementMean = buff[2];
			            float requirementDev = buff[3];
			            int queueSize = Float.valueOf(buff[4]).intValue();
			
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
			            
			            float rejectedResult = 0f;
			            float meanSystemDelayResult = 0f;
			            float processingTimeResult = 0f;
			            
			            for (int i = 0; i < elementCount; i++) {
				            rejectedResult += rejected.getBuffer().get(i);
				            meanSystemDelayResult += meanSystemDelay.getBuffer().get(i);
				            processingTimeResult += processingTime.getBuffer().get(i);
			            }
			            rejectedResult /= elementCount;
			            meanSystemDelayResult /= elementCount;
			            processingTimeResult /= elementCount;
			            
			            out.println("Params: intervalMean " + intervalMean + ", intervalDev" + intervalDev +
			            		", requirementMean " + requirementMean + ", requirementDev " + requirementDev +
			            		", queueSize " + queueSize + ", rejectedResult " + rejectedResult + 
			            		", meanSystemDelayResult " + meanSystemDelayResult + 
			            		", processingTimeResult " + processingTimeResult); 
			
			            out.println("computation took: "+(time/1000000)+"ms");
			            
			            sendBuff = new float[] {intervalMean, intervalDev, requirementMean, requirementDev,
			            		Integer.valueOf(queueSize).floatValue(), rejectedResult, meanSystemDelayResult,
			            		processingTimeResult};
			            MPI.COMM_WORLD.Send(sendBuff, 0, 8, MPI.FLOAT, 0, 0);
			            
			        }finally{
			            
			        }
				}
			}
	        try {
	        	context.release();
	        } catch (ConcurrentModificationException e) {
	        	
	        }
	        MPI.Finalize();
		} else { // master
			double startTime = MPI.Wtime();
			ArrayList<Params> params = new ArrayList<>();
			BufferedReader br = null;
			BufferedWriter bw = null;
			String line;
			float[] recvBuff = new float[8];
			float[] buff = new float[5];
			
			ArrayList<Integer> workingSlaves = new ArrayList<>();
			for (int i = 1 ; i < mpiSize; i++) {
				workingSlaves.add(i);
			}
			try {
				 
				br = new BufferedReader(new FileReader("/macierz/home/131550km/Code/queuing-opencl/queuing/input.txt"));
//				br = new BufferedReader(new FileReader("./input.txt"));
				bw = new BufferedWriter(new FileWriter("/macierz/home/131550km/Code/queuing-opencl/queuing/result.csv"));
//				bw = new BufferedWriter(new FileWriter("./result.csv"));
				while ((line = br.readLine()) != null) {
					String[] param = line.split(",");
					params.add(new Params(
						Float.valueOf(param[0]), Float.valueOf(param[1]), Float.valueOf(param[2]), 
						Float.valueOf(param[3]), Float.valueOf(param[4])
					));
				}
				
				out.println("Params size: " + params.size());
				
				for (int i = 1; i < mpiSize; i++) {
					if (!params.isEmpty()) {
						Params param = params.get(0);
						buff = new float[] {param.intervalMean, param.intervalDev, param.requirementMean, param.requirementDev, param.queueSize};
						params.remove(0);
					} else {
						buff = new float[] {0f, 0f, 0f, 0f, 0f};
						workingSlaves.remove(Integer.valueOf(i));
					}
					MPI.COMM_WORLD.Send(buff, 0, 5, MPI.FLOAT, i, 0);
				}

				List<Integer> slavesToRemove = new ArrayList<>();
				while (workingSlaves.size() > 0) {
					workingSlaves.removeAll(slavesToRemove);
					slavesToRemove.clear();
					for (Integer i : workingSlaves) {
						MPI.COMM_WORLD.Recv(recvBuff, 0, 8, MPI.FLOAT, i, MPI.ANY_TAG);
						out.print("Received data from " + i + ":");
						String writeLine = "";
						for (float j : recvBuff) {
							writeLine += j + ",";
						}
						out.println(writeLine);
						bw.write(writeLine);
						bw.newLine();
						if (!params.isEmpty()) {
							Params param = params.get(0);
							buff = new float[] {param.intervalMean, param.intervalDev, param.requirementMean, param.requirementDev, param.queueSize};
							params.remove(0);
						} else {
							buff = new float[] {0f, 0f, 0f, 0f, 0f};
							slavesToRemove.add(i);
//							workingSlaves.remove(Integer.valueOf(i));
						}
						MPI.COMM_WORLD.Send(buff, 0, 5, MPI.FLOAT, i, 0);
					}
				}
		 
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
				if (bw != null) {
					try {
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
			double endTime = MPI.Wtime();
			double computationTime = endTime - startTime;
			out.println("I am the master, my rank is " + myRank);
			out.println("Whole computation took " + computationTime + " seconds");
			MPI.Finalize();
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
