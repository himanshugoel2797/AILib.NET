using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib;
using AILib.ANN;
using AILib.ANN.ActivationFunctions;
using AILib.ANN.LossFunctions;
using AILib.ANN.Optimizers;
#if CPUMATH
using AILib.Math;
#else
using AILib.Math.GPU;
#endif

namespace NeuralNetworkTest
{
    class Program
    {
        static void Main(string[] args)
        {
            CoreLibrary.Initialize();

            /*
            Matrix a = new Matrix(2, 2);
            Matrix b = new Matrix(2, 2);
            Matrix c = new Matrix(2, 2);

            Vector v0 = new Vector(2);
            Vector v1 = new Vector(2);
            Vector v2 = new Vector(2);

            a[0, 0] = 1;
            a[0, 1] = 2;
            a[1, 0] = 4;
            a[1, 1] = 8;

            b[0, 0] = 4;
            b[0, 1] = 2;
            b[1, 0] = 5;
            b[1, 1] = 8;

            v0[0] = 1;
            v0[1] = 2;

            v1[0] = 4;
            v1[1] = 8;

            Matrix.Multiply(a, b, ref c);
            Console.WriteLine("Multiply:");
            Console.WriteLine("a=\n" + a);
            Console.WriteLine("b=\n" + b);
            Console.WriteLine("c=\n" + c);

            Matrix.Madd(a, v0, v1, ref v2);
            Console.WriteLine("Madd:");
            Console.WriteLine("a=\n" + a);
            Console.WriteLine("v0=\n" + v0);
            Console.WriteLine("v1=\n" + v1);
            Console.WriteLine("v2=\n" + v2);

            Matrix.MSub(a, 10, ref c);
            Console.WriteLine("Msub:");
            Console.WriteLine("a=\n" + a);
            Console.WriteLine("c=\n" + c);


            Vector.MSub(v0, 10, ref v1);
            Console.WriteLine("Msub:");
            Console.WriteLine("v0=\n" + v0);
            Console.WriteLine("v1=\n" + v1);

            Matrix.MultiplyToMatrix(v0, v1, ref c);
            Console.WriteLine("VECVECMUL:");
            Console.WriteLine("v0=\n" + v0);
            Console.WriteLine("v1=\n" + v1);
            Console.WriteLine("c=\n" + c);

            Matrix.TransposedMultiply(a, v0, v1, ref v2);
            Console.WriteLine("TransposedMultiply:");
            Console.WriteLine("a=\n" + a);
            Console.WriteLine("v0=\n" + v0);
            Console.WriteLine("v1=\n" + v1);
            Console.WriteLine("v2=\n" + v2);


            Console.ReadLine();
            */

            AnimeGANTest.GAN();
            Console.ReadLine();
            return;
            /*
            NeuralNetwork net = new NeuralNetwork(new int[] { 2, 2, 1 }, new IActivationFunction[] { null, new Sigmoid(), new Sigmoid() }, new Quadratic(), new SGD());

            float[][] inputs = new float[][] { new float[] { 0, 1 }, new float[] { 1, 0 }, new float[] { 0, 0 }, new float[] { 1, 1 } };
            float[][] outputs = new float[][] { new float[] { 1 }, new float[] { 1 }, new float[] { 0 }, new float[] { 0 } };

            for(int i = 0; i < 500000; i++)
            {
                var (a , z) = net.Train(inputs[i % 4]);
                var (nabla_w, nabla_b) = GradientSolver.Solve(net, a, z, new Vector(outputs[i % 4]));
                (net.Weights, net.Biases) = net.Optimizer.Optimize(net.Weights, net.Biases, nabla_w, nabla_b, 0.05f);
            }

            for(int i = 0; i < 4; i++)
            {
                float res = net.Activate(inputs[i])[0];
                Console.WriteLine($"Result {i} : {res}");
            }
            Console.ReadLine();*/
        }
    }
}
