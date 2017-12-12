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
using AILib.Math.GPU;

namespace NeuralNetworkTest
{
    class Program
    {
        static void Main(string[] args)
        {
            CoreLibrary.Initialize();

            Matrix a = new Matrix(2, 2);
            Matrix b = new Matrix(2, 2);

            a[0, 0] = 1;
            a[0, 1] = 2;
            a[1, 0] = 4;
            a[1, 1] = 8;

            b[0, 0] = 4;
            b[0, 1] = 2;
            b[1, 0] = 5;
            b[1, 1] = 8;

            var c = a * b;
            Console.WriteLine(c[0, 0]);

            Console.ReadLine();

            /*
            AnimeGANTest.GAN();
            Console.ReadLine();
            return;
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
