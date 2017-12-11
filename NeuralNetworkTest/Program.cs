using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using AILib.ANN;
using AILib.ANN.ActivationFunctions;
using AILib.ANN.LossFunctions;
using AILib.ANN.Optimizers;
using AILib.Math;

namespace NeuralNetworkTest
{
    class Program
    {
        static void Main(string[] args)
        {
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
            Console.ReadLine();
        }
    }
}
