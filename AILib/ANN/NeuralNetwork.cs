using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
#if CPUMATH
using AILib.Math;
#else
using AILib.Math.GPU;
#endif
using Superbest_random;

namespace AILib.ANN
{
    public class NeuralNetwork
    {
        private IActivationFunction[] function;
        private ILossFunction lossFunc;
        private int[] layers;

        public Matrix[] Weights { get; set; }
        public Vector[] Biases { get; set; }

        public int LayerCount { get { return layers.Length; } }
        public int[] Layers { get { return layers; } }
        public IActivationFunction[] ActivationFunctions { get { return function; } }
        public ILossFunction LossFunction { get { return lossFunc; } }
        public IOptimizer Optimizer { get; set; }
        public GradientSolver Solver { get; set; }

        public NeuralNetwork(int[] layers, IActivationFunction[] function, ILossFunction loss, IOptimizer optimizer)
        {
            this.layers = layers;
            this.function = function;
            this.lossFunc = loss;
            this.Optimizer = optimizer;

            this.Weights = new Matrix[layers.Length];
            this.Biases = new Vector[layers.Length];

            int w = layers[0];

            Random rng = new Random(0);

            for (int i = 1; i < Weights.Length; i++)
            {
                int h = layers[i];

                Weights[i] = new Matrix(w, h);
                Biases[i] = new Vector(h);

                for (int j = 0; j < h; j++)
                {
                    Biases[i][j] = (float)rng.NextGaussian();// 1e-10f;

                    for (int k = 0; k < w; k++)
                        Weights[i][k, j] = (float)rng.NextGaussian(0, System.Math.Pow(w, -0.5f));// * 1e-10f;
                }

                w = h;
            }

            this.Solver = new GradientSolver(this);
        }

        /// <summary>
        /// Connect networks with 0th entry as input
        /// </summary>
        /// <param name="nets"></param>
        public NeuralNetwork(params NeuralNetwork[] nets)
        {
            int layerCount = 1;
            for (int i = 0; i < nets.Length; i++)
                layerCount += (nets[i].LayerCount - 1);

            this.layers = new int[layerCount];
            this.function = new IActivationFunction[layerCount];
            this.lossFunc = nets[0].LossFunction;
            this.Optimizer = nets[0].Optimizer;

            this.Weights = new Matrix[layerCount];
            this.Biases = new Vector[layerCount];

            int idx = 1;
            for (int i = 0; i < nets.Length; i++)
            {
                for (int j = 1; j < nets[i].LayerCount; j++)
                {
                    Weights[idx] = nets[i].Weights[j];
                    Biases[idx] = nets[i].Biases[j];

                    layers[idx] = nets[i].Layers[j];
                    function[idx] = nets[i].ActivationFunctions[j];

                    idx++;
                }
            }

            this.Solver = new GradientSolver(this);
        }

        public float[] Activate(float[] inputs)
        {
            Vector res = new Vector(inputs);
            for (int i = 1; i < layers.Length; i++)
            {
                Matrix.Madd(Weights[i], res, Biases[i], ref res);
                res = ActivationFunctions[i].Activation(res);
            }

            return res;
        }

        Vector[] a;
        Vector[] z;

        public (Vector[], Vector[]) Train(float[] input)
        {
            if (a == null) a = new Vector[layers.Length];
            if (z == null) z = new Vector[layers.Length];

            if (a[0] == null) a[0] = new Vector(input);

            for (int i = 1; i < layers.Length; i++)
            {
                if (a[i] == null) a[i] = new Vector(layers[i]);
                if (z[i - 1] == null) z[i - 1] = new Vector(layers[i]);

                Matrix.Madd(Weights[i], a[i - 1], Biases[i], ref z[i - 1]);
                a[i] = ActivationFunctions[i].Activation(z[i - 1]);
            }

            return (a, z);
        }
    }
}
