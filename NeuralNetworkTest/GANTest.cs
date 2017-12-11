using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.ANN;
using AILib.ANN.ActivationFunctions;
using AILib.ANN.LossFunctions;
using AILib.ANN.Optimizers;
using AILib.Math;
using Superbest_random;

namespace NeuralNetworkTest
{
    class GANTest
    {
        static float[] generator_input = new float[100];
        static float[][] real_in;
        static float[][] expected_vec;

        static Random rng0 = new Random();

        static FileStream f = null;
        static BinaryReader br = null;

        static FileStream l = null;
        static BinaryReader lbl = null;

        private static void readMNIST(int idx)
        {
            if (f == null)
            {
                f = File.OpenRead("train-images.idx3-ubyte");
                br = new BinaryReader(f);

                l = File.OpenRead("train-labels.idx1-ubyte");
                lbl = new BinaryReader(l);

                lbl.ReadInt32(); //magic
                lbl.ReadInt32(); //item count

                br.ReadInt32();    //magic
                br.ReadInt32();    //item count
                br.ReadInt32();    //rows
                br.ReadInt32();    //columns
            }

            float hi = 1;
            float lo = 0;

            if (rng0.NextDouble() >= 0.5f)
            {
                //hi = 0;
                //lo = 1;
            }

            for (int i = 0; i < 28 * 28; i++)
            {
                if (br.ReadByte() >= 128)
                    real_in[idx][i] = hi;
                else
                    real_in[idx][i] = lo;
                //real_in[idx][i] = 1.0f - br.ReadByte() / 255.0f;

            }

            int label = lbl.ReadByte();
            for (int i = 0; i < 11; i++) expected_vec[idx][i] = 0;
            expected_vec[idx][label] = 1;
            expected_vec[idx][10] = 1;
        }

        public static void imsave(float[] Gz)
        {
            Bitmap bmp = new Bitmap(28, 28);
            for (int x = 0; x < 28; x++)
                for (int y = 0; y < 28; y++)
                {
                    float v = Gz[28 * y + x];

                    if (v < 0)
                        v = 0;

                    if (v > 1)
                        v = 1;

                    if (float.IsNaN(v))
                        v = 0;

                    bmp.SetPixel(x, y, Color.FromArgb((int)(v * 255), (int)(v * 255), (int)(v * 255)));
                }
            bmp.Save("generated.png");
            bmp.Dispose();
        }

        public static void GAN()
        {
            //Generator genFunc = new Generator();
            //Discriminator disc = new Discriminator();
            NeuralNetwork generator = new NeuralNetwork(new int[] { 100, 128, 28 * 28 }, new IActivationFunction[] { null, new Tanh(), new Sigmoid() }, new CrossEntropy(), new SGD());
            NeuralNetwork discriminator = new NeuralNetwork(new int[] { 28 * 28, 128, 1 }, new IActivationFunction[] { null, new Tanh(), new Sigmoid() }, new CrossEntropy(), new SGD());

            Random rng = new Random(0);

            real_in = new float[50000][];
            expected_vec = new float[50000][];
            for (int i = 0; i < real_in.Length; i++)
            {
                real_in[i] = new float[28 * 28];
                expected_vec[i] = new float[11];
                readMNIST(i);
            }

            float[] gen_out = new float[11];
            gen_out[10] = 0;

            float max_gen_res = 0;

            for (int q = 0; q < 30; q++)
            {
                float[] Gz = null;
                for (int i = 0; i < 50000; i++)
                {
                    if (expected_vec[i][6] != 1)
                        continue;

                    for (int j = 0; j < generator_input.Length; j++)
                        generator_input[j] = (float)rng.NextGaussian(0, System.Math.Pow(28 * 28, 0.5f)) * 1;

                    Gz = generator.Activate(generator_input);

                    //Train discriminator
                    var (Dz_a, Dz_z) = discriminator.Train(real_in[i]);
                    var (Dz_nabla_w, Dz_nabla_b) = GradientSolver.Solve(discriminator, Dz_a, Dz_z, new Vector( new float[] { 1 }));
                    (discriminator.Weights, discriminator.Biases) = discriminator.Optimizer.Optimize(discriminator.Weights, discriminator.Biases, Dz_nabla_w, Dz_nabla_b, 3f / 1000);

                    var (DGz_a, DGz_z) = discriminator.Train(Gz);
                    var (DGz_nabla_w, DGz_nabla_b) = GradientSolver.Solve(discriminator, DGz_a, DGz_z, new Vector(new float[] { 0 }));
                    (discriminator.Weights, discriminator.Biases) = discriminator.Optimizer.Optimize(discriminator.Weights, discriminator.Biases, DGz_nabla_w, DGz_nabla_b, 3f / 1000);

                    //Train Generator
                    //Now compute the error from the output of D, but don't change anything until G
                    NeuralNetwork chained = new NeuralNetwork(generator, discriminator);
                    var (Gz_a_ex, Gz_z_ex) = chained.Train(generator_input);
                    var (Gz_nabla_w_ex, Gz_nabla_b_ex) = GradientSolver.Solve(chained, Gz_a_ex, Gz_z_ex, new Vector(new float[] { 1 }));

                    Matrix[] Gz_nabla_w = new ArraySegment<Matrix>(Gz_nabla_w_ex, 0, generator.LayerCount).Array;
                    Vector[] Gz_nabla_b = new ArraySegment<Vector>(Gz_nabla_b_ex, 0, generator.LayerCount).Array;
                    (generator.Weights, generator.Biases) = generator.Optimizer.Optimize(generator.Weights, generator.Biases, Gz_nabla_w, Gz_nabla_b, 3f / 1000);

                    if(max_gen_res < DGz_a[2][0])
                    {
                        imsave(Gz);
                        max_gen_res = DGz_a[2][0];
                    }


                    Console.WriteLine($"[{i}]Discriminator Real: {Dz_a[2][0]} Fake: {DGz_a[2][0]}");
                }
                //imsave(Gz);
                Console.WriteLine($"Epoch {q} Finished.");
            }

            for (int i = 0; i < 50; i++)
            {
                var Dx = discriminator.Activate(real_in[i]);

                float max = Dx.Max();
                for (int k = 0; k < Dx.Length; k++)
                {
                    if (expected_vec[i][k] == 1)
                        Console.WriteLine($"Expected:{k}");

                    if (Dx[k] == max)
                        Console.WriteLine($"Discriminator: {k}");
                }

                Console.WriteLine();
            }


        }
    }
}
