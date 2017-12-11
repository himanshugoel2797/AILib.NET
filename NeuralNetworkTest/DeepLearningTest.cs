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

namespace NeuralNetworkTest
{
    class DeepLearningTest
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
            //NeuralNetwork generator = new NeuralNetwork(new int[] { 100, 128, 28 * 28 }, new IActivationFunction[] { null, new Tanh(), new Sigmoid() }, genFunc);
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

            for (int q = 0; q < 30; q++)
            {
                float[] Gz = null;
                for (int i = 0; i < 50000; i++)
                {
                    //if (expected_vec[i][0] != 1)
                    //    continue;

                    //for (int j = 0; j < generator_input.Length; j++)
                    //    generator_input[j] = (float)rng.NextDouble() * 1;

                    //Gz = generator.Activate(generator_input);
                    //var Dg = discriminator.Activate(Gz);
                    var Dz = discriminator.Activate(real_in[i]);

                    //disc.DiscriminatorValue = Dg[0] - 1;
                    //discriminator.Train(Gz, new float[] { 0 }, 3f / 1000);

                    //disc.DiscriminatorValue = Dz[0] + 0.001f;
                    discriminator.Train(real_in[i]); //, new float[] { expected_vec[i][0] }, 3f / 1000

                    //generator.Train(generator_input, real_in[i], 3f / 5000);


                    //Console.WriteLine($"Discriminator Output: {Dx.}");


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
