using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
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
    class AnimeGANTest
    {
        static float[] generator_input = new float[100];
        static float[][] real_in, real_in_2;
        static float[][] expected_vec;

        static Random rng0 = new Random();
        static string category_name = "nishikino_maki";
        static string vs_category_name = "akemi_homura";
        static IEnumerable<string> files, vs_files;

        public static Bitmap MakeGrayscale3(Bitmap original)
        {
            //create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width / 2, original.Height / 2);

            //get a graphics object from the new image
            Graphics g = Graphics.FromImage(newBitmap);

            //create the grayscale ColorMatrix
            ColorMatrix colorMatrix = new ColorMatrix(
               new float[][]
               {
         new float[] {.3f, .3f, .3f, 0, 0},
         new float[] {.59f, .59f, .59f, 0, 0},
         new float[] {.11f, .11f, .11f, 0, 0},
         new float[] {0, 0, 0, 1, 0},
         new float[] {0, 0, 0, 0, 1}
               });

            //create some image attributes
            ImageAttributes attributes = new ImageAttributes();

            //set the color matrix attribute
            attributes.SetColorMatrix(colorMatrix);

            //draw the original image on the new image
            //using the grayscale color matrix
            g.DrawImage(original, new Rectangle(0, 0, original.Width / 2, original.Height / 2),
               0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);

            //dispose the Graphics object
            g.Dispose();
            return newBitmap;
        }

        private static void readMaki(int idx)
        {
            if (files == null)
            {
                files = Directory.EnumerateFiles("anime-faces/" + category_name);
            }

            Bitmap img = new Bitmap(files.ElementAt(idx));
            Bitmap img0 = MakeGrayscale3(img);
            img.Dispose();

            for (int i = 0; i < 48 * 48; i++)
            {
                real_in[idx][i] = img0.GetPixel(i % 48, i / 48).R / 255.0f;
                //real_in[idx][i] = 1.0f - br.ReadByte() / 255.0f;

            }
            img0.Dispose();
        }

        private static void readAkemi(int idx)
        {
            if (vs_files == null)
            {
                vs_files = Directory.EnumerateFiles("anime-faces/" + vs_category_name);
            }

            Bitmap img = new Bitmap(vs_files.ElementAt(idx));
            Bitmap img0 = MakeGrayscale3(img);
            img.Dispose();

            for (int i = 0; i < 48 * 48; i++)
            {
                real_in_2[idx][i] = img0.GetPixel(i % 48, i / 48).R / 255.0f;
                //real_in[idx][i] = 1.0f - br.ReadByte() / 255.0f;

            }
            img0.Dispose();
        }

        public static void imsave(float[] Gz, int i)
        {
            Bitmap bmp = new Bitmap(48, 48);
            for (int x = 0; x < 48; x++)
                for (int y = 0; y < 48; y++)
                {
                    float v = Gz[48 * y + x];

                    if (v < 0)
                        v = 0;

                    if (v > 1)
                        v = 1;

                    if (float.IsNaN(v))
                        v = 0;

                    bmp.SetPixel(x, y, Color.FromArgb((int)(v * 255), (int)(v * 255), (int)(v * 255)));
                }
            bmp.Save($"generated{i}.png");
            bmp.Dispose();
        }

        public static void GAN()
        {
            //Generator genFunc = new Generator();
            //Discriminator disc = new Discriminator();
            NeuralNetwork generator = new NeuralNetwork(new int[] { 100, 128, 256, 512, 1024, 48 * 48 }, new IActivationFunction[] { null, new Tanh(), new Tanh(), new Tanh(), new Tanh(), new SoftStep() }, new CrossEntropy(), new SGD());
            NeuralNetwork discriminator = new NeuralNetwork(new int[] { 48 * 48, 1024, 512, 256, 256, 128, 1 }, new IActivationFunction[] { null, new Tanh(), new Tanh(), new Tanh(), new Tanh(), new Tanh(), new SoftStep() }, new CrossEntropy(), new SGD());

            Random rng = new Random(0);

            real_in = new float[1000][];
            for (int i = 0; i < real_in.Length; i++)
            {
                real_in[i] = new float[48 * 48];
                readMaki(i);
            }

            float max_gen_res = float.NegativeInfinity;

            for (int q = 0; q < 30; q++)
            {
                float[] Gz = null;
                for (int i = 0; i < real_in.Length; i++)
                {

                    for (int j = 0; j < generator_input.Length; j++)
                        generator_input[j] = (float)rng.NextGaussian(0, System.Math.Pow(100, 0.5f)) * 1;

                    Gz = generator.Activate(generator_input);

                    //Train discriminator
                    var (Dz_a, Dz_z) = discriminator.Train(real_in[i]);
                    var (Dz_nabla_w, Dz_nabla_b) = GradientSolver.Solve(discriminator, Dz_a, Dz_z, new Vector(new float[] { 1 }));
                    (discriminator.Weights, discriminator.Biases) = discriminator.Optimizer.Optimize(discriminator.Weights, discriminator.Biases, Dz_nabla_w, Dz_nabla_b, 3f / 100);

                    var (DGz_a, DGz_z) = discriminator.Train(Gz);
                    var (DGz_nabla_w, DGz_nabla_b) = GradientSolver.Solve(discriminator, DGz_a, DGz_z, new Vector(new float[] { 0 }));
                    (discriminator.Weights, discriminator.Biases) = discriminator.Optimizer.Optimize(discriminator.Weights, discriminator.Biases, DGz_nabla_w, DGz_nabla_b, 3f / 100);

                    //Train Generator
                    //Now compute the error from the output of D, but don't change anything until G
                    NeuralNetwork chained = new NeuralNetwork(generator, discriminator);
                    var (Gz_a_ex, Gz_z_ex) = chained.Train(generator_input);
                    var (Gz_nabla_w_ex, Gz_nabla_b_ex) = GradientSolver.Solve(chained, Gz_a_ex, Gz_z_ex, new Vector(new float[] { 1 }));

                    Matrix[] Gz_nabla_w = new ArraySegment<Matrix>(Gz_nabla_w_ex, 0, generator.LayerCount).Array;
                    Vector[] Gz_nabla_b = new ArraySegment<Vector>(Gz_nabla_b_ex, 0, generator.LayerCount).Array;
                    (generator.Weights, generator.Biases) = generator.Optimizer.Optimize(generator.Weights, generator.Biases, Gz_nabla_w, Gz_nabla_b, 3f / 100);

                    if ((i % 25 == 0))
                    {
                        imsave(Gz, i);
                        max_gen_res = DGz_a[2][0];
                    }


                    Console.WriteLine($"[{i}]Discriminator Real: {Dz_a.Last()[0]} Fake: {DGz_a.Last()[0]}");
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
