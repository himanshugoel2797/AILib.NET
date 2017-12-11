using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math;

namespace AILib.ANN
{
    public class GradientSolver
    {
        public static (Matrix[], Vector[]) Solve(NeuralNetwork net, Vector[] a, Vector[] z, Vector expected)
        {
            Vector[] delta = new Vector[net.LayerCount];
            Matrix[] nabla_w = new Matrix[net.LayerCount];
            Vector[] nabla_b = new Vector[net.LayerCount];

            for(int i = 1; i < net.LayerCount; i++)
            {
                delta[i] = new Vector(net.Layers[i]);
                nabla_w[i] = new Matrix(net.Layers[i - 1], net.Layers[i]);
                nabla_b[i] = new Vector(net.Layers[i]);
            }

            delta[net.LayerCount - 1] = net.LossFunction.LossDeriv(net, a[net.LayerCount - 1], z[net.LayerCount - 2], expected);// Vector.Hadamard(, derivActivationFunc(z[layers.Length - 2], layers.Length - 1));
            nabla_b[net.LayerCount - 1] = delta[net.LayerCount - 1];
            nabla_w[net.LayerCount - 1] = (Matrix)delta[net.LayerCount - 1] * Matrix.Transpose((Matrix)a[net.LayerCount - 2]);

            for (int i = net.LayerCount - 2; i > 0; i--)
            {
                Vector derivAct = net.ActivationFunctions[i].DerivActivation(z[i - 1]);
                delta[i] = Vector.Hadamard(Matrix.Transpose(net.Weights[i + 1]) * delta[i + 1], derivAct);

                nabla_b[i] = delta[i];
                nabla_w[i] = (Matrix)delta[i] * Matrix.Transpose((Matrix)a[i - 1]);
            }

            return (nabla_w, nabla_b);
        }
    }
}
