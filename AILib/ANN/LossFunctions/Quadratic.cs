using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN.LossFunctions
{
    public class Quadratic : ILossFunction
    {
        public float Loss(float output, float expectedOutput)
        {
            throw new NotImplementedException();
        }

        public Vector Loss(Vector output, Vector expectedOutput)
        {
            throw new NotImplementedException();
        }

        public float LossDeriv(float output, float expectedOutput)
        {
            return (output - expectedOutput);
        }

        public Vector LossDeriv(NeuralNetwork net, Vector output, Vector logit, Vector expectedOutput)
        {
            Vector n = new Vector(output.Length);

            for (int i = 0; i < n.Length; i++)
                n[i] = LossDeriv(output[i], expectedOutput[i]);

            //return Vector.Hadamard(n, net.ActivationFunctions[net.LayerCount - 1].DerivActivation(logit));
            return null;
        }
    }
}
