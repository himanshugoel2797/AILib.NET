using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math;

namespace AILib.ANN.ActivationFunctions
{
    public class SoftStep : IActivationFunction
    {
        public float Activation(float o)
        {
            return o / (1 + System.Math.Abs(o));
        }

        public float DerivActivation(float o)
        {
            return (float)System.Math.Pow(1 + System.Math.Abs(o), -2);
        }

        public Vector Activation(Vector o)
        {
            return o / (1 + Vector.Abs(o));
        }

        public Vector DerivActivation(Vector o)
        {
            return 1 / ((o + 1.0f) * (o + 1.0f));
        }
    }
}
