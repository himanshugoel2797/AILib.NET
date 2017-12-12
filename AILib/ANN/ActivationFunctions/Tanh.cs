using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN.ActivationFunctions
{
    public class Tanh : IActivationFunction
    {
        private Vector tanh(Vector o)
        {
            //var i = Vector.Max(Vector.Min(o, 2), -2);
            //Vector i_sq = (i * i);
            //return (i * (i_sq * 4.0f + 15.0f)) / (i_sq * i_sq + i_sq * 9 + 15.0f);
            return null;
        }

        public float Activation(float o)
        {
            return (float)System.Math.Tanh(o);
        }

        public float DerivActivation(float o)
        {
            return (float)(1.0d - System.Math.Pow(System.Math.Tanh(o), 2));
        }

        public Vector Activation(Vector o)
        {
            return tanh(o);
        }

        public Vector DerivActivation(Vector o)
        {
            //Vector a = tanh(o);
            //return 1 - (a * a);
            return null;
        }
    }
}
