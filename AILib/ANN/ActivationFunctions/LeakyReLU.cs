using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math.GPU;

namespace AILib.ANN.ActivationFunctions
{
    public class LeakyReLU : IActivationFunction
    {
        public float Activation(float o)
        {
            if (o >= 0)
                return o;

            return 0.01f * o;
        }

        public float DerivActivation(float o)
        {
            if (o >= 0)
                return 1;
            return 0.01f;
        }

        public Vector Activation(Vector o)
        {
            Vector n = new Vector(o.Length);
            for (int i = 0; i < n.Length; i++)
                n[i] = Activation(o[i]);
            return n;
        }

        public Vector DerivActivation(Vector o)
        {
            Vector n = new Vector(o.Length);
            for (int i = 0; i < n.Length; i++)
                n[i] = DerivActivation(o[i]);
            return n;
        }
    }
}
