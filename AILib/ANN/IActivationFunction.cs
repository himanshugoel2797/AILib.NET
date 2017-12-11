using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AILib.Math;

namespace AILib.ANN
{
    public interface IActivationFunction
    {
        Vector Activation(Vector o);
        Vector DerivActivation(Vector o);
    }
}
