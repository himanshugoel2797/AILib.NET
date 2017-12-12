using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AILib.Math.GPU
{
    public class MatrixAccelerator
    {
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            //dispatch an opencl program to do this, caching it if the matrix sizes haven't been multiplied before
            return null;
        }

        /*
        public static void Multiply()//Matrix, Vector
        {

        }

        public static void Multiply()//Vector, Vector
        {

        }

        public static void Multiply()//Vector, Matrix
        {

        }*/
    }
}
