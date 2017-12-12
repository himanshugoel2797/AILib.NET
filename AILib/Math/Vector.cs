using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace AILib.Math
{
    [Serializable]
    public class Vector
    {
        private float[] data;

        public int Length { get; private set; }

        public Vector(int w)
        {
            Length = w;
            data = new float[w];
        }

        public Vector(float[] f)
        {
            Length = f.Length;
            data = f;
        }

        private Vector(Vector<float> v, int len)
        {
            data = new float[len];
            Length = len;
            v.CopyTo(data);
        }

        private Vector()
        {

        }

        public float this[int x]
        {
            get
            {
                return data[x];
            }
            set
            {
                data[x] = value;
            }
        }

        public static Vector operator +(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();
            
            /*
            Vector ret = new Vector(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] + b.data[i];
            }
            return ret;*/

            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for(i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) + new Vector<float>(b.data, i);
                res.CopyTo(ret.data, i);
            }
            for(; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] + b.data[i];
            }
            return ret;
        }

        public static Vector operator -(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            /*
            Vector ret = new Vector(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] - b.data[i];
            }
            return ret;*/
            
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) - new Vector<float>(b.data, i);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] - b.data[i];
            }
            return ret;
        }

        public static Vector operator *(Vector a, float b)
        {
            /*
            Vector ret = new Vector(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] * b;
            }
            return ret;*/

            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) * b;
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] * b;
            }
            return ret;
        }

        public static Vector operator *(Vector a, Vector b)
        {
            return Hadamard(a, b);
        }

        public static Vector operator /(float b, Vector a)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = b * new Vector<float>(1.0f) / new Vector<float>(a.data, i);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = b * 1.0f/a.data[i];
            }
            return ret;
        }

        public static Vector operator /(Vector b, Vector a)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(b.data, i) * new Vector<float>(1.0f) / new Vector<float>(a.data, i);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = b.data[i] * 1.0f / a.data[i];
            }
            return ret;
        }

        public static Vector operator +(float a, Vector b)
        {
            return b + a;
        }

        public static Vector operator +(Vector a, float b)
        {
            /*
            Vector ret = new Vector(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] * b;
            }
            return ret;*/

            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) + new Vector<float>(b);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] + b;
            }
            return ret;
        }

        public static Vector operator -(float a, Vector b)
        {
            Vector ret = new Vector(b.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a) - new Vector<float>(b.data, i);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a - b.data[i];
            }
            return ret;
        }

        public static Vector operator -(Vector a, float b)
        {
            /*
            Vector ret = new Vector(a.Length);
            for (int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] * b;
            }
            return ret;*/

            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) - new Vector<float>(b);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] - b;
            }
            return ret;
        }

        public static Vector Hadamard(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            /*Vector ret = new Vector(a.Length);
            for(int i = 0; i < a.Length; i++)
            {
                ret.data[i] = a.data[i] * b.data[i];
            }
            return ret;*/
            
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = new Vector<float>(a.data, i) * new Vector<float>(b.data, i);
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = a.data[i] * b.data[i];
            }
            return ret;
        }

        public static Vector Abs(Vector a)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = System.Numerics.Vector.Abs(new Vector<float>(a.data, i));
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = System.Math.Abs(a.data[i]);
            }
            return ret;
        }

        public static Vector Min(Vector a, Vector b)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = System.Numerics.Vector.Min(new Vector<float>(a.data, i), new Vector<float>(b.data, i));
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = System.Math.Min(a.data[i], b.data[i]);
            }
            return ret;
        }

        public static Vector Max(Vector a, Vector b)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = System.Numerics.Vector.Max(new Vector<float>(a.data, i), new Vector<float>(b.data, i));
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = System.Math.Max(a.data[i], b.data[i]);
            }
            return ret;
        }

        public static Vector Min(Vector a, float b)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = System.Numerics.Vector.Min(new Vector<float>(a.data, i), new Vector<float>(b));
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = System.Math.Min(a.data[i], b);
            }
            return ret;
        }

        public static Vector Max(Vector a, float b)
        {
            Vector ret = new Vector(a.Length);
            int i = 0;
            int simdLength = Vector<float>.Count;
            for (i = 0; i <= ret.Length - simdLength; i += simdLength)
            {
                var res = System.Numerics.Vector.Max(new Vector<float>(a.data, i), new Vector<float>(b));
                res.CopyTo(ret.data, i);
            }
            for (; i < ret.Length; ++i)
            {
                ret.data[i] = System.Math.Max(a.data[i], b);
            }
            return ret;
        }

        public static implicit operator float[] (Vector a)
        {
            return a.data;
        }

        public static float Dot(Vector a, Vector b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException();

            /*float res = 0;
            for (int i = 0; i < a.Length; i++)
                res += a.data[i] * b.data[i];

            return res;*/

            Vector tmp = Hadamard(a, b);
            float res = 0;

            for (int i = 0; i < tmp.Length; i++)
                res += tmp.data[i];

            return res;
        }
    }
}
