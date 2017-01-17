using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace math
{
    public class Indexer
    {
        private int[] _idx;
        private Random _gen;

        public int this[int i]
        {
            get
            {
                return _idx[i];
            }
        }

        public Indexer(int sz)
        {
            _idx = new int[sz];
            for (int i = 0; i < sz; i++)
                _idx[i] = i;
        }

        public void Shuffle()
        {
            Random gen = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < _idx.Length; i++)
            {
                int j = gen.Next(_idx.Length - i - 1);
                int temp = _idx[i];
                _idx[i] = _idx[j];
                _idx[j] = temp;
            }
        }
    }

    public class mlp
    {
        const double Mu = 0.1;

        Matrix[] w;
        Matrix[] a;
        int _inpSz;
        int _hiddenNodes;
        int _hiddenLayers;
        int _outSz;
        public static double Prime(double x)
        {
            return (1.0 - Math.Pow(Math.Tanh(x), 2.0));
        }
        public static double Activate(double x)
        {
            return Math.Tanh(x);
        }
        /*        
        public mlp(
                int inpSz,
                int hiddenNodes,
                int hiddenLayers,
                int outSz)
        {
            _inpSz = inpSz;
            _hiddenNodes = hiddenNodes;
            _hiddenLayers = hiddenLayers;
            _outSz = outSz;
            w = new Matrix[_hiddenLayers];
            a = new Matrix[_hiddenLayers + 1];
            // build weights matrices
            w[0] = new Matrix(_hiddenNodes, _inpSz);
            for (int i = 1; i < _hiddenLayers-1; i++)
                w[i] = new Matrix(_hiddenNodes, _hiddenNodes);
            w[_hiddenLayers - 1] = new Matrix(_outSz, _hiddenNodes);
        }
        */
        public mlp()
        {
            w = new Matrix[2];
            a = new Matrix[3];
            w[0] = new Matrix(2, 2);
            w[1] = new Matrix(1, 2);
            a[0] = null;
            a[1] = null;
        }
        public void InitWeights()
        {
            //for (int i = 0; i < _hiddenLayers; i++)
            //    w[i].FillRandom(-1, 1);
            w[0].FillRandom(-1.0, 1.0);
            w[1].FillRandom(-1.0, 1.0);
        }

        void FF(Matrix x)
        {
            a[0] = x.Transpose();
            a[1] = w[0].Dot(a[0]).Map(Activate);
            a[2] = w[1].Dot(a[1]).Map(Activate);
        }

        double BP(Matrix y)
        {
            Matrix e = y.Transpose() - a[2];
            Matrix d2 = e * (a[2].Map(Prime));
            Matrix d1 = w[1].Transpose().Dot(d2) * (a[1].Map(Prime));
            //
            Matrix dw1 = d2.Dot(a[1].Transpose());
            Matrix dw0 = d1.Dot(a[0].Transpose());
            //
            w[0] = w[0] + dw0.Multiply(Mu);
            w[1] = w[1] + dw1.Multiply(Mu);

            return e.SquaredError();
        }

        public void Train(Matrix x, Matrix y)
        {
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            InitWeights();
            while ((error > 0.001) && ((epoch < 10000)))
            {
                for (int i = 0; i < x.Rows; i++)
                {
                    FF(x.Row(idx[i]));
                    error = BP(y.Row(idx[i]));
                    Console.WriteLine("=====> Error = {0}", error);
                    w[0].Print("w0");
                    w[1].Print("w1");
                }
                idx.Shuffle();
                epoch++;
                if ((epoch % 50) == 0)
                    Console.WriteLine("=====> Error = {0}", error);
            }
        }

        public void Verify(Matrix x, Matrix y)
        {
            for (int i = 0; i < x.Rows; i++)
            {
                FF(x.Row(i));
                Console.WriteLine("target = {0} | actual = {1}", a[1][0, 0], y[i, 0]);
            }
        }
    }
}
