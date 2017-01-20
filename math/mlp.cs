using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace math
{
    public delegate void ReportProgress(int pct, object context);

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
        const double Mu = -0.1;

        Matrix[] w;
        Matrix[] a;
        Matrix[] z;
        int _inpSz;
        int _hiddenNodes;
        int _hiddenLayers;
        int _outSz;
        double _epsilon;

        public static double Prime(double x)
        {
            return (1.0 - Math.Pow(Math.Tanh(x), 2.0));
        }
        public static double Activate(double x)
        {
            return Math.Tanh(x);
        }

        public mlp(
                int inpSz,
                int hiddenNodes,
                int hiddenLayers,
                int outSz,
                double epsilon = 0.1)
        {
            _inpSz = inpSz;
            _hiddenNodes = hiddenNodes;
            _hiddenLayers = hiddenLayers;
            _outSz = outSz;
            _epsilon = epsilon;
            //
            w = new Matrix[_hiddenLayers];
            a = new Matrix[_hiddenLayers + 1];
            z = new Matrix[_hiddenLayers + 1];
            //
            // build weights matrices
            w[0] = new Matrix(_hiddenNodes, _inpSz);
            for (int i = 1; i < _hiddenLayers-1; i++)
                w[i] = new Matrix(_hiddenNodes, _hiddenNodes);
            w[_hiddenLayers - 1] = new Matrix(_outSz, _hiddenNodes);
        }

        public mlp()
        {
            w = new Matrix[3];
            a = new Matrix[4];
            z = new Matrix[4];
            w[0] = new Matrix(2, 2);
            w[1] = new Matrix(2, 2);
            w[2] = new Matrix(1, 2);
        }

        public void InitWeights()
        {
            for (int i = 0; i < _hiddenLayers; i++)
                w[i].FillRandom(-0.99, 0.99);
        }

        void FF(Matrix x)
        {
            /*
            z[0] = x.Transpose();
            a[0] = x.Transpose();
            z[1] = w[0].Dot(a[0]);
            a[1] = z[1].MapNew(Activate);
            z[2] = w[1].Dot(a[1]);
            a[2] = z[2].MapNew(Activate);
            z[3] = w[2].Dot(a[2]);
            a[3] = z[3].MapNew(Activate);
            */

            //z[0] = x.Transpose();
            a[0] = x.Transpose();
            
            for (int i = 1; i <= _hiddenLayers; i++)
            {
                //z[i] = w[i-1].Dot(a[i-1]);
                //a[i] = z[i].MapNew(Activate);
                a[i] = w[i-1].Dot(a[i-1]).MapNew(Activate);
            }
        }

        double BP(Matrix y)
        {
            /*
            Matrix e = y.Transpose() - a[3];

            Matrix d3 = e * (z[3].MapNew(Prime));
            Matrix d2 = w[2].Transpose().Dot(d3) * (z[2].MapNew(Prime));
            Matrix d1 = w[1].Transpose().Dot(d2) * (z[1].MapNew(Prime));
            //
            Matrix dw2 = d3.Dot(a[2].Transpose());
            Matrix dw1 = d2.Dot(a[1].Transpose());
            Matrix dw0 = d1.Dot(a[0].Transpose());

            //
            w[0] = w[0] + dw0.Multiply(Mu);
            w[1] = w[1] + dw1.Multiply(Mu);
            w[2] = w[2] + dw2.Multiply(Mu);

            return e.SquaredError();
            */

            Matrix err = y.Transpose() - a[_hiddenLayers];
            Matrix[] d = new Matrix[_hiddenLayers + 1];

            // output layer
            //d[_hiddenLayers] = err * (z[_hiddenLayers].MapNew(Prime));
            d[_hiddenLayers] = err * (a[_hiddenLayers].MapNew((v) => 1.0 - v*v));
            for (int i  = _hiddenLayers - 1; i  >= 1; i--)
            {
                //d[i] = w[i].Transpose().Dot(d[i + 1]) * (z[i].MapNew(Prime));
                d[i] = w[i].Transpose().Dot(d[i + 1]) * (a[i].MapNew((v) => 1.0 - v * v));
            }
            for (int i = 0; i < _hiddenLayers; i++)
            {
                w[i].Sum(d[i + 1].Dot(a[i].Transpose()).Multiply(Mu));
            }
            return err.SquaredError();
        }

        public void Train(Matrix x, Matrix y, ReportProgress pFn = null)
        {
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            double minError = 100000000000.0;
            int minIter = 0;
            InitWeights();
            while ((error > _epsilon) && ((epoch < 50000)))
            {
                double epoch_err = 0.0;
                for (int i = 0; i < x.Rows; i++)
                {
                    FF(x.Row(idx[i]));
                    double e = BP(y.Row(idx[i]));
                    epoch_err += e;
                    /*
                    if (e < minError)
                    {
                        minError = e;
                        minIter = epoch;
                        Console.WriteLine("e = {0} - iter= {1},{2}", e, epoch, i);
                    }
                    */
                    //Console.WriteLine("e = {0}", e);
                    //w[0].Print("w0");
                    //w[1].Print("w1");
                }
                error = epoch_err/x.Rows;
                /*
                if (error < minError)
                {
                    minError = error;
                    minIter = epoch;
                    Console.WriteLine("e = {0} - iter= {1}", error, epoch);
                }
                */
                idx.Shuffle();
                epoch++;
                if ((epoch % 50) == 0)
                {
                    pFn?.Invoke(50000 / epoch, error);
                    //Console.WriteLine("=====> Error = {0}", error);
                }
            }
            //Console.WriteLine("e = {0} - iter= {1}", minError, minIter);
            //Console.WriteLine("Final Error = {0} iter = {1}", error, epoch);
        }

        public void Verify(Matrix x, Matrix y)
        {
            double epsilon = 0.1;
            double matches = 0.0;
            for (int i = 0; i < x.Rows; i++)
            {
                FF(x.Row(i));
                matches = matches + ((Math.Abs(a[1][0, 0] - y[i, 0]) <= epsilon) ? 1.0 : 0.0);
                Console.WriteLine("target = {0} | actual = {1}", a[1][0, 0], y[i, 0]);
            }
            Console.WriteLine("Accuracy= {0:N2}", matches / x.Rows);
        }
    }
}
