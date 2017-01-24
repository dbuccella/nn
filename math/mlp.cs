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
            _gen = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < sz; i++)
                _idx[i] = i;
        }

        public void Shuffle()
        {
            for (int i = 0; i < _idx.Length; i++)
            {
                int j = _gen.Next(_idx.Length - i - 1);
                int temp = _idx[i];
                _idx[i] = _idx[j];
                _idx[j] = temp;
            }
        }
    }

    public class mlp
    {
        const double Mu = -0.3;

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
            //return ((x > 0.0) ? 1.0 : 0.0);
            return (1.0 - x*x);
        }
        public static double Activate(double x)
        {
            //return ((x > 0.0) ? x : 0.0);
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
                w[i].FillRandom(-0.5, 0.5);
        }

        void FF(Matrix x)
        {
            a[0] = x.Transpose();            
            for (int i = 1; i <= _hiddenLayers; i++)
            {
                a[i] = w[i-1].Dot(a[i-1]).MapNew(Activate);
            }
        }

        public void Predict(Matrix x)
        {
            FF(x);
            a[_hiddenLayers - 1].Transpose().Print("a");
        }

        double BP(Matrix y)
        {
            Matrix err = y.Transpose() - a[_hiddenLayers];
            Matrix[] d = new Matrix[_hiddenLayers + 1];

            // output layer
            d[_hiddenLayers] = err * (a[_hiddenLayers].MapNew(Prime));
            for (int i  = _hiddenLayers - 1; i  >= 1; i--)
            {
                d[i] = w[i].Transpose().Dot(d[i + 1]) * (a[i].MapNew(Prime));
            }
            for (int i = 0; i < _hiddenLayers; i++)
            {
                Matrix dw = d[i + 1].Dot(a[i].Transpose()).Multiply(Mu);
                //dw.Print(String.Format("dw[{0}]", i));
                w[i].Sum(dw);
                //w[i].Sum(d[i + 1].Dot(a[i].Transpose()).Multiply(Mu));
            }
            //err.Print("error");
            return err.SquaredError();
        }

        public void Train(Matrix x, Matrix y, int maxEpochs, ReportProgress pFn = null)
        {
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            int repStep = maxEpochs / 100;
            InitWeights();
            while ((error > _epsilon) && ((epoch < maxEpochs)))
            {
                double epoch_err = 0.0;
                for (int i = 0; i < x.Rows; i++)
                {
                    FF(x.Row(idx[i]));
                    double e = BP(y.Row(idx[i]));
                    epoch_err += e;
                }
                error = epoch_err/x.Rows;
                idx.Shuffle();
                epoch++;
                if ((epoch % repStep) == 0)
                {
                    pFn?.Invoke(100*epoch/ repStep, error);
                    Console.WriteLine("Error= {0}", error);
                }
            }
        }

        public void TrainMB(Matrix x, Matrix y, ReportProgress pFn = null)
        {
            const int BatchSize = 5;

            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            InitWeights();
            while ((error > _epsilon) && ((epoch < 100000)))
            {
                double epoch_err = 0.0;
                int j = 0;
                for (int i = 0; i < x.Rows/BatchSize; i++)
                {
                    // make batch
                    Matrix xb = x.Row(idx[j]);
                    Matrix yb = y.Row(idx[j]);
                    j++;
                    for (int k = 0; k < BatchSize-1; k++)
                    {
                        xb.AppendRows(x, idx[j], 1);
                        yb.AppendRows(y, idx[j], 1);
                        j++;                        
                    }
                    FF(xb);
                    double e = BP(yb);
                    epoch_err += e;
                }

                error = epoch_err / (x.Rows/ BatchSize);
                idx.Shuffle();
                epoch++;
                if ((epoch % 1000) == 0)
                {
                    pFn?.Invoke(100 * epoch / 100000, error);
                }
            }
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
