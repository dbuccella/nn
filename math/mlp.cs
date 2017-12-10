using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utilities;

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

    public class PlotData
    {
        public double LossX { get; set; }
        public double LossY { get; set; }

        public double DeltaX { get; set; }
        public double DeltaY { get; set; }
    }

    public class TrainResult
    {
        public int Epochs { get; set; }
        public double Error { get; set; }
    }

    public class VerifyResult
    {
        public double Error { get; set; }
        public double Accuracy { get; set; }
        public List<int> Misses { get; set; }
        public VerifyResult()
        {
            Misses = new List<int>();
        }
    }

    public class mlp
    {
        const double Mu = 0.9;

        Matrix[] w;
        Matrix[] a;
        Matrix[] d;
        int _inpSz;
        int _hiddenNodes;
        int _hiddenLayers;
        int _outSz;
        double _epsilon;
        Matrix[] v;

        public double InitLow { get; set; }
        public double InitHigh { get; set; }

        public static double Prime(double x)
        {
            //return ((x >= 0.0) ? 1.0 : 0.01);
            return (1.0 - x*x);
        }
        public static double Activate(double x)
        {
            //return ((x >= 0.0) ? x : 0.01*x);
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
            d = new Matrix[_hiddenLayers + 1];
            v = new Matrix[_hiddenLayers];
            //
            // build weights matrices
            w[0] = new Matrix(_hiddenNodes, _inpSz);
            v[0] = new Matrix(_hiddenNodes, _inpSz);
            for (int i = 1; i < _hiddenLayers - 1; i++)
            {
                w[i] = new Matrix(_hiddenNodes, _hiddenNodes);
                v[i] = new Matrix(_hiddenNodes, _hiddenNodes);
            }
            w[_hiddenLayers - 1] = new Matrix(_outSz, _hiddenNodes);
            v[_hiddenLayers - 1] = new Matrix(_outSz, _hiddenNodes);
            InitLow = -0.01;
            InitHigh = 0.01;
        }

        public void InitWeights()
        {
            for (int i = 0; i < _hiddenLayers; i++)
            {
                w[i].FillRandom(InitLow, InitHigh);
                v[i].FillZero();
            }
        }

        public void Predict(Matrix x)
        {
            FF(x);
            a[_hiddenLayers].Transpose().Print("a");
        }

        void FF(Matrix x)
        {
            a[0] = x.Transpose();            
            for (int i = 1; i <= _hiddenLayers; i++)
            {
                //a[i] = w[i - 1].Dot(a[i - 1]).MapNew(Activate);
                a[i] = w[i - 1].Dot(a[i - 1]).Map(Activate);
            }
        }
        
        public delegate void WeightUpdate(Matrix dx, int i, double alpha);

        public void Standard(Matrix dx, int i, double alpha)
        {
            // Standard
            w[i].Sum(dx.Multiply(alpha));
        }

        public void Momentum(Matrix dx, int i, double alpha)
        {
            // Momentum
            v[i] = v[i].Multiply(Mu) - dx.Multiply(alpha);
            w[i].Sum(v[i]);
        }

        public void Nesterov(Matrix dx, int i, double alpha)
        {
            // Nesterov Momentum
            Matrix vprev = new Matrix(v[i]);
            v[i] = v[i].Multiply(Mu) - dx.Multiply(alpha);
            w[i].Sum(v[i] + (v[i] - vprev).Multiply(Mu));
        }

        double BP(Matrix y, double alpha, WeightUpdate dwFn = null)
        {
            dwFn = (dwFn == null) ? Standard : dwFn;
            //
            Matrix err =  a[_hiddenLayers] - y.Transpose();
            // output layer
            d[_hiddenLayers] = err * (a[_hiddenLayers].MapNew(Prime));
            for (int i  = _hiddenLayers - 1; i  >= 1; i--)
            {
                d[i] = w[i].Transpose().Dot(d[i + 1]) * (a[i].MapNew(Prime));
            }
            // update weights
            for (int i = 0; i < _hiddenLayers; i++)
            {
                Matrix dx = d[i + 1].Dot(a[i].Transpose());
                dwFn(dx, i, alpha);
            }
            return err.SquaredError();
        }

        public double NextRate(double alpha, int k, double l)
        {
            //return alpha / (1.0 + l * k);
            return alpha*(1.0 - l*k);
        }

        public TrainResult Train(Matrix x, Matrix y, int maxEpochs, double alpha, ref bool cancel, ReportProgress pFn = null)
        {
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            int repStep = maxEpochs / 100;
            double l = alpha/(3.0*maxEpochs);
            //double delta = 0.0;
            InitWeights();
            while ((! cancel) && (error > _epsilon) && ((epoch < maxEpochs)))
            {
                double epoch_err = 0.0;
                for (int i = 0; i < x.Rows; i++)
                {
                    FF(x.Row(idx[i]));
                    double e = BP(y.Row(idx[i]), alpha);
                    epoch_err += e;
                }
                error = epoch_err/x.Rows;
                idx.Shuffle();
                epoch++;
                //alpha = NextRate(alpha, epoch, l);
                //PrintModel();
                if ((epoch < 100) || ((epoch % (repStep)) == 0))
                {
                    //delta = 0.0;
                    /*
                    for (int i = _hiddenLayers; i >= 1; i--)
                    {
                        delta += d[i].SquaredError();
                    }
                    delta /= (_hiddenLayers - 1);
                    */
                    /*
                    for (int i = 0; i<_hiddenLayers; i++)
                    {
                        delta += v[i].SquaredError();
                    }
                    delta /= _hiddenLayers;
                    delta *= alpha;
                    */
                    PlotData p = new PlotData();
                    //p.DeltaX = epoch;
                    //p.DeltaY = delta;
                    p.LossX = epoch;
                    p.LossY = error;
                    pFn?.Invoke(0, p);
                    //Console.WriteLine("Error= {0}", error);
                    //PrintModel();
                }
            }
            TrainResult r = new TrainResult();
            r.Epochs = epoch;
            r.Error = error;
            return r;
        }

        static Matrix MakeBatch(Matrix x, int start, int batchSize, Indexer idx)
        {
            // make batch
            Matrix xb = x.Row(idx[start]);
            for (int k = start + 1; k < (start + batchSize); k++)
                xb.AppendRows(x, idx[k], 1);
            return xb;
        }

        public TrainResult Train2(Matrix x, Matrix y, int maxEpochs, double alpha, int batchSize, ref bool cancel, ReportProgress pFn = null)
        {
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            int repStep = maxEpochs / 100;
            double l = alpha / (3.0 * maxEpochs);
            InitWeights();
            while ((! cancel) && (error > _epsilon) && ((epoch < maxEpochs)))
            {
                double epoch_err = 0.0;
                for (int i = 0; i < x.Rows / batchSize; i++)
                {
                    Matrix xb = MakeBatch(x, i * batchSize, batchSize, idx);
                    Matrix yb = MakeBatch(y, i * batchSize, batchSize, idx);
                    FF(xb);
                    double e = BP(yb, alpha, Nesterov);
                    epoch_err += e;
                }
                error = epoch_err / x.Rows;
                idx.Shuffle();
                epoch++;
                alpha = NextRate(alpha, epoch, l);
                //PrintModel();
                if ((epoch < 100) || ((epoch % (repStep)) == 0))
                {
                    PlotData p = new PlotData();
                    //p.DeltaX = epoch;
                    //p.DeltaY = delta;
                    p.LossX = epoch;
                    p.LossY = error;
                    pFn?.Invoke(0, p);
                }
            }
            TrainResult r = new TrainResult();
            r.Epochs = epoch;
            r.Error = error;
            return r;
        }

        public void PrintModel()
        {
            for (int i = 0; i < _hiddenLayers; i++)
            {
                w[i].Print(String.Format("w[{0}]", i));
            }
            for (int i = 0; i < _hiddenLayers; i++)
            {
                if (a[i] != null)
                    a[i].Print(String.Format("a[{0}]", i));
            }
        }

        public void TrainMB(Matrix x, Matrix y, int maxEpochs, int batchSize, double alpha, ReportProgress pFn = null)
        {
            int repStep = maxEpochs / 100;
            Indexer idx = new Indexer(x.Rows);
            int epoch = 0;
            double error = 1.0;
            InitWeights();
            while ((error > _epsilon) && ((epoch < maxEpochs)))
            {
                double epoch_err = 0.0;
                int j = 0;
                for (int i = 0; i < x.Rows/batchSize; i++)
                {
                    // make batch
                    Matrix xb = x.Row(idx[j]);
                    Matrix yb = y.Row(idx[j]);
                    j++;
                    for (int k = 0; k < batchSize-1; k++)
                    {
                        xb.AppendRows(x, idx[j], 1);
                        yb.AppendRows(y, idx[j], 1);
                        j++;                        
                    }
                    //
                    FF(xb);
                    double e = BP(yb, alpha);
                    epoch_err += e;
                }
                error = epoch_err / (x.Rows/ batchSize);
                //Console.WriteLine("error= {0}", error);
                idx.Shuffle();
                epoch++;
                if ((epoch % (repStep/4)) == 0)
                {
                    pFn?.Invoke(epoch , error);
                    //pFn?.Invoke(100 * epoch / repStep, error);
                    Console.WriteLine("error= {0}", error);
                }
            }
        }


        public void Verify2(Matrix x, Matrix y, double epsilon = 0.1)
        {
            int matches = 0;
            for (int i = 0; i < x.Rows; i++)
            { 
                FF(x.Row(i));
                Matrix e = y.Row(i).Transpose() - a[_hiddenLayers];
                double err = e.SquaredError();
                if (err <= epsilon)
                {
                    matches++;
                }
                else
                {
                    Console.WriteLine("==== Miss ====");
                    y.Row(i).Print("Actual");
                    a[_hiddenLayers].Transpose().Print("target");
                }
            }
            Console.WriteLine("Accuracy= {0:N2}", (1.0* matches) / x.Rows);
        }

        public VerifyResult Verify(Matrix x, Matrix y, double epsilon = 0.15)
        {
            VerifyResult r = new VerifyResult();
            Logger log= new Logger("./verify.log");
            log.Log("verifying results");
            int matches = 0;
            FF(x);
            Matrix e = a[_hiddenLayers] - y.Transpose();
            r.Error = e.SquaredError()/e.Rows;
            for (int i = 0; i < x.Rows; i++)
            {
                bool match = true;
                for (int k = 0; (k < e.Rows) && match; k++)
                {
                    match = match && (Math.Abs(e[k, i]) <= epsilon);
                }
                if (match)
                {
                    matches++;
                }
                else
                {
                    r.Misses.Add(i);
                    log.Log("Miss");
                    log.Log(y.Row(i).PrintFn, "actual");
                    log.Log(a[_hiddenLayers].Column(i).Transpose().PrintFn, "target");
                }
            }
            r.Accuracy = (1.0 * matches) / x.Rows;
            log.Log("Samples= {0} Errors = {1}", x.Rows, r.Misses.Count);
            log.Log("Accuracy= {0:N2}", r.Accuracy);
            log.Flush();
            return r;
        }
    }
}
